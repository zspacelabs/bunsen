//! # Stage 3: the normalized lag search, on device.
//!
//! Maps one hop's excitation history to the two half-hop correlation slots it
//! contributes, plus the energies that weight them:
//!
//! ```text
//! exc [rows, exc_len]  ->  xcorr [rows, 2, max_period],  energy [rows, 2]
//! ```
//!
//! **Stateless given the history.** Everything this stage needs is inside the
//! `exc_len`-sample window the excitation stage maintains, so it carries
//! nothing of its own — the correlation *ring* belongs to the tracker, which
//! consumes a sliding window of these slots. That is why this stage can be
//! built and tested before the excitation stage exists: drive the host,
//! read its excitation buffer out, and correlate that.
//!
//! ## Per half-hop
//!
//! ```text
//! inst[lag] = Σ_{j<half} exc[max_period + base + j] · exc[base + lag + j]
//! ref_e     = Σ_{j<half} exc²[max_period + base + j]
//! lag_e[0]  = Σ_{j<half} exc²[base + j]
//! lag_e[l]  = max(lag_e[l-1] − exc²[base+l-1], 0) + exc²[base+l+half-1]
//! xcorr[lag]= 2·inst[lag] / max(lag_e[lag] + (1 + ref_e), 1e-12)
//! ```
//!
//! Normalizing by the energy *under the lagged window* is what makes this a
//! correlation coefficient rather than a raw correlation; the `1 +` keeps
//! silence from amplifying noise into a confident-looking peak.
//!
//! ## Three details that shape the implementation
//!
//! **The correlation is a shift-and-accumulate, not a matmul.** Written as
//! `[max_period, half]` windows times a reference vector it would materialize
//! a `rows × 64 × 32` intermediate — tens of megabytes over a long sequence.
//! Written as `half` broadcast multiply-adds over `[rows, max_period]` slices
//! it is ~30x smaller, *and* it accumulates in the reference's order.
//!
//! **The lagged energy is not a `cumsum`.** The `max(…, 0)` sits inside the
//! recurrence, which makes it non-associative: a prefix-sum difference gives a
//! different answer whenever the clamp fires. It stays a `max_period`-step
//! sequential loop — but batched across every row, so it is ~190 tiny kernels
//! for a whole sequence rather than per hop.
//!
//! **The octave suppression vectorizes exactly.** The reference walks lags in
//! order and rescales `xcorr[lag]` in place while reading three neighbours at
//! roughly `lag/2 + 32`. Those reads are *always strictly ahead* of the write
//! (tightest margin 8, at `lag = 46`), so no iteration ever observes a value a
//! previous one modified, and reading the unmodified array is equivalent
//! rather than approximate. [`SHARPEN_IS_WRITE_SAFE`] pins that.

use burn::{
    config::Config,
    prelude::*,
};

use super::super::coeff::{
    MAX_PERIOD_16KHZ,
    MIN_PERIOD_16KHZ,
    PROC_RESAMPLE_RATE,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
    WithOkOrPanic,
};

/// The number of half-hop slots each hop contributes.
pub const SUBS_PER_HOP: usize = 2;

/// Config for [`PitchCorrelate`].
///
/// Defaults match the ten-vad front end: a 256-sample hop decimated to 4 kHz,
/// giving a 64-lag search over 32-sample half-hops.
#[derive(Config, Debug, Copy)]
pub struct PitchCorrelateConfig {
    /// The hop size, in samples at 16 kHz.
    #[config(default = "256")]
    pub hop_size: usize,
}

impl PitchCorrelateConfig {
    /// The longest candidate period, in samples at the correlation rate.
    pub fn max_period(&self) -> usize {
        MAX_PERIOD_16KHZ / PROC_RESAMPLE_RATE
    }

    /// The shortest candidate period, in samples at the correlation rate.
    pub fn min_period(&self) -> usize {
        MIN_PERIOD_16KHZ / PROC_RESAMPLE_RATE
    }

    /// The correlation window length: half a decimated hop.
    pub fn half_hop(&self) -> usize {
        self.hop_size / (PROC_RESAMPLE_RATE * SUBS_PER_HOP)
    }

    /// The excitation history length this stage reads.
    pub fn exc_len(&self) -> usize {
        self.max_period() + self.hop_size.div_ceil(PROC_RESAMPLE_RATE) + 1
    }

    /// How many lags the octave suppression rescales.
    pub fn sharpen_len(&self) -> usize {
        self.max_period() - SUBS_PER_HOP * self.min_period()
    }

    /// Validates the geometry.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the hop does not divide into whole half-hops
    /// at the decimated rate, or if the resulting search range is empty.
    pub fn validate(&self) -> BunsenResult<()> {
        let divisor = PROC_RESAMPLE_RATE * SUBS_PER_HOP;
        if self.hop_size == 0 || !self.hop_size.is_multiple_of(divisor) {
            return Err(BunsenError::Invalid(format!(
                "PitchCorrelate hop_size ({}) must be a non-zero multiple of {divisor}",
                self.hop_size,
            )));
        }
        if self.max_period() <= SUBS_PER_HOP * self.min_period() {
            return Err(BunsenError::Invalid(format!(
                "PitchCorrelate period range ({}..{}) leaves nothing to sharpen",
                self.min_period(),
                self.max_period(),
            )));
        }
        Ok(())
    }

    /// The three neighbour indices the octave suppression compares each lag
    /// against, as `(a, b, c)` vectors of length
    /// [`sharpen_len`](Self::sharpen_len).
    ///
    /// Exposed so a test can check the write-safety invariant the vectorized
    /// form depends on.
    pub fn to_vec_sharpen_indices(&self) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let max_period = self.max_period();
        let n = self.sharpen_len();
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        for lag in 0..n {
            a.push(((max_period + lag) / 2) as i32);
            b.push(((max_period + lag + 2) / 2) as i32);
            c.push(((max_period + lag - 1) / 2) as i32);
        }
        (a, b, c)
    }

    /// Builds the stage.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<PitchCorrelate<B>> {
        self.validate()?;
        let (a, b, c) = self.to_vec_sharpen_indices();

        Ok(PitchCorrelate {
            hop_size: self.hop_size,
            max_period: self.max_period(),
            min_period: self.min_period(),
            half_hop: self.half_hop(),
            exc_len: self.exc_len(),
            sharpen_a: Tensor::from_ints(a.as_slice(), device),
            sharpen_b: Tensor::from_ints(b.as_slice(), device),
            sharpen_c: Tensor::from_ints(c.as_slice(), device),
        })
    }

    /// Builds the stage, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PitchCorrelate<B> {
        self.try_init(device).ok_or_panic()
    }
}

/// The normalized lag-search stage.
///
/// Stateless, so one instance serves any number of streams. Built by
/// [`PitchCorrelateConfig::try_init`].
#[derive(Debug, Clone)]
pub struct PitchCorrelate<B: Backend> {
    hop_size: usize,
    max_period: usize,
    min_period: usize,
    half_hop: usize,
    exc_len: usize,

    sharpen_a: Tensor<B, 1, Int>,
    sharpen_b: Tensor<B, 1, Int>,
    sharpen_c: Tensor<B, 1, Int>,
}

impl<B: Backend> PitchCorrelate<B> {
    /// The hop size, in samples at 16 kHz.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// The longest candidate period, at the correlation rate.
    pub fn max_period(&self) -> usize {
        self.max_period
    }

    /// The shortest candidate period, at the correlation rate.
    pub fn min_period(&self) -> usize {
        self.min_period
    }

    /// The excitation history length this stage reads.
    pub fn exc_len(&self) -> usize {
        self.exc_len
    }

    /// Correlates both half-hops of every row.
    ///
    /// # Arguments
    /// * `exc`: `[rows, exc_len]` decimated excitation history, newest last —
    ///   the state the excitation stage holds after appending a hop.
    ///
    /// # Returns
    /// `(xcorr, energy)`, shaped `[rows, 2, max_period]` and `[rows, 2]`. The
    /// slot axis is in stream order: index `0` is the earlier half-hop.
    ///
    /// # Panics
    /// If `exc`'s trailing axis is not [`exc_len`](Self::exc_len).
    pub fn forward(
        &self,
        exc: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [rows, len] = exc.dims();
        assert_eq!(
            len, self.exc_len,
            "PitchCorrelate expects {} excitation samples",
            self.exc_len,
        );

        let squared = exc.clone() * exc.clone();

        let mut slots = Vec::with_capacity(SUBS_PER_HOP);
        let mut energies = Vec::with_capacity(SUBS_PER_HOP);
        for sub in 0..SUBS_PER_HOP {
            let (xcorr, energy) = self.correlate_sub(&exc, &squared, sub, rows);
            slots.push(xcorr.unsqueeze_dim::<3>(1));
            energies.push(energy);
        }

        (Tensor::cat(slots, 1), Tensor::cat(energies, 1))
    }

    /// One half-hop's correlations and reference energy.
    fn correlate_sub(
        &self,
        exc: &Tensor<B, 2>,
        squared: &Tensor<B, 2>,
        sub: usize,
        rows: usize,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let base = sub * self.half_hop;
        let max_period = self.max_period;
        let half = self.half_hop;
        let device = exc.device();

        // Shift-and-accumulate, in the reference's `j` order. Each step is one
        // broadcast multiply-add over `[rows, max_period]`.
        let mut inst: Tensor<B, 2> = Tensor::zeros([rows, max_period], &device);
        for j in 0..half {
            let at = max_period + base + j;
            let reference = exc.clone().slice_dim(1, at as isize..(at + 1) as isize);
            let lagged = exc
                .clone()
                .slice_dim(1, (base + j) as isize..(base + j + max_period) as isize);
            inst = inst + lagged * reference;
        }

        // The reference window's energy, which is also this slot's weight in
        // the tracker.
        let reference_energy = squared
            .clone()
            .slice_dim(
                1,
                (max_period + base) as isize..(max_period + base + half) as isize,
            )
            .sum_dim(1);

        let lagged_energy = self.sliding_energy(squared, base);

        let denominator =
            (lagged_energy + reference_energy.clone().add_scalar(1.0f32)).clamp_min(1e-12f32);
        let xcorr = inst.mul_scalar(2.0f32) / denominator;

        (self.suppress_octaves(xcorr), reference_energy)
    }

    /// The energy under each lagged window, as a clamped sliding sum.
    ///
    /// Sequential by necessity: the `max(…, 0)` inside the recurrence makes it
    /// non-associative, so no prefix-sum reformulation reproduces it.
    ///
    /// Written into a preallocated `[rows, max_period]` buffer rather than
    /// collected and concatenated: there is no same-shaped input to
    /// cannibalize here, but `slice_assign` still beats a 64-way `cat` of
    /// `[rows, 1]` slices.
    fn sliding_energy(
        &self,
        squared: &Tensor<B, 2>,
        base: usize,
    ) -> Tensor<B, 2> {
        let half = self.half_hop;
        let rows = squared.dims()[0];

        let mut running = squared
            .clone()
            .slice_dim(1, base as isize..(base + half) as isize)
            .sum_dim(1);

        let mut out: Tensor<B, 2> = Tensor::zeros([rows, self.max_period], &squared.device());
        out = out.slice_assign(s![.., 0..1], running.clone());

        for lag in 1..self.max_period {
            let leaving = squared
                .clone()
                .slice_dim(1, (base + lag - 1) as isize..(base + lag) as isize);
            let entering = squared.clone().slice_dim(
                1,
                (base + lag + half - 1) as isize..(base + lag + half) as isize,
            );
            running = (running - leaving).clamp_min(0.0f32) + entering;
            out = out.slice_assign(s![.., lag..lag + 1], running.clone());
        }

        out
    }

    /// Discounts lags that fail to clearly beat their own half-lag
    /// neighbourhood, which is where period doubling shows up.
    ///
    /// Vectorized against the unmodified input, which is exact: see the module
    /// docs and [`SHARPEN_IS_WRITE_SAFE`].
    fn suppress_octaves(
        &self,
        xcorr: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let sharpen_len = self.max_period - SUBS_PER_HOP * self.min_period;

        let rival = xcorr
            .clone()
            .select(1, self.sharpen_a.clone())
            .max_pair(xcorr.clone().select(1, self.sharpen_b.clone()))
            .max_pair(xcorr.clone().select(1, self.sharpen_c.clone()));

        let head = xcorr.clone().slice_dim(1, 0..sharpen_len as isize);
        let doubled = head.clone().lower(rival.mul_scalar(1.1f32));
        let discounted = head.clone().mul_scalar(0.8f32);

        Tensor::cat(
            vec![
                head.mask_where(doubled, discounted),
                xcorr.slice_dim(1, sharpen_len as isize..),
            ],
            1,
        )
    }
}

/// Pins the invariant the vectorized octave suppression relies on.
///
/// The reference rescales `xcorr[lag]` in place while reading neighbours; this
/// is `true` only if every such read is strictly ahead of the write, so that
/// no iteration can observe a value an earlier one changed.
pub const SHARPEN_IS_WRITE_SAFE: bool = {
    let max_period = MAX_PERIOD_16KHZ / PROC_RESAMPLE_RATE;
    let min_period = MIN_PERIOD_16KHZ / PROC_RESAMPLE_RATE;
    let sharpen_len = max_period - SUBS_PER_HOP * min_period;

    let mut lag = 0;
    let mut safe = true;
    while lag < sharpen_len {
        // The smallest of the three neighbour indices.
        let lowest = (max_period + lag - 1) / 2;
        if lowest <= lag || lowest >= max_period {
            safe = false;
        }
        lag += 1;
    }
    safe
};

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::{
        super::super::{
            TenVadPitchEstimator,
            TenVadPitchScalarSource,
        },
        *,
    };
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    const HOP: usize = 256;
    const N_BINS: usize = 513;

    fn config() -> PitchCorrelateConfig {
        PitchCorrelateConfig::new()
    }

    /// A glottal-like pulse train at the reference's int16 scale.
    fn pulse_hop(
        f0: f32,
        at: usize,
    ) -> Vec<f32> {
        let period = 16000.0 / f0;
        (0..HOP)
            .map(|i| {
                let pos = (at + i) as f32 % period;
                8000.0 * (-pos / (period * 0.08)).exp()
            })
            .collect()
    }

    /// A plausible hop spectrum; the correlation stage only needs the
    /// excitation to have structure, not for the spectrum to be its transform.
    fn spectrum(step: usize) -> Vec<f32> {
        (0..N_BINS)
            .map(|k| {
                let k = k as f32;
                1e7 * (-k / 70.0).exp() * (1.0 + 0.4 * (k * 0.05 + step as f32 * 0.3).sin())
            })
            .collect()
    }

    /// Drives the host estimator and captures, per hop, the excitation history
    /// it correlated and the two slots it produced.
    fn host_reference(steps: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut est = TenVadPitchEstimator::new();
        let slots = est.slots();

        let mut exc = Vec::new();
        let mut xcorr = Vec::new();
        let mut energy = Vec::new();

        for step in 0..steps {
            est.frame_pitch(&pulse_hop(150.0, step * HOP), &spectrum(step));

            exc.extend_from_slice(est.exc_buf());
            // After the call, stream slots `slots-2` and `slots-1` are the two
            // this hop just wrote, in order.
            for sub in 0..SUBS_PER_HOP {
                xcorr.extend_from_slice(est.xcorr_slot(slots - SUBS_PER_HOP + sub));
                energy.push(est.frm_weight()[slots - SUBS_PER_HOP + sub]);
            }
        }
        (exc, xcorr, energy)
    }

    #[test]
    fn test_config_meta() {
        let cfg = config();
        assert_eq!(cfg.hop_size, 256);
        assert_eq!(cfg.max_period(), 64);
        assert_eq!(cfg.min_period(), 8);
        assert_eq!(cfg.half_hop(), 32);
        assert_eq!(cfg.exc_len(), 129);
        assert_eq!(cfg.sharpen_len(), 48);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        assert!(config().with_hop_size(0).validate().is_err());
        // Not a whole number of half-hops once decimated.
        assert!(config().with_hop_size(100).validate().is_err());
        assert!(config().with_hop_size(256).validate().is_ok());
    }

    #[test]
    fn test_init_meta_matches_config() {
        let device = Default::default();
        let stage: PitchCorrelate<B> = config().init(&device);
        assert_eq!(stage.hop_size(), 256);
        assert_eq!(stage.max_period(), 64);
        assert_eq!(stage.min_period(), 8);
        assert_eq!(stage.exc_len(), 129);
    }

    #[test]
    fn test_octave_suppression_reads_are_write_safe() {
        // The whole vectorized form rests on this. Checked as a const, and
        // re-derived here so a geometry change cannot quietly invalidate it.
        assert!(SHARPEN_IS_WRITE_SAFE);

        let cfg = config();
        let (a, b, c) = cfg.to_vec_sharpen_indices();
        for lag in 0..cfg.sharpen_len() {
            let lowest = a[lag].min(b[lag]).min(c[lag]);
            assert!(
                lowest as usize > lag,
                "lag {lag} reads neighbour {lowest}, which it may already have written",
            );
            let highest = a[lag].max(b[lag]).max(c[lag]) as usize;
            assert!(highest < cfg.max_period(), "lag {lag} reads out of range");
        }
    }

    #[test]
    fn test_forward_matches_host_stage() {
        // The differential test: feed the device exactly the excitation the
        // host correlated, and compare both outputs.
        let device = Default::default();
        let stage: PitchCorrelate<B> = config().init(&device);
        let steps = 12;
        let (exc, want_xcorr, want_energy) = host_reference(steps);

        let exc_t =
            Tensor::<B, 1>::from_floats(exc.as_slice(), &device).reshape([steps, stage.exc_len()]);
        let (got_xcorr, got_energy) = stage.forward(exc_t);

        assert_eq!(got_xcorr.dims(), [steps, SUBS_PER_HOP, stage.max_period()]);
        assert_eq!(got_energy.dims(), [steps, SUBS_PER_HOP]);

        got_xcorr.to_data().assert_approx_eq::<f32>(
            &TensorData::new(want_xcorr, [steps, SUBS_PER_HOP, stage.max_period()]),
            Tolerance::relative(1e-4),
        );
        got_energy.to_data().assert_approx_eq::<f32>(
            &TensorData::new(want_energy, [steps, SUBS_PER_HOP]),
            Tolerance::relative(1e-4),
        );
    }

    #[test]
    fn test_forward_batches_rows_independently() {
        let device = Default::default();
        let stage: PitchCorrelate<B> = config().init(&device);
        let steps = 6;
        let (exc, _, _) = host_reference(steps);

        let batched =
            Tensor::<B, 1>::from_floats(exc.as_slice(), &device).reshape([steps, stage.exc_len()]);
        let (batched_xcorr, batched_energy) = stage.forward(batched.clone());

        for row in 0..steps {
            let solo = batched
                .clone()
                .slice_dim(0, row as isize..(row + 1) as isize);
            let (solo_xcorr, solo_energy) = stage.forward(solo);

            batched_xcorr
                .clone()
                .slice_dim(0, row as isize..(row + 1) as isize)
                .to_data()
                .assert_approx_eq::<f32>(&solo_xcorr.to_data(), Tolerance::permissive());
            batched_energy
                .clone()
                .slice_dim(0, row as isize..(row + 1) as isize)
                .to_data()
                .assert_approx_eq::<f32>(&solo_energy.to_data(), Tolerance::permissive());
        }
    }

    #[test]
    fn test_silence_is_quiet_and_finite() {
        // The `1 +` in the denominator is what keeps an all-zero excitation
        // from dividing zero by zero into a confident peak.
        let device = Default::default();
        let stage: PitchCorrelate<B> = config().init(&device);

        let (xcorr, energy) = stage.forward(Tensor::<B, 2>::zeros([1, stage.exc_len()], &device));

        let flat: Vec<f32> = xcorr.to_data_as::<f32>().to_vec_as::<f32>().unwrap();
        for (i, v) in flat.iter().enumerate() {
            assert!(v.is_finite(), "xcorr[{i}] = {v}");
            assert_eq!(*v, 0.0, "silence should correlate to zero, got {v}");
        }
        assert_eq!(energy.sum().into_scalar().elem::<f32>(), 0.0);
    }

    #[test]
    fn test_a_periodic_excitation_peaks_at_its_period() {
        // A sanity check that the stage measures what it claims to: an
        // impulse train at period p should score highest near lag p.
        let device = Default::default();
        let stage: PitchCorrelate<B> = config().init(&device);
        let period = 20usize;

        let exc: Vec<f32> = (0..stage.exc_len())
            .map(|i| if i % period == 0 { 1000.0 } else { 0.0 })
            .collect();

        let exc_t =
            Tensor::<B, 1>::from_floats(exc.as_slice(), &device).reshape([1, stage.exc_len()]);
        let (xcorr, _) = stage.forward(exc_t);

        // Slot 0, searching lags that are multiples of the period.
        let row: Vec<f32> = xcorr
            .slice_dim(1, 0..1)
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        let best = row
            .iter()
            .enumerate()
            .skip(stage.min_period())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // The reference window starts at `max_period + base`, not at the start
        // of the buffer, so two impulse trains align when
        // `max_period - lag ≡ 0 (mod period)` — not when `lag` itself is a
        // multiple of the period.
        assert_eq!(
            best % period,
            stage.max_period() % period,
            "peak at lag {best} does not align the reference window against \
             a pulse (period {period}, reference offset {})",
            stage.max_period(),
        );
    }
}
