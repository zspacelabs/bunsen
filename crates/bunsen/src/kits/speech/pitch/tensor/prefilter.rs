//! # Stage 1: the whitening filter design, on device.
//!
//! Maps `[rows, n_bins]` bin powers to `[rows, lpc_order]` whitening filter
//! coefficients — the tensor form of
//! [`HostPitchEstimator`](super::super::HostPitchEstimator)'s pre-filter
//! stage.
//!
//! **This stage carries no state.** It reads one hop's spectrum and writes one
//! hop's filter, with nothing threaded between hops, so `rows` can be a whole
//! `steps × batch` sequence flattened into the row axis and the entire stage
//! runs in one pass. That is what makes it a coefficient-only object with no
//! `*Context`, structurally like
//! a filterbank.
//!
//! ```text
//! bands  = bin_power @ BANDS                      # [rows, 18]
//! ly     = clamped log10(bands + 1e-2)            # a 18-step scan
//! cep    = ly @ DCT                               # [rows, 18]
//! gain   = 10^(cep @ DCTᵀ) * BAND_LPC_COMP        # [rows, 18]
//! ac     = gain @ AC_FROM_BANDS                   # [rows, 17]
//! ac[0] += ac[0]*1e-4 + dc0_bias;  ac[i] *= lagwin[i]
//! lpc    = levinson(ac)                           # [rows, 16]
//! ```
//!
//! ## The two parts that are not matmuls
//!
//! **The log-compression clamp is a coupled scan** across the 18 bands: each
//! band's floor depends on the running peak and on a decaying follower, and
//! both are updated from the *clamped* value. It is only 18 steps, so it is
//! unrolled — but those 18 steps are batched across every row of the sequence
//! at once, so the cost is 18 tiny kernels per call, not per hop.
//!
//! **The Levinson-Durbin recursion has a data-dependent early exit.** The
//! reference breaks out once the residual error drops 30 dB below lag zero,
//! leaving the remaining coefficients at their zero initializer. A batched
//! tensor version cannot branch per row, so it runs all 16 steps and *freezes*
//! each row under a mask once its condition trips. Two details make that
//! faithful:
//!
//! * The reference checks **after** completing an iteration, so iteration `i`
//!   always commits and only `i+1..` are skipped. The freeze is therefore
//!   applied after the update, and the mask is recomputed after the freeze.
//! * The freeze needs no separate "sticky" bookkeeping: once a row's `error` is
//!   frozen below the threshold it stays below it, so re-deriving the mask from
//!   `error` each step is already monotone.
//!
//! The reference's other guard — `ac[0] == 0`, where it returns all zeros — is
//! unreachable here. `dc0_bias` adds an absolute floor of `window/12/38` to lag
//! zero before the solve, so `ac[0] >= 1.68` regardless of input, and the
//! threshold is always strictly positive.

use burn::{
    config::Config,
    prelude::*,
};

use super::{
    super::{
        coeff::NB_BANDS,
        lpc::{
            CELT_LPC_BAIL_RATIO,
            dc0_bias,
        },
    },
    tables::{
        PitchTables,
        PitchTablesConfig,
    },
};
use crate::{
    errors::{
        BunsenResult,
        WithOkOrPanic,
    },
    ops::signal::levinson_durbin_batched,
};

/// Config for [`PitchPrefilter`].
#[derive(Config, Debug)]
pub struct PitchPrefilterConfig {
    /// The projection-table geometry.
    #[config(default = "PitchTablesConfig::new()")]
    pub tables: PitchTablesConfig,
}

impl PitchPrefilterConfig {
    /// Validates the geometry.
    ///
    /// # Errors
    ///
    /// See [`PitchTablesConfig::validate`].
    pub fn validate(&self) -> BunsenResult<()> {
        self.tables.validate()
    }

    /// The number of frequency bins this stage expects.
    pub fn n_bins(&self) -> usize {
        self.tables.n_bins()
    }

    /// Builds the stage, uploading its tables.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<PitchPrefilter<B>> {
        Ok(PitchPrefilter {
            tables: self.tables.try_init(device)?,
        })
    }

    /// Builds the stage, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PitchPrefilter<B> {
        self.try_init(device).ok_or_panic()
    }
}

/// The whitening-filter design stage.
///
/// Stateless, so one instance serves any number of streams. Built by
/// [`PitchPrefilterConfig::try_init`].
#[derive(Debug, Clone)]
pub struct PitchPrefilter<B: Backend> {
    /// The fixed projection tables.
    pub tables: PitchTables<B>,
}

impl<B: Backend> PitchPrefilter<B> {
    /// The number of frequency bins this stage expects.
    pub fn n_bins(&self) -> usize {
        self.tables.n_bins()
    }

    /// The FFT size the bin powers come from.
    pub fn fft_size(&self) -> usize {
        self.tables.fft_size()
    }

    /// Designs a whitening filter per row.
    ///
    /// # Arguments
    /// * `bin_power`: `[rows, n_bins]` bin powers, **un-normalized** — the same
    ///   values the pitch branch receives, before the mel branch's `1 /
    ///   32768^2` division.
    ///
    /// # Returns
    /// `[rows, lpc_order]` whitening filter coefficients.
    ///
    /// # Panics
    /// If `bin_power`'s trailing axis is not [`n_bins`](Self::n_bins).
    pub fn forward(
        &self,
        bin_power: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let n_bins = bin_power.dims()[1];
        assert_eq!(
            n_bins,
            self.n_bins(),
            "PitchPrefilter expects {} bins",
            self.n_bins(),
        );

        // [rows, nb_bands]
        let bands = bin_power.matmul(self.tables.bands.clone());
        let ly = self.log_compress(bands);

        // Round-trip through the cepstrum, which is where the envelope gets
        // smoothed; the same matrix serves both directions.
        let cepstrum = ly.matmul(self.tables.dct.clone());
        let log_gain = cepstrum.matmul(self.tables.dct.clone().transpose());

        // 10^x, then the per-band compensation.
        let gain = log_gain.mul_scalar(core::f32::consts::LN_10).exp()
            * self.tables.band_lpc_comp.clone().unsqueeze::<2>();

        // [rows, lpc_order + 1]
        let ac = gain.matmul(self.tables.ac_from_bands.clone());
        let ac = self.apply_noise_floor(ac);

        levinson_durbin_batched(ac, Some(CELT_LPC_BAIL_RATIO))
    }

    /// The clamped log compression, as an unrolled 18-step scan.
    ///
    /// Nothing may sit more than 8 decades below the running peak, nor fall
    /// faster than 2.5 decades per band. Both trackers update from the
    /// *clamped* value, which is what couples them.
    ///
    /// burn exposes only a natural logarithm, so `log10` is `log(x)/ln(10)`.
    /// That differs from the host's `f32::log10` by a ULP or so — well inside
    /// this stage's tolerance, and the `10^x` on the way back out is spelled
    /// as its matching inverse, `exp(x·ln(10))`.
    ///
    /// The scan writes into a preallocated buffer rather than collecting 18
    /// slices and concatenating. When autodiff is off it cannibalizes its own
    /// input, which is the same shape — following
    /// [`SileroVad::forward_sequence`](crate::kits::speech::silero_vad::SileroVad::forward_sequence),
    /// where the single live reference lets `slice_assign` lower to an in-place
    /// update. With autodiff on it accumulates into a fresh tensor instead, so
    /// the graph stays intact.
    fn log_compress(
        &self,
        bands: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let [rows, _] = bands.dims();
        let device = bands.device();

        let mut log_max: Tensor<B, 2> = Tensor::full([rows, 1], -2.0, &device);
        let mut follow: Tensor<B, 2> = Tensor::full([rows, 1], -2.0, &device);

        let mut out = if B::ad_enabled(&device) {
            Tensor::zeros_like(&bands)
        } else {
            bands.clone()
        };

        for band in 0..NB_BANDS {
            let raw = bands
                .clone()
                .slice_dim(1, band as isize..(band + 1) as isize)
                .add_scalar(1e-2f32)
                .log()
                .div_scalar(core::f32::consts::LN_10);

            let decayed = follow.sub_scalar(2.5f32);
            let ly = log_max
                .clone()
                .sub_scalar(8.0f32)
                .max_pair(decayed.clone().max_pair(raw));

            log_max = log_max.max_pair(ly.clone());
            follow = decayed.max_pair(ly.clone());
            out = out.slice_assign(s![.., band..band + 1], ly);
        }

        out
    }

    /// The `-40 dB` noise floor on lag zero, then the lag window.
    ///
    /// The reference's operation order is preserved: lag zero takes
    /// `ac0 + (ac0*1e-4 + dc0_bias)` as three separate roundings, and the
    /// window's entry `0` is `1.0` so it leaves lag zero alone.
    fn apply_noise_floor(
        &self,
        ac: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let windowed = ac * self.tables.lag_window.clone().unsqueeze::<2>();

        let ac0 = windowed.clone().slice_dim(1, 0..1);
        let floored = ac0.clone().add(
            ac0.mul_scalar(1e-4f32)
                .add_scalar(dc0_bias(self.tables.window_size())),
        );

        Tensor::cat(vec![floored, windowed.slice_dim(1, 1..)], 1)
    }
}

/// Solves for whitening coefficients by Levinson-Durbin, batched over rows.
///
/// See the module docs for why the reference's early exit becomes a mask.
///
/// # Arguments
/// * `ac`: `[rows, lpc_order + 1]` autocorrelation lags, already floored and
///   windowed.
/// * `rows`: `ac`'s leading extent.
///
/// # Returns
/// `[rows, lpc_order]` coefficients.
#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::{
        super::super::{
            HostPitchEstimator,
            PitchScalarSource,
            coeff::LPC_ORDER,
            lpc::celt_lpc,
        },
        *,
    };
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;
    type D = <B as burn::tensor::backend::BackendTypes>::Device;

    const N_BINS: usize = 513;

    /// A plausible hop spectrum: smooth, non-negative, decaying, at the
    /// reference's int16 power scale.
    fn spectrum(seed: f32) -> Vec<f32> {
        (0..N_BINS)
            .map(|k| {
                let k = k as f32;
                1e7 * (-k / (60.0 + 20.0 * seed)).exp() * (1.0 + 0.5 * (k * 0.05 + seed).sin())
            })
            .collect()
    }

    fn stage(device: &D) -> PitchPrefilter<B> {
        PitchPrefilterConfig::new().init(device)
    }

    /// The host stage, reached through the oracle: `frame_pitch` runs the
    /// pre-filter design first and nothing afterwards rewrites `lpc`.
    fn host_lpc(bin_power: &[f32]) -> [f32; LPC_ORDER] {
        let mut est = HostPitchEstimator::new();
        est.frame_pitch(&[0.0; 256], bin_power);
        *est.lpc()
    }

    #[test]
    fn test_config_meta() {
        let cfg = PitchPrefilterConfig::new();
        assert_eq!(cfg.n_bins(), N_BINS);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        let bad =
            PitchPrefilterConfig::new().with_tables(PitchTablesConfig::new().with_fft_size(0));
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_init_meta_matches_config() {
        let device = Default::default();
        let stage = stage(&device);
        assert_eq!(stage.n_bins(), N_BINS);
        assert_eq!(stage.fft_size(), 1024);
    }

    #[test]
    fn test_forward_matches_host_stage() {
        // The differential test this phase exists for.
        let device = Default::default();
        let stage = stage(&device);

        for seed in [0.0f32, 0.7, 1.9, 3.3] {
            let power = spectrum(seed);
            let want = host_lpc(&power);

            let input = Tensor::<B, 1>::from_floats(power.as_slice(), &device).reshape([1, N_BINS]);
            let got: Vec<f32> = stage
                .forward(input)
                .to_data_as::<f32>()
                .to_vec_as::<f32>()
                .unwrap();

            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let rel = (g - w).abs() / w.abs().max(1e-3);
                assert!(rel < 1e-3, "seed {seed}, lpc[{i}]: {g} vs {w} (rel {rel})");
            }
        }
    }

    #[test]
    fn test_forward_batches_rows_independently() {
        // Stage 1 carries no state, so a batched call must equal per-row calls.
        let device = Default::default();
        let stage = stage(&device);

        let seeds = [0.2f32, 1.4, 2.6];
        let mut flat = Vec::new();
        for s in seeds {
            flat.extend_from_slice(&spectrum(s));
        }
        let batched =
            Tensor::<B, 1>::from_floats(flat.as_slice(), &device).reshape([seeds.len(), N_BINS]);
        let batched_out = stage.forward(batched);

        for (row, seed) in seeds.iter().enumerate() {
            let solo = Tensor::<B, 1>::from_floats(spectrum(*seed).as_slice(), &device)
                .reshape([1, N_BINS]);
            let solo_out = stage.forward(solo);

            batched_out
                .clone()
                .slice_dim(0, row as isize..(row + 1) as isize)
                .to_data()
                .assert_approx_eq::<f32>(&solo_out.to_data(), Tolerance::permissive());
        }
    }

    #[test]
    fn test_silence_produces_a_finite_filter() {
        // An all-zero spectrum drives every band to the log floor. The noise
        // floor on lag zero is what keeps the solve well-posed.
        let device = Default::default();
        let stage = stage(&device);

        let out = stage.forward(Tensor::<B, 2>::zeros([1, N_BINS], &device));
        let got: Vec<f32> = out.to_data_as::<f32>().to_vec_as::<f32>().unwrap();

        assert_eq!(got.len(), LPC_ORDER);
        for (i, c) in got.iter().enumerate() {
            assert!(c.is_finite(), "lpc[{i}] = {c}");
        }
        // And it agrees with the host on the same degenerate input.
        let want = host_lpc(&[0.0; N_BINS]);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-3, "lpc[{i}]: {g} vs {w}");
        }
    }

    #[test]
    fn test_levinson_reproduces_the_early_break() {
        // The masked freeze is what is under test here, so the fixture has to
        // be one that genuinely trips the reference's early exit. The sum of
        // two sinusoids is exactly predictable by a 4th-order recursion, so
        // the residual collapses and the break fires with most of the
        // coefficients still at their zero initializer.
        //
        // An AR(1) autocorrelation does *not* work: at `a = 0.98` the residual
        // settles near `1 - a²`, four decades above the 30 dB threshold, and
        // the recursion runs all sixteen steps.
        let device = Default::default();

        let ac: [f32; LPC_ORDER + 1] =
            core::array::from_fn(|i| 500.0 * ((0.3 * i as f32).cos() + (0.8 * i as f32).cos()));
        let want = celt_lpc(&ac);

        // Guard the guard, without pinning *where* the break lands: the host
        // runs in f32 and may exit a step earlier or later than an f64
        // analysis suggests. What matters is that it exited at all, which
        // shows as an untouched zero tail.
        let live = want.iter().rposition(|c| *c != 0.0).map_or(0, |i| i + 1);
        assert!(
            live < LPC_ORDER,
            "fixture should trip the early break, got {want:?}",
        );

        let input = Tensor::<B, 1>::from_floats(ac.as_slice(), &device).reshape([1, LPC_ORDER + 1]);
        let got: Vec<f32> = levinson_durbin_batched::<B>(input, Some(CELT_LPC_BAIL_RATIO))
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-3, "lpc[{i}]: {g} vs {w}");
        }
        // And the frozen tail is exactly zero on the device too, not merely
        // small: a freeze that leaked would show up here.
        for (i, g) in got.iter().enumerate().skip(live) {
            assert_eq!(*g, 0.0, "lpc[{i}] should be frozen at zero, got {g}");
        }
    }

    #[test]
    fn test_levinson_matches_the_host_across_shapes() {
        let device = Default::default();

        let cases: Vec<[f32; LPC_ORDER + 1]> = vec![
            core::array::from_fn(|i| 1000.0 * 0.8f32.powi(i as i32)),
            core::array::from_fn(|i| 500.0 / (1.0 + i as f32)),
            core::array::from_fn(|i| 100.0 * (1.0 + (i as f32 * 0.9).cos())),
            core::array::from_fn(|i| if i == 0 { 42.0 } else { 0.0 }),
        ];

        let mut flat = Vec::new();
        for c in &cases {
            flat.extend_from_slice(c);
        }
        let input = Tensor::<B, 1>::from_floats(flat.as_slice(), &device)
            .reshape([cases.len(), LPC_ORDER + 1]);
        let got: Vec<f32> = levinson_durbin_batched::<B>(input, Some(CELT_LPC_BAIL_RATIO))
            .to_data_as::<f32>()
            .to_vec_as::<f32>()
            .unwrap();

        for (row, case) in cases.iter().enumerate() {
            let want = celt_lpc(case);
            for (i, w) in want.iter().enumerate() {
                let g = got[row * LPC_ORDER + i];
                assert!((g - w).abs() < 1e-4, "case {row}, lpc[{i}]: {g} vs {w}");
            }
        }
    }
}
