//! # Triangular filterbanks and frequency scales.
//!
//! The construction every mel, bark, and ERB filterbank shares: a set of band
//! edges in bin coordinates, and a triangle per band rising from one edge to
//! the next and falling to the one after.
//!
//! ```text
//! edges:   e[0]      e[1]      e[2]      e[3]
//!            \        /\        /\        /
//! band 0:     \______/  \      /  \      /
//! band 1:               \____/     \____/
//! ```
//!
//! Two pieces, deliberately separate:
//!
//! * [`TriangularBankConfig`] builds the weights from edges you supply. It
//!   knows nothing about frequency.
//! * [`mel_bin_edges`] and [`MelScale`] produce mel-spaced edges to hand it.
//!
//! Keeping them apart is what makes the bank reusable. A filterbank on any
//! warped scale — bark, ERB, a learned spacing, or one transcribed from a
//! reference implementation with its own rounding — is the same triangles over
//! different edges, and only the edge computation has to change.
//!
//! ## Edges are fractional
//!
//! Edges are `f32` bin coordinates, not integers, so a band boundary can fall
//! between bins and the slopes reflect where it actually landed. Passing
//! integer-valued edges is fine and reproduces an integer-rounded bank exactly;
//! that is a property of the input, not a limitation of the construction.

use burn::{
    config::Config,
    prelude::*,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
    WithOkOrPanic,
};

/// A perceptual frequency warping.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum MelScale {
    /// `2595 * log10(1 + hz / 700)`, the HTK / Slaney-"htk" formula.
    ///
    /// Log-warped everywhere, which makes it smooth but gives it no linear
    /// region at low frequency. Calibrated so 1000 Hz is very nearly 1000 mel.
    Htk,

    /// Linear below 1 kHz, logarithmic above, as in Slaney's Auditory Toolbox.
    ///
    /// librosa's default. The two are not interchangeable: they disagree by
    /// tens of mel across the speech band, so a model trained against one will
    /// not accept the other's filterbank.
    Slaney,
}

/// The breakpoint of [`MelScale::Slaney`], in Hz.
const SLANEY_BREAK_HZ: f32 = 1000.0;

/// Mel per Hz below the Slaney breakpoint.
const SLANEY_LINEAR_SLOPE: f32 = 3.0 / 200.0;

/// The logarithmic step above the Slaney breakpoint.
const SLANEY_LOG_STEP: f32 = 0.068_751_777;

impl MelScale {
    /// Maps a frequency in Hz to its mel value.
    pub fn hz_to_mel(
        &self,
        hz: f32,
    ) -> f32 {
        match self {
            Self::Htk => 2595.0 * (1.0 + hz / 700.0).log10(),
            Self::Slaney => {
                let linear = hz * SLANEY_LINEAR_SLOPE;
                if hz < SLANEY_BREAK_HZ {
                    linear
                } else {
                    SLANEY_BREAK_HZ * SLANEY_LINEAR_SLOPE
                        + (hz / SLANEY_BREAK_HZ).ln() / SLANEY_LOG_STEP
                }
            }
        }
    }

    /// Maps a mel value back to Hz; the inverse of
    /// [`hz_to_mel`](Self::hz_to_mel).
    pub fn mel_to_hz(
        &self,
        mel: f32,
    ) -> f32 {
        match self {
            Self::Htk => 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0),
            Self::Slaney => {
                let breakpoint = SLANEY_BREAK_HZ * SLANEY_LINEAR_SLOPE;
                if mel < breakpoint {
                    mel / SLANEY_LINEAR_SLOPE
                } else {
                    SLANEY_BREAK_HZ * ((mel - breakpoint) * SLANEY_LOG_STEP).exp()
                }
            }
        }
    }
}

/// How each triangle is scaled.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum BankNorm {
    /// Every triangle peaks at exactly `1.0`, whatever its width.
    ///
    /// Wide bands therefore accumulate more energy than narrow ones. This is
    /// what HTK does, and what most reference C implementations do.
    Peak,

    /// Every triangle is scaled by `2 / (upper - lower)`, so its area is
    /// constant.
    ///
    /// Slaney's normalization, and librosa's `norm="slaney"` default. Keeps a
    /// flat spectrum flat across bands rather than tilting it by bandwidth.
    Area,
}

/// Config for [`TriangularBank`].
#[derive(Config, Debug, Copy)]
pub struct TriangularBankConfig {
    /// The number of spectrum bins the bank consumes.
    pub n_bins: usize,

    /// How each triangle is scaled.
    #[config(default = "BankNorm::Peak")]
    pub norm: BankNorm,
}

impl TriangularBankConfig {
    /// The number of bands `edges` describes.
    ///
    /// Each band needs a lower, centre and upper edge, and consecutive bands
    /// share two of them, so `n + 2` edges give `n` bands.
    pub fn n_bands(edges: &[f32]) -> usize {
        edges.len().saturating_sub(2)
    }

    /// Validates the geometry against a set of edges.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if there are fewer than three edges, if the
    /// edges are not strictly increasing, or if the bin count is zero.
    pub fn validate(
        &self,
        edges: &[f32],
    ) -> BunsenResult<()> {
        if self.n_bins == 0 {
            return Err(BunsenError::Invalid(
                "TriangularBank n_bins must be non-zero".to_string(),
            ));
        }
        if edges.len() < 3 {
            return Err(BunsenError::Invalid(format!(
                "TriangularBank needs at least 3 edges to describe one band, got {}",
                edges.len(),
            )));
        }
        if let Some(i) = edges.windows(2).position(|w| w[1] <= w[0]) {
            return Err(BunsenError::Invalid(format!(
                "TriangularBank edges must strictly increase; edge {i} is {} and edge \
                 {} is {}",
                edges[i],
                i + 1,
                edges[i + 1],
            )));
        }
        Ok(())
    }

    /// The `[n_bands, n_bins]` filter matrix, row-major.
    ///
    /// Band `i` rises from `edges[i]` to `edges[i + 1]` and falls to
    /// `edges[i + 2]`. Bins outside `0..n_bins` are dropped, so edges may run
    /// past the spectrum without special-casing.
    ///
    /// # Panics
    /// If the geometry is invalid; see [`validate`](Self::validate).
    pub fn to_vec_weights(
        &self,
        edges: &[f32],
    ) -> Vec<f32> {
        self.validate(edges).ok_or_panic();

        let n_bands = Self::n_bands(edges);
        let mut weights = vec![0.0f32; n_bands * self.n_bins];

        for band in 0..n_bands {
            let (lo, mid, hi) = (edges[band], edges[band + 1], edges[band + 2]);
            let row = band * self.n_bins;

            let scale = match self.norm {
                BankNorm::Peak => 1.0,
                BankNorm::Area => 2.0 / (hi - lo),
            };

            // Rising, over the bins in [lo, mid).
            let start = lo.ceil().max(0.0) as usize;
            let centre = mid.ceil().max(0.0) as usize;
            for j in start..centre.min(self.n_bins) {
                weights[row + j] = scale * (j as f32 - lo) / (mid - lo);
            }

            // Falling, over the bins in [mid, hi). A bin landing exactly on
            // `mid` gets the peak from this branch, not the rising one.
            let end = hi.ceil().max(0.0) as usize;
            for j in centre..end.min(self.n_bins) {
                weights[row + j] = scale * (hi - j as f32) / (hi - mid);
            }
        }

        weights
    }

    /// Builds the bank, uploading a matmul-ready `[n_bins, n_bands]` matrix.
    ///
    /// # Errors
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        edges: &[f32],
        device: &B::Device,
    ) -> BunsenResult<TriangularBank<B>> {
        self.validate(edges)?;
        let n_bands = Self::n_bands(edges);

        // Stored transposed: callers contract `[rows, n_bins] @ [n_bins,
        // n_bands]`, so the transpose is paid once here rather than per call.
        let weights: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(self.to_vec_weights(edges), [n_bands, self.n_bins]),
            device,
        );

        Ok(TriangularBank {
            n_bins: self.n_bins,
            n_bands,
            weights: weights.transpose(),
        })
    }

    /// Builds the bank, panicking on error.
    pub fn init<B: Backend>(
        &self,
        edges: &[f32],
        device: &B::Device,
    ) -> TriangularBank<B> {
        self.try_init(edges, device).ok_or_panic()
    }
}

/// A materialized triangular filterbank.
///
/// Built by [`TriangularBankConfig::try_init`]. Deliberately not a burn
/// `Module`: nothing here is learnable, and the weights are derived from the
/// geometry rather than trained.
#[derive(Debug, Clone)]
pub struct TriangularBank<B: Backend> {
    n_bins: usize,
    n_bands: usize,

    /// `[n_bins, n_bands]`, matmul-ready.
    pub weights: Tensor<B, 2>,
}

impl<B: Backend> TriangularBank<B> {
    /// The number of spectrum bins consumed.
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// The number of bands produced.
    pub fn n_bands(&self) -> usize {
        self.n_bands
    }

    /// Folds `[rows, n_bins]` spectra into `[rows, n_bands]` band energies.
    pub fn forward(
        &self,
        spectra: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["rows", "n_bins"],
            &spectra,
            &[("n_bins", self.n_bins)],
        );

        spectra.matmul(self.weights.clone())
    }
}

/// Mel-spaced band edges, in fractional bin coordinates.
///
/// Returns `n_bands + 2` edges, equally spaced on the mel scale between `fmin`
/// and `fmax`, mapped back to Hz and then to bins.
///
/// # Arguments
/// * `n_bands`: how many bands the edges should describe.
/// * `fft_size`: the FFT the bins come from.
/// * `sample_rate`: in Hz.
/// * `fmin` / `fmax`: the band range, in Hz.
/// * `scale`: which mel warping to use — see [`MelScale`], and note the two are
///   not interchangeable.
///
/// # Panics
/// If `n_bands` is zero or `fmax` does not exceed `fmin`.
pub fn mel_bin_edges(
    n_bands: usize,
    fft_size: usize,
    sample_rate: usize,
    fmin: f32,
    fmax: f32,
    scale: MelScale,
) -> Vec<f32> {
    assert_ne!(n_bands, 0, "mel_bin_edges needs at least one band");
    assert!(
        fmax > fmin,
        "mel_bin_edges needs fmax ({fmax}) > fmin ({fmin})"
    );

    let (lo, hi) = (scale.hz_to_mel(fmin), scale.hz_to_mel(fmax));
    let step = (hi - lo) / (n_bands + 1) as f32;
    let per_hz = fft_size as f32 / sample_rate as f32;

    (0..n_bands + 2)
        .map(|i| scale.mel_to_hz(lo + step * i as f32) * per_hz)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        prelude::*,
        support::testing::PerformanceBackend,
    };

    type B = PerformanceBackend;

    const FFT: usize = 512;
    const BINS: usize = FFT / 2 + 1;
    const SR: usize = 16000;

    #[test]
    fn test_htk_mel_round_trips() {
        for hz in [0.0f32, 100.0, 700.0, 1000.0, 4000.0, 8000.0] {
            let back = MelScale::Htk.mel_to_hz(MelScale::Htk.hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-2, "{hz} -> {back}");
        }
    }

    #[test]
    fn test_slaney_mel_round_trips() {
        // Both sides of the 1 kHz breakpoint, and the breakpoint itself.
        for hz in [0.0f32, 250.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0] {
            let back = MelScale::Slaney.mel_to_hz(MelScale::Slaney.hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-2, "{hz} -> {back}");
        }
    }

    #[test]
    fn test_htk_mel_is_calibrated_at_1khz() {
        // The formula's constants are chosen so 1000 Hz lands on ~1000 mel.
        let mel = MelScale::Htk.hz_to_mel(1000.0);
        assert!((mel - 1000.0).abs() < 1.0, "1000 Hz -> {mel} mel");
    }

    #[test]
    fn test_slaney_is_linear_below_the_breakpoint() {
        // The defining property of the Slaney warping, and what separates it
        // from HTK where it matters most -- the low end of the speech band.
        let a = MelScale::Slaney.hz_to_mel(200.0);
        let b = MelScale::Slaney.hz_to_mel(400.0);
        let c = MelScale::Slaney.hz_to_mel(600.0);
        assert!(
            (b - a - (c - b)).abs() < 1e-4,
            "{a} {b} {c} not equally spaced"
        );
    }

    #[test]
    fn test_the_two_scales_actually_differ() {
        // Guard the guard: if these ever agreed, every test above would pass
        // against a single implementation.
        let htk = MelScale::Htk.hz_to_mel(300.0);
        let slaney = MelScale::Slaney.hz_to_mel(300.0);
        assert!(
            (htk - slaney).abs() > 1.0,
            "HTK {htk} and Slaney {slaney} should disagree",
        );
    }

    #[test]
    fn test_mel_edges_are_increasing_and_bounded() {
        let edges = mel_bin_edges(20, FFT, SR, 0.0, 8000.0, MelScale::Htk);
        assert_eq!(edges.len(), 22);
        assert!(edges.windows(2).all(|w| w[1] > w[0]), "{edges:?}");
        assert!(edges[0] >= 0.0);
        assert!(*edges.last().unwrap() <= BINS as f32);
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        let cfg = TriangularBankConfig::new(BINS);
        assert!(cfg.validate(&[1.0, 2.0]).is_err(), "too few edges");
        assert!(cfg.validate(&[1.0, 1.0, 2.0]).is_err(), "not increasing");
        assert!(cfg.validate(&[3.0, 2.0, 1.0]).is_err(), "decreasing");
        assert!(
            TriangularBankConfig::new(0)
                .validate(&[1.0, 2.0, 3.0])
                .is_err(),
            "zero bins",
        );
        cfg.validate(&[1.0, 2.0, 3.0]).unwrap();
    }

    #[test]
    fn test_peak_normalization_reaches_exactly_one() {
        // With integer edges the centre bin sits exactly on the peak.
        let edges = [2.0f32, 6.0, 11.0];
        let w = TriangularBankConfig::new(BINS).to_vec_weights(&edges);
        assert_eq!(w[6], 1.0, "peak should be exactly 1 at the centre edge");
        assert_eq!(w[2], 0.0, "the lower edge itself carries no weight");
    }

    #[test]
    fn test_triangles_partition_unity_on_shared_spans() {
        // The closed-form property of a triangular bank: where two adjacent
        // bands overlap, their weights sum to exactly one. It pins both slopes
        // and their alignment at once.
        let edges = mel_bin_edges(16, FFT, SR, 0.0, 8000.0, MelScale::Htk);
        let cfg = TriangularBankConfig::new(BINS);
        let w = cfg.to_vec_weights(&edges);
        let n_bands = TriangularBankConfig::n_bands(&edges);

        // Bins strictly inside the second..last spans are covered by exactly
        // two triangles.
        let lo = edges[1].ceil() as usize;
        let hi = edges[n_bands].floor() as usize;
        assert!(hi > lo, "need an interior region to test");

        for j in lo..hi {
            let total: f32 = (0..n_bands).map(|b| w[b * BINS + j]).sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "bin {j} sums to {total}, expected 1",
            );
        }
    }

    #[test]
    fn test_area_normalization_equalizes_bands() {
        // A flat spectrum should come out flat, which is the whole point of
        // the Slaney scaling -- and visibly does not under peak scaling.
        let edges = mel_bin_edges(16, FFT, SR, 0.0, 8000.0, MelScale::Htk);
        let n_bands = TriangularBankConfig::n_bands(&edges);

        let area = TriangularBankConfig::new(BINS)
            .with_norm(BankNorm::Area)
            .to_vec_weights(&edges);
        let peak = TriangularBankConfig::new(BINS).to_vec_weights(&edges);

        let sums = |w: &[f32]| -> Vec<f32> {
            (0..n_bands)
                .map(|b| w[b * BINS..(b + 1) * BINS].iter().sum())
                .collect()
        };

        // Skip the narrowest low bands, where bin quantization dominates.
        let a = sums(&area);
        let p = sums(&peak);
        let spread = |v: &[f32]| {
            let tail = &v[n_bands / 2..];
            let max = tail.iter().fold(0.0f32, |m, x| m.max(*x));
            let min = tail.iter().fold(f32::MAX, |m, x| m.min(*x));
            max / min
        };

        assert!(
            spread(&a) < spread(&p),
            "area normalization should even the bands out: area {} vs peak {}",
            spread(&a),
            spread(&p),
        );
    }

    #[test]
    fn test_forward_folds_a_spectrum() {
        let device = Default::default();
        let edges = mel_bin_edges(12, FFT, SR, 0.0, 8000.0, MelScale::Htk);
        let bank: TriangularBank<B> = TriangularBankConfig::new(BINS).init(&edges, &device);

        assert_eq!(bank.n_bins(), BINS);
        assert_eq!(bank.n_bands(), 12);
        assert_eq!(bank.weights.dims(), [BINS, 12]);

        let spectra = Tensor::<B, 2>::ones([3, BINS], &device);
        let out = bank.forward(spectra);
        assert_eq!(out.dims(), [3, 12]);

        // A flat spectrum folds to each band's weight sum.
        let got: Vec<f32> = out.to_data_as::<f32>().to_vec_as::<f32>().ok_or_panic();
        let w = TriangularBankConfig::new(BINS).to_vec_weights(&edges);
        for band in 0..12 {
            let want: f32 = w[band * BINS..(band + 1) * BINS].iter().sum();
            assert!(
                (got[band] - want).abs() < 1e-3 * want.max(1e-3),
                "band {band}: {} vs {want}",
                got[band],
            );
        }
    }

    #[test]
    fn test_edges_past_the_spectrum_are_dropped() {
        // Edges may run past Nyquist; the bank simply stops rather than
        // needing the caller to clamp.
        let cfg = TriangularBankConfig::new(8);
        let w = cfg.to_vec_weights(&[2.0, 5.0, 40.0]);
        assert_eq!(w.len(), 8);
        assert!(w.iter().all(|v| v.is_finite()));
        assert!(w[7] > 0.0, "the band should still cover the last bin");
    }
}
