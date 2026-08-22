//! # Projection tables for the tensor pitch front end.
//!
//! Three fixed matrices, plus two coefficient vectors, covering every linear
//! step of the pre-filter design.
//!
//! ## Built by probing, not by transcription
//!
//! Each matrix is the linearization of a host function in the `lpc` module,
//! and every one of those functions is linear by
//! construction. So rather than re-deriving the index arithmetic — the
//! rounded band spans, the doubled edge bands, the "later span wins" overlap
//! rule — each column is obtained by *evaluating the host function on a basis
//! vector*.
//!
//! That makes agreement with the host a property of the construction rather
//! than something a test has to establish, and it means the awkward parts
//! (`band_span`'s independent end rounding, `interp_band_gain`'s assignment
//! semantics) are defined in exactly one place.
//!
//! ## The composition that matters
//!
//! `interp_band_gain` → zero Nyquist → `autocorrelate` is three linear steps
//! from `[nb_bands]` to `[lpc_order + 1]`, so it folds into a single
//! `[nb_bands, lpc_order + 1]` matrix. That removes the `n_bins`-wide
//! intermediate from the device path entirely: the device contracts 18 terms
//! where the host contracts 513.
//!
//! It also sidesteps a precision trap. [`Autocorrelator::autocorrelate`]
//! deliberately accumulates in `f64` over 511 `f32` terms, because a flat
//! `f32` sum there would be materially worse than the FFT it replaces. A naive
//! device `f32` matmul over 513 bins would reintroduce exactly that error.
//! Folding the composition on the host keeps the `f64` accumulation where it
//! belongs and leaves the device an 18-term contraction.

use burn::prelude::*;

use super::super::{
    coeff::{
        BAND_LPC_COMP,
        LPC_ORDER,
        NB_BANDS,
    },
    lpc::{
        Autocorrelator,
        DctTable,
        band_energy,
        interp_band_gain,
    },
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Config for [`PitchTables`].
///
/// Defaults match the ten-vad front end: a 1024-point STFT over a 768-sample
/// analysis window.
#[derive(Config, Debug, Copy)]
pub struct PitchTablesConfig {
    /// The FFT size the bin powers come from.
    #[config(default = "1024")]
    pub fft_size: usize,

    /// The STFT analysis window length, which sets the LPC noise floor.
    #[config(default = "768")]
    pub window_size: usize,
}

impl PitchTablesConfig {
    /// The number of frequency bins, `fft_size / 2 + 1`.
    pub fn n_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Validates the geometry.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if `fft_size` is not a positive even number,
    /// or if `window_size` is zero.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.fft_size == 0 || !self.fft_size.is_multiple_of(2) {
            return Err(BunsenError::Invalid(format!(
                "PitchTables fft_size ({}) must be a positive even number",
                self.fft_size,
            )));
        }
        if self.window_size == 0 {
            return Err(BunsenError::Invalid(
                "PitchTables window_size must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    /// The `[n_bins, nb_bands]` band-folding matrix, row-major.
    ///
    /// Column `i` is [`band_energy`] evaluated on the unit spectrum at bin
    /// `i`, so the doubled edge bands and the clamped tail index come along
    /// for free.
    pub fn to_vec_bands(&self) -> Vec<f32> {
        let n_bins = self.n_bins();
        let mut out = vec![0.0f32; n_bins * NB_BANDS];
        let mut probe = vec![0.0f32; n_bins];

        for bin in 0..n_bins {
            probe[bin] = 1.0;
            let row = band_energy(&probe, self.fft_size);
            out[bin * NB_BANDS..(bin + 1) * NB_BANDS].copy_from_slice(&row);
            probe[bin] = 0.0;
        }
        out
    }

    /// The `[nb_bands, nb_bands]` DCT matrix, row-major.
    ///
    /// One matrix serves both directions: `cepstrum = bands @ dct` and
    /// `log_gain = cepstrum @ dctᵀ`, because the reference's forward and
    /// inverse transforms differ only in which index of the table they walk.
    pub fn to_vec_dct(&self) -> Vec<f32> {
        let table = DctTable::new();
        let mut out = vec![0.0f32; NB_BANDS * NB_BANDS];
        let mut probe = [0.0f32; NB_BANDS];

        for j in 0..NB_BANDS {
            probe[j] = 1.0;
            let row = table.dct(&probe);
            out[j * NB_BANDS..(j + 1) * NB_BANDS].copy_from_slice(&row);
            probe[j] = 0.0;
        }
        out
    }

    /// The `[nb_bands, lpc_order + 1]` band-gain to autocorrelation matrix,
    /// row-major.
    ///
    /// The composition of [`interp_band_gain`], the Nyquist zeroing, and
    /// [`Autocorrelator::autocorrelate`] — the whole linear part of
    /// `lpc_from_bands`, up to the elementwise affine on the lags.
    pub fn to_vec_ac_from_bands(&self) -> Vec<f32> {
        let n_bins = self.n_bins();
        let autocorrelator = Autocorrelator::new(self.fft_size);

        let mut out = vec![0.0f32; NB_BANDS * (LPC_ORDER + 1)];
        let mut probe = [0.0f32; NB_BANDS];
        let mut bins = vec![0.0f32; n_bins];
        let mut lags = [0.0f32; LPC_ORDER + 1];

        for band in 0..NB_BANDS {
            probe[band] = 1.0;
            interp_band_gain(&probe, &mut bins);
            // As `lpc_from_bands` does, before transforming.
            bins[n_bins - 1] = 0.0;
            autocorrelator.autocorrelate(&bins, &mut lags);
            out[band * (LPC_ORDER + 1)..(band + 1) * (LPC_ORDER + 1)].copy_from_slice(&lags);
            probe[band] = 0.0;
        }
        out
    }

    /// The `[lpc_order + 1]` lag window, `1 - 6e-5·i²`.
    ///
    /// Entry `0` is `1.0`: lag zero takes the noise-floor affine instead, and
    /// applying that separately keeps the reference's operation order.
    pub fn to_vec_lag_window(&self) -> Vec<f32> {
        let mut out = vec![1.0f32; LPC_ORDER + 1];
        for (i, slot) in out.iter_mut().enumerate().skip(1) {
            *slot = 1.0 - 6e-5 * i as f32 * i as f32;
        }
        out
    }

    /// Uploads the tables.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<PitchTables<B>> {
        self.validate()?;
        let n_bins = self.n_bins();

        Ok(PitchTables {
            fft_size: self.fft_size,
            window_size: self.window_size,
            bands: Tensor::from_data(
                TensorData::new(self.to_vec_bands(), [n_bins, NB_BANDS]),
                device,
            ),
            dct: Tensor::from_data(
                TensorData::new(self.to_vec_dct(), [NB_BANDS, NB_BANDS]),
                device,
            ),
            ac_from_bands: Tensor::from_data(
                TensorData::new(self.to_vec_ac_from_bands(), [NB_BANDS, LPC_ORDER + 1]),
                device,
            ),
            lag_window: Tensor::from_data(
                TensorData::new(self.to_vec_lag_window(), [LPC_ORDER + 1]),
                device,
            ),
            band_lpc_comp: Tensor::from_data(
                TensorData::new(BAND_LPC_COMP.to_vec(), [NB_BANDS]),
                device,
            ),
        })
    }

    /// Uploads the tables, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PitchTables<B> {
        use crate::errors::WithOkOrPanic;
        self.try_init(device).ok_or_panic()
    }
}

/// The fixed projection tables of the tensor pitch pre-filter.
///
/// Stateless; one instance can be shared by any number of streams. Built by
/// [`PitchTablesConfig::try_init`].
#[derive(Debug, Clone)]
pub struct PitchTables<B: Backend> {
    fft_size: usize,
    window_size: usize,

    /// `[n_bins, nb_bands]`: `bands = bin_power @ this`.
    pub bands: Tensor<B, 2>,

    /// `[nb_bands, nb_bands]`: `cepstrum = bands @ this`, and
    /// `log_gain = cepstrum @ this.transpose()`.
    pub dct: Tensor<B, 2>,

    /// `[nb_bands, lpc_order + 1]`: `ac = band_gain @ this`.
    pub ac_from_bands: Tensor<B, 2>,

    /// `[lpc_order + 1]` lag window; entry `0` is `1.0`.
    pub lag_window: Tensor<B, 1>,

    /// `[nb_bands]` per-band LPC compensation.
    pub band_lpc_comp: Tensor<B, 1>,
}

impl<B: Backend> PitchTables<B> {
    /// The FFT size these tables were built for.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// The number of frequency bins.
    pub fn n_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// The STFT analysis window length.
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::{
        super::super::lpc::{
            celt_lpc,
            dc0_bias,
            lpc_from_bands,
        },
        *,
    };
    use crate::support::testing::PerformanceBackend;

    type B = PerformanceBackend;

    fn config() -> PitchTablesConfig {
        PitchTablesConfig::new()
    }

    /// A plausible non-negative band envelope.
    fn band_gains() -> [f32; NB_BANDS] {
        core::array::from_fn(|i| 100.0 * (-(i as f32) / 3.5).exp() * (1.0 + 0.3 * i as f32).sqrt())
    }

    #[test]
    fn test_config_meta() {
        let cfg = config();
        assert_eq!(cfg.fft_size, 1024);
        assert_eq!(cfg.window_size, 768);
        assert_eq!(cfg.n_bins(), 513);
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        assert!(config().validate().is_ok());
        assert!(config().with_fft_size(0).validate().is_err());
        assert!(config().with_fft_size(1023).validate().is_err());
        assert!(config().with_window_size(0).validate().is_err());
    }

    #[test]
    fn test_init_meta_matches_config() {
        let device = Default::default();
        let tables: PitchTables<B> = config().init(&device);

        assert_eq!(tables.fft_size(), 1024);
        assert_eq!(tables.n_bins(), 513);
        assert_eq!(tables.window_size(), 768);
        assert_eq!(tables.bands.dims(), [513, NB_BANDS]);
        assert_eq!(tables.dct.dims(), [NB_BANDS, NB_BANDS]);
        assert_eq!(tables.ac_from_bands.dims(), [NB_BANDS, LPC_ORDER + 1]);
        assert_eq!(tables.lag_window.dims(), [LPC_ORDER + 1]);
        assert_eq!(tables.band_lpc_comp.dims(), [NB_BANDS]);
    }

    #[test]
    fn test_bands_matrix_matches_band_energy() {
        // Linearity is the premise of the whole file; check it on a spectrum
        // that is nothing like the basis vectors it was probed with.
        let cfg = config();
        let n_bins = cfg.n_bins();
        let matrix = cfg.to_vec_bands();

        let spectrum: Vec<f32> = (0..n_bins)
            .map(|k| 1e3 * (-(k as f32) / 90.0).exp() * (1.0 + 0.4 * (k as f32 * 0.07).sin()))
            .collect();

        let expected = band_energy(&spectrum, cfg.fft_size);
        for (band, want) in expected.iter().enumerate() {
            let got: f32 = (0..n_bins)
                .map(|k| spectrum[k] * matrix[k * NB_BANDS + band])
                .sum();
            let rel = (got - want).abs() / want.abs().max(1.0);
            assert!(rel < 1e-5, "band {band}: {got} vs {want} (rel {rel})");
        }
    }

    #[test]
    fn test_bands_matrix_doubles_the_edge_bands() {
        // The reference doubles bands 0 and 17 because each only ever receives
        // one side of a ramp. If the probe lost that, every log-mel-adjacent
        // number downstream would shift.
        let cfg = config();
        let matrix = cfg.to_vec_bands();
        // Bin 0 sits at the head of band 0's ramp, weight (1 - 0) doubled.
        // Row-major, so band 0 of bin 0 is simply entry 0.
        assert!((matrix[0] - 2.0).abs() < 1e-6, "{}", matrix[0]);
    }

    #[test]
    fn test_dct_matrix_matches_the_host_table_both_directions() {
        let cfg = config();
        let matrix = cfg.to_vec_dct();
        let table = DctTable::new();
        let input = band_gains();

        let forward = table.dct(&input);
        let inverse = table.idct(&input);

        for i in 0..NB_BANDS {
            // dct: row-major contraction over the first index.
            let got_fwd: f32 = (0..NB_BANDS)
                .map(|j| input[j] * matrix[j * NB_BANDS + i])
                .sum();
            // idct: the same matrix, transposed.
            let got_inv: f32 = (0..NB_BANDS)
                .map(|j| input[j] * matrix[i * NB_BANDS + j])
                .sum();

            assert!(
                (got_fwd - forward[i]).abs() < 1e-4,
                "dct[{i}]: {got_fwd} vs {}",
                forward[i],
            );
            assert!(
                (got_inv - inverse[i]).abs() < 1e-4,
                "idct[{i}]: {got_inv} vs {}",
                inverse[i],
            );
        }
    }

    #[test]
    fn test_ac_from_bands_matches_the_host_composition() {
        // The load-bearing simplification: interp -> zero Nyquist ->
        // autocorrelate, folded into one [18, 17] matrix.
        let cfg = config();
        let n_bins = cfg.n_bins();
        let matrix = cfg.to_vec_ac_from_bands();
        let gains = band_gains();

        let mut bins = vec![0.0f32; n_bins];
        interp_band_gain(&gains, &mut bins);
        bins[n_bins - 1] = 0.0;
        let mut expected = [0.0f32; LPC_ORDER + 1];
        Autocorrelator::new(cfg.fft_size).autocorrelate(&bins, &mut expected);

        for lag in 0..=LPC_ORDER {
            let got: f32 = (0..NB_BANDS)
                .map(|b| gains[b] * matrix[b * (LPC_ORDER + 1) + lag])
                .sum();
            let rel = (got - expected[lag]).abs() / expected[lag].abs().max(1.0);
            assert!(
                rel < 1e-5,
                "lag {lag}: {got} vs {} (rel {rel})",
                expected[lag],
            );
        }
    }

    #[test]
    fn test_ac_from_bands_feeds_the_same_lpc_as_the_host() {
        // End-to-end for the linear half: apply the reference's affine to the
        // matrix-derived lags and check the solve agrees.
        let cfg = config();
        let matrix = cfg.to_vec_ac_from_bands();
        let lag_window = cfg.to_vec_lag_window();
        let gains = band_gains();

        let mut ac = [0.0f32; LPC_ORDER + 1];
        for (lag, slot) in ac.iter_mut().enumerate() {
            *slot = (0..NB_BANDS)
                .map(|b| gains[b] * matrix[b * (LPC_ORDER + 1) + lag])
                .sum();
        }
        // The reference's ordering: lag 0 takes the noise floor, the rest the
        // lag window.
        ac[0] += ac[0] * 1e-4 + dc0_bias(cfg.window_size);
        for (i, slot) in ac.iter_mut().enumerate().skip(1) {
            *slot *= lag_window[i];
        }
        let got = celt_lpc(&ac);

        let mut scratch = vec![0.0f32; cfg.n_bins()];
        let want = lpc_from_bands(
            &gains,
            cfg.window_size,
            &Autocorrelator::new(cfg.fft_size),
            &mut scratch,
        );

        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-4, "lpc[{i}]: {g} vs {w}");
        }
    }

    #[test]
    fn test_lag_window_matches_the_reference_formula() {
        let window = config().to_vec_lag_window();
        assert_eq!(window[0], 1.0, "lag 0 takes the noise floor instead");
        for (i, w) in window.iter().enumerate().skip(1) {
            let want = 1.0 - 6e-5 * i as f32 * i as f32;
            assert_eq!(*w, want, "lag {i}");
        }
        // Monotonically tapering, and never inverting the sign of a lag.
        assert!(window[LPC_ORDER] > 0.0);
        assert!(window[LPC_ORDER] < window[1]);
    }

    #[test]
    fn test_uploaded_tables_match_their_host_vectors() {
        let device = Default::default();
        let cfg = config();
        let tables: PitchTables<B> = cfg.init(&device);

        tables.bands.to_data().assert_approx_eq::<f32>(
            &TensorData::new(cfg.to_vec_bands(), [cfg.n_bins(), NB_BANDS]),
            Tolerance::permissive(),
        );
        tables.ac_from_bands.to_data().assert_approx_eq::<f32>(
            &TensorData::new(cfg.to_vec_ac_from_bands(), [NB_BANDS, LPC_ORDER + 1]),
            Tolerance::permissive(),
        );
    }
}
