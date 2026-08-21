//! # ten-vad mel filterbank.
//!
//! The 40-band triangular filterbank the ten-vad front end folds its 513 bin
//! powers through (`ALGO_TRACE.md` §3.6).
//!
//! This is **not** a standard mel filterbank, and a librosa / Slaney-style
//! builder will not reproduce it. Three details are load-bearing.
//!
//! **Edge mapping.** Band edges are equally spaced on the HTK mel scale, then
//! mapped to FFT bins by a truncating integer cast:
//!
//! ```text
//! mel  = 2595 * log10(1 + hz / 700)
//! bin  = (usize) ((fft_size + 1) * hz / sample_rate)
//! ```
//!
//! Note both the `fft_size + 1` and that the cast truncates, not rounds.
//!
//! **Integer slopes.** The triangles are built from integer bin differences,
//! so their slopes follow where the truncation landed rather than the exact
//! edge frequencies.
//!
//! **No area normalization.** Each filter rises from zero to one and falls
//! back to zero, peaking at exactly `1.0`.
//!
//! The edge arithmetic is deliberately done in `f32`, matching the reference;
//! computing it in `f64` can round an edge across an integer boundary and
//! silently produce a different filterbank.
//!
//! The pieces are:
//! * [`TenVadMelConfig`] — the geometry.
//! * [`TenVadMelBank`] — the materialized `[n_mels, n_bins]` filter matrix;
//!   built by [`TenVadMelConfig::try_init`].

use burn::{
    config::Config,
    prelude::*,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        WithOkOrPanic,
    },
    kits::speech::ten_vad::context::coeff::{
        N_MELS,
        SAMPLE_RATE,
    },
};

/// Common meta for [`TenVadMelConfig`] and [`TenVadMelBank`].
pub trait TenVadMelMeta {
    /// The number of mel bands.
    fn n_mels(&self) -> usize;

    /// The number of frequency bins consumed: `fft_size / 2 + 1`.
    fn n_bins(&self) -> usize;
}

/// Config for [`TenVadMelBank`].
///
/// Defaults match the ten-vad reference: 40 bands spanning 0 Hz to 8 kHz over
/// a 1024-point FFT at 16 kHz.
///
/// Builds [`TenVadMelBank`]. Implements [`TenVadMelMeta`].
#[derive(Config, Debug)]
pub struct TenVadMelConfig {
    /// The number of mel bands.
    #[config(default = "N_MELS")]
    pub n_mels: usize,

    /// The FFT size the bin powers came from.
    #[config(default = "1024")]
    pub fft_size: usize,

    /// The sample rate, in Hz.
    #[config(default = "SAMPLE_RATE")]
    pub sample_rate: usize,

    /// The low edge of the filterbank, in Hz.
    #[config(default = "0.0")]
    pub f_min: f32,

    /// The high edge of the filterbank, in Hz.
    #[config(default = "8000.0")]
    pub f_max: f32,
}

impl TenVadMelMeta for TenVadMelConfig {
    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn n_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }
}

/// Converts a frequency, in Hz, to the HTK mel scale.
///
/// `mel = 2595 * log10(1 + hz / 700)`, evaluated in `f32` to match the
/// reference.
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0f32 * (1.0f32 + hz / 700.0f32).log10()
}

/// Converts an HTK mel value back to a frequency, in Hz.
///
/// The inverse of [`hz_to_mel`], evaluated in `f32` to match the reference.
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0f32 * (10.0f32.powf(mel / 2595.0f32) - 1.0f32)
}

impl TenVadMelConfig {
    /// The `n_mels + 2` triangular band edges, as FFT bin indices.
    ///
    /// Edges are equally spaced on the mel scale between
    /// [`f_min`](Self::f_min) and [`f_max`](Self::f_max), then mapped to bins
    /// by the reference's truncating `(fft_size + 1) * hz / sample_rate` cast.
    ///
    /// Band `i` rises from `edges[i]`, peaks at `edges[i + 1]`, and falls to
    /// `edges[i + 2]`.
    pub fn bin_edges(&self) -> Vec<usize> {
        let low_mel = hz_to_mel(self.f_min);
        let high_mel = hz_to_mel(self.f_max);

        let steps = self.n_mels + 1;
        (0..=steps)
            .map(|i| {
                let mel = low_mel + (high_mel - low_mel) * i as f32 / steps as f32;
                let hz = mel_to_hz(mel);
                // Truncating, as in the reference; note the `fft_size + 1`.
                ((self.fft_size + 1) as f32 * hz / self.sample_rate as f32) as usize
            })
            .collect()
    }

    /// Validates the filterbank geometry.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if `n_mels` or `fft_size` is zero, if the
    /// frequency range is not increasing, or if two adjacent band edges land
    /// on the same FFT bin (which would make a triangle infinitely steep).
    pub fn validate(&self) -> BunsenResult<()> {
        if self.n_mels == 0 {
            return Err(BunsenError::Invalid(
                "TenVadMel n_mels must be non-zero".to_string(),
            ));
        }
        if self.fft_size == 0 {
            return Err(BunsenError::Invalid(
                "TenVadMel fft_size must be non-zero".to_string(),
            ));
        }
        if self.sample_rate == 0 {
            return Err(BunsenError::Invalid(
                "TenVadMel sample_rate must be non-zero".to_string(),
            ));
        }
        // Written through `partial_cmp` so a NaN bound is rejected, rather
        // than silently passing a negated comparison.
        if !matches!(
            self.f_max.partial_cmp(&self.f_min),
            Some(core::cmp::Ordering::Greater),
        ) {
            return Err(BunsenError::Invalid(format!(
                "TenVadMel f_max ({}) must be > f_min ({})",
                self.f_max, self.f_min,
            )));
        }

        let edges = self.bin_edges();
        for i in 1..edges.len() {
            if edges[i] == edges[i - 1] {
                return Err(BunsenError::Invalid(format!(
                    "TenVadMel band edges {} and {} both land on bin {}; \
                     the geometry is too coarse for {} bands",
                    i - 1,
                    i,
                    edges[i],
                    self.n_mels,
                )));
            }
        }
        Ok(())
    }

    /// The `[n_mels, n_bins]` filter matrix, as a host-side row-major vector.
    ///
    /// Exposed so callers (and tests) can inspect the coefficients without a
    /// device round trip.
    pub fn to_vec_weights(&self) -> Vec<f32> {
        let n_bins = self.n_bins();
        let edges = self.bin_edges();
        let mut weights = vec![0.0f32; self.n_mels * n_bins];

        for i in 0..self.n_mels {
            let (lo, mid, hi) = (edges[i], edges[i + 1], edges[i + 2]);
            let row = i * n_bins;

            // Rising slope: 0 -> 1 across [lo, mid).
            for j in lo..mid {
                if j < n_bins {
                    weights[row + j] = (j - lo) as f32 / (mid - lo) as f32;
                }
            }
            // Falling slope: 1 -> 0 across [mid, hi). The peak of exactly 1.0
            // lands at `mid`, from this loop's first iteration.
            for j in mid..hi {
                if j < n_bins {
                    weights[row + j] = (hi - j) as f32 / (hi - mid) as f32;
                }
            }
        }
        weights
    }

    /// Initializes a [`TenVadMelBank`] on `device`.
    ///
    /// # Errors
    ///
    /// See [`validate`](Self::validate).
    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BunsenResult<TenVadMelBank<B>> {
        self.validate()?;

        let weights = Tensor::from_data(
            TensorData::new(self.to_vec_weights(), [self.n_mels, self.n_bins()]),
            device,
        );

        Ok(TenVadMelBank { weights })
    }

    /// Initializes a [`TenVadMelBank`] on `device`, panicking on error.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> TenVadMelBank<B> {
        self.try_init(device).ok_or_panic()
    }
}

/// The materialized ten-vad mel filterbank.
///
/// Holds the dense `[n_mels, n_bins]` filter matrix. Stateless, so one
/// instance can be shared by (or cheaply cloned into) any number of streams.
/// This is deliberately **not** a burn `Module`: nothing here is a learnable
/// parameter, and the coefficients are fixed by the pretrained weights.
///
/// Built by [`TenVadMelConfig`]. Implements [`TenVadMelMeta`].
#[derive(Debug, Clone)]
pub struct TenVadMelBank<B: Backend> {
    /// The `[n_mels, n_bins]` triangular filter matrix.
    pub weights: Tensor<B, 2>,
}

impl<B: Backend> TenVadMelMeta for TenVadMelBank<B> {
    fn n_mels(&self) -> usize {
        self.weights.dims()[0]
    }

    fn n_bins(&self) -> usize {
        self.weights.dims()[1]
    }
}

impl<B: Backend> TenVadMelBank<B> {
    /// Folds bin powers into mel band energies.
    ///
    /// # Arguments
    /// * `bin_power`: `[rows, n_bins]` non-negative bin powers. The driver
    ///   passes a `[steps * batch, n_bins]` flatten, so `rows` is whatever
    ///   leading extent the caller has collapsed.
    ///
    /// # Returns
    /// `[rows, n_mels]` band energies.
    pub fn forward(
        &self,
        bin_power: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        #[cfg(any(test, debug_assertions))]
        crate::contracts::assert_shape_contract!(
            ["rows", "n_bins"],
            &bin_power,
            &[("n_bins", self.n_bins())],
        );

        bin_power.matmul(self.weights.clone().transpose())
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
        backend::BackendTypes,
    };

    use super::*;
    use crate::{
        prelude::*,
        support::testing::CpuBackend,
    };

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    /// The reference band edges, in FFT bins, for the stock ten-vad geometry.
    ///
    /// Independently recomputed from the reference formula in `f32`; these are
    /// the numbers the pretrained weights were trained against.
    const REFERENCE_EDGES: [usize; 42] = [
        0, 2, 5, 9, 12, 16, 19, 24, 28, 33, 38, 43, 48, 54, 61, 67, 75, 82, 90, 99, 108, 118, 128,
        139, 151, 163, 176, 190, 205, 221, 238, 256, 275, 296, 317, 340, 365, 391, 418, 448, 479,
        512,
    ];

    #[test]
    fn test_config_meta() {
        let cfg = TenVadMelConfig::new();
        assert_eq!(cfg.n_mels, 40);
        assert_eq!(cfg.fft_size, 1024);
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.f_min, 0.0);
        assert_eq!(cfg.f_max, 8000.0);

        assert_eq!(cfg.n_mels(), 40);
        assert_eq!(cfg.n_bins(), 513);

        cfg.validate().unwrap();
    }

    #[test]
    fn test_mel_scale_round_trips() {
        // The HTK mel scale, anchored at its defining points.
        assert_eq!(hz_to_mel(0.0), 0.0);
        assert_eq!(mel_to_hz(0.0), 0.0);

        // 700 Hz is one doubling of the `1 + hz/700` term: 2595 * log10(2).
        assert!((hz_to_mel(700.0) - 2595.0 * 2.0f32.log10()).abs() < 1e-2);

        for hz in [100.0f32, 700.0, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-2, "{hz} -> {back}");
        }
    }

    #[test]
    fn test_bin_edges_match_the_reference() {
        let edges = TenVadMelConfig::new().bin_edges();
        assert_eq!(edges.len(), 42, "n_mels + 2 edges");
        assert_eq!(edges.as_slice(), REFERENCE_EDGES.as_slice());

        // The bank spans the whole spectrum, exactly: bin 0 through bin 512.
        assert_eq!(edges[0], 0);
        assert_eq!(*edges.last().unwrap(), 512);
        assert_eq!(*edges.last().unwrap(), TenVadMelConfig::new().n_bins() - 1);

        // Strictly increasing, so no triangle is degenerate.
        for i in 1..edges.len() {
            assert!(edges[i] > edges[i - 1], "edge {i} did not advance");
        }
    }

    #[test]
    fn test_weights_shape_and_range() {
        let cfg = TenVadMelConfig::new();
        let weights = cfg.to_vec_weights();
        assert_eq!(weights.len(), 40 * 513);

        for (i, &w) in weights.iter().enumerate() {
            assert!((0.0..=1.0).contains(&w), "weight[{i}] = {w} outside [0, 1]",);
        }

        // 949 non-zero coefficients across the bank; a change here means the
        // triangle geometry moved.
        assert_eq!(weights.iter().filter(|&&w| w > 0.0).count(), 949);
    }

    #[test]
    fn test_each_filter_peaks_at_exactly_one() {
        // The filters are *not* area-normalized: each rises 0 -> 1 and falls
        // 1 -> 0, peaking at its centre edge.
        let cfg = TenVadMelConfig::new();
        let edges = cfg.bin_edges();
        let weights = cfg.to_vec_weights();
        let n_bins = cfg.n_bins();

        for i in 0..cfg.n_mels {
            let row = &weights[i * n_bins..(i + 1) * n_bins];
            let peak = row.iter().copied().fold(0.0f32, f32::max);
            assert_eq!(peak, 1.0, "filter {i} peak");
            assert_eq!(row[edges[i + 1]], 1.0, "filter {i} peak position");
        }
    }

    #[test]
    fn test_first_filter_coefficients() {
        // Band 0 spans edges (0, 2, 5): it rises across bins 0..2 and falls
        // across bins 2..5, so the exact triangle is checkable by hand.
        let cfg = TenVadMelConfig::new();
        let weights = cfg.to_vec_weights();

        let expected = [0.0, 0.5, 1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0];
        for (j, &e) in expected.iter().enumerate() {
            assert!(
                (weights[j] - e).abs() < 1e-6,
                "band 0 bin {j}: {} vs {e}",
                weights[j],
            );
        }
    }

    #[test]
    fn test_validate_rejects_bad_geometry() {
        for bad in [
            TenVadMelConfig::new().with_n_mels(0),
            TenVadMelConfig::new().with_fft_size(0),
            TenVadMelConfig::new().with_sample_rate(0),
            // f_max must exceed f_min.
            TenVadMelConfig::new().with_f_max(0.0),
            TenVadMelConfig::new().with_f_min(9000.0),
            // Far too few bins to separate 40 bands: edges collide.
            TenVadMelConfig::new().with_fft_size(64),
        ] {
            assert!(
                matches!(bad.validate(), Err(BunsenError::Invalid(_))),
                "expected Invalid: {bad:?}",
            );
        }
    }

    #[test]
    fn test_init_meta_matches_config() {
        let device = Default::default();
        let cfg = TenVadMelConfig::new();
        let bank: TenVadMelBank<B> = cfg.init(&device);

        assert_eq!(bank.n_mels(), cfg.n_mels());
        assert_eq!(bank.n_bins(), cfg.n_bins());
        assert_eq!(bank.weights.dims(), [40, 513]);

        // The device matrix matches the host construction elementwise.
        bank.weights.to_data_as::<F>().assert_approx_eq::<F>(
            &TensorData::new(cfg.to_vec_weights(), [40, 513]).convert::<F>(),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_forward_matches_naive_host_dot() {
        let device = Default::default();
        let cfg = TenVadMelConfig::new();
        let bank: TenVadMelBank<B> = cfg.init(&device);

        let rows = 3;
        let n_bins = cfg.n_bins();

        // Bin powers are non-negative by construction.
        let bin_power =
            Tensor::<B, 2>::random([rows, n_bins], Distribution::Uniform(0.0, 4.0), &device);

        let out = bank.forward(bin_power.clone());
        assert_eq!(out.dims(), [rows, cfg.n_mels()]);

        let host_power: Vec<f32> = bin_power.to_data_as::<f32>().to_vec_as::<f32>().unwrap();
        let host_weights = cfg.to_vec_weights();

        let mut expected = Vec::with_capacity(rows * cfg.n_mels());
        for r in 0..rows {
            for m in 0..cfg.n_mels() {
                let mut acc = 0.0f32;
                for j in 0..n_bins {
                    acc += host_power[r * n_bins + j] * host_weights[m * n_bins + j];
                }
                expected.push(acc);
            }
        }

        out.to_data_as::<F>().assert_approx_eq::<F>(
            &TensorData::new(expected, [rows, cfg.n_mels()]).convert::<F>(),
            Tolerance::permissive(),
        );
    }

    #[test]
    fn test_forward_is_non_negative_and_linear() {
        // The bank is a non-negative linear map, so scaling the input scales
        // the output and the result never goes negative.
        let device = Default::default();
        let bank: TenVadMelBank<B> = TenVadMelConfig::new().init(&device);

        let bin_power = Tensor::<B, 2>::random([2, 513], Distribution::Uniform(0.0, 1.0), &device);

        let once = bank.forward(bin_power.clone());
        let twice = bank.forward(bin_power.mul_scalar(2.0));

        let host: Vec<f32> = once.to_data_as::<f32>().to_vec_as::<f32>().unwrap();
        assert!(host.iter().all(|&v| v >= 0.0));

        twice.to_data_as::<F>().assert_approx_eq::<F>(
            &once.mul_scalar(2.0).to_data_as::<F>(),
            Tolerance::permissive(),
        );
    }
}
