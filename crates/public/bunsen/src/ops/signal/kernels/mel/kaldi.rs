//! [`KaldiMelBanksConfig`]: Kaldi's triangular mel filterbank.

use alloc::{
    vec,
    vec::Vec,
};

use burn::config::Config;

use super::MelScale;

/// Kaldi's mel filterbank, as `torchaudio.compliance.kaldi`'s `get_mel_banks`
/// computes it: `num_mel_bins` triangles on Kaldi's mel scale
/// ([`MelScale::Kaldi`]), their edges equally spaced in mels between the low
/// and high edges, over each FFT bin's centre frequency.
///
/// Unlike `librosa`'s ([`MelFilterbankConfig`](super::MelFilterbankConfig)),
/// the triangles are linear in mels, not in Hz, and are not normalized; the
/// Nyquist bin weighs nothing in any.
#[derive(Config, Copy, Debug, PartialEq)]
pub struct KaldiMelBanksConfig {
    /// The mel bins, `M`.
    pub num_mel_bins: usize,

    /// The FFT size; the banks cover its `fft_size / 2 + 1` bins.
    pub fft_size: usize,

    /// The sample rate, in Hz.
    pub sample_rate: f64,

    /// The bins' lower edge, in Hz.
    pub low_freq: f64,

    /// The bins' upper edge, in Hz.
    pub high_freq: f64,
}

impl KaldiMelBanksConfig {
    /// The banks, `[fft_size / 2 + 1, M]` row-major: FFT bin `k`'s weight in
    /// mel bin `m` at `k * M + m`.
    pub fn to_vec(&self) -> Vec<f64> {
        let mel = |hz: f64| MelScale::Kaldi.hz_to_mel(hz);
        let (bins, fft_bins) = (self.num_mel_bins, self.fft_size / 2);
        let (mel_low, mel_high) = (mel(self.low_freq), mel(self.high_freq));
        let delta = (mel_high - mel_low) / (bins + 1) as f64;
        let bin_width = self.sample_rate / self.fft_size as f64;
        let mut banks = vec![0.0; (fft_bins + 1) * bins];
        for k in 0..fft_bins {
            let at = mel(bin_width * k as f64);
            for m in 0..bins {
                let left = mel_low + m as f64 * delta;
                let (center, right) = (left + delta, left + 2.0 * delta);
                let up = (at - left) / (center - left);
                let down = (right - at) / (right - center);
                banks[k * bins + m] = up.min(down).max(0.0);
            }
        }
        banks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each triangle peaks at 1 or near it, is zero outside its edges, and
    /// the Nyquist row is empty; neighbours overlap, summing to at most 1.
    #[test]
    fn test_banks_shape() {
        let config = KaldiMelBanksConfig::new(23, 512, 16000.0, 20.0, 8000.0);
        let banks = config.to_vec();
        let bins = config.num_mel_bins;
        assert_eq!(banks.len(), 257 * bins);
        assert!(banks[256 * bins..].iter().all(|&w| w == 0.0));
        for m in 0..bins {
            let column: Vec<f64> = (0..257).map(|k| banks[k * bins + m]).collect();
            let peak = column.iter().copied().fold(0.0, f64::max);
            assert!(peak > 0.5 && peak <= 1.0, "bin {m} peaks at {peak}");
        }
        for k in 0..257 {
            let row: f64 = banks[k * bins..(k + 1) * bins].iter().sum();
            assert!(row <= 1.0 + 1e-12, "row {k} sums to {row}");
        }
    }
}
