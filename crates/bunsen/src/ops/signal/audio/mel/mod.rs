//! # Mel Scale Mechanics

use burn::prelude::Config;
use serde::{
    Deserialize,
    Serialize,
};

/// Mel scale variant for frequency conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum MelScale {
    /// HTK mel scale formula: 2595 * log10(1 + hz/700)
    Htk,

    /// Slaney mel scale: linear below 1kHz, logarithmic above
    /// Compatible with librosa (default)
    #[default]
    Slaney,
}

/// Normalization method for mel filterbank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum MelNorm {
    /// No normalization
    None,

    /// Slaney normalization: area under each filter = 1
    #[default]
    Slaney,
}

/// Configuration for mel spectrogram computation.
#[derive(Config, Debug, PartialEq)]
pub struct MelConfig {
    /// Number of mel bands (default: 80 for speech/Whisper)
    #[config(default = "80")]
    pub n_mels: usize,

    /// Minimum frequency in Hz (default: 0.0)
    #[config(default = "0.0")]
    pub fmin: f64,

    /// Maximum frequency in Hz (default: None = sample_rate/2)
    #[config(default = "None")]
    pub fmax: Option<f64>,

    /// Mel scale variant (default: Slaney)
    #[config(default = "Default::default()")]
    pub mel_scale: MelScale,

    /// Filterbank normalization (default: Slaney)
    #[config(default = "Default::default()")]
    pub norm: MelNorm,

    /// Use power spectrum instead of magnitude (default: true)
    #[config(default = "true")]
    pub use_power: bool,
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_scale() {
        let scale: MelScale = Default::default();
        assert_eq!(scale, MelScale::Slaney);
    }

    #[test]
    fn test_mel_norm() {
        let norm: MelNorm = Default::default();
        assert_eq!(norm, MelNorm::Slaney);
    }
}
