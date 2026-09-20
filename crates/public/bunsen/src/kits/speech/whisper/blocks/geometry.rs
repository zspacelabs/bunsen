//! # Whisper geometry
//!
//! The comparable core of a [`WhisperApiConfig`]: the numbers a checkpoint
//! reports, and nothing it cannot.

use super::{
    AUDIO_ENCODER_STRIDE,
    WHISPER_DEFAULT_D_MODEL,
    WhisperApiConfig,
};

/// The geometry a prefab fixes: the numbers
/// [`PytorchWhisperScanner`](crate::kits::speech::whisper::pretrained::PytorchWhisperScanner)
/// reads back from a checkpoint, and nothing a checkpoint cannot report.
///
/// A [`WhisperApiConfig`] carries more (the front end, the token layout)
/// and is not `PartialEq`; this is the comparable core of one, which is
/// what lets a checkpoint be checked against the shape its name promised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperGeometry {
    /// Mel bands in.
    pub n_mels: usize,
    /// Vocabulary size, specials included.
    pub vocab_size: usize,
    /// Embedding width.
    pub d_model: usize,
    /// Audio context, in mel frames.
    pub max_audio_ctx: usize,
    /// Encoder layers.
    pub n_encoder_layers: usize,
    /// Text context, in tokens.
    pub max_text_ctx: usize,
    /// Decoder layers.
    pub n_decoder_layers: usize,
    /// Head width.
    pub d_head: usize,
}

/// Upstream's audio context: 1500 encoder positions over the encoder's
/// stride, which is what the scanner reads from the positional embedding.
pub const OPENAI_MAX_AUDIO_CTX: usize = 1500 * AUDIO_ENCODER_STRIDE;

/// Upstream's text context.
pub const OPENAI_MAX_TEXT_CTX: usize = 448;

impl WhisperGeometry {
    /// An `OpenAI` geometry: the family's fixed contexts and 64-wide heads.
    pub const fn openai(
        n_mels: usize,
        vocab_size: usize,
        d_model: usize,
        n_encoder_layers: usize,
        n_decoder_layers: usize,
    ) -> Self {
        Self {
            n_mels,
            vocab_size,
            d_model,
            max_audio_ctx: OPENAI_MAX_AUDIO_CTX,
            n_encoder_layers,
            max_text_ctx: OPENAI_MAX_TEXT_CTX,
            n_decoder_layers,
            d_head: WHISPER_DEFAULT_D_MODEL,
        }
    }

    /// Attention heads per layer.
    pub fn n_heads(&self) -> usize {
        self.d_model / self.d_head
    }

    /// The config this geometry builds, with upstream's front end and token
    /// layout.
    pub fn to_api_config(&self) -> WhisperApiConfig {
        WhisperApiConfig::new(
            self.n_mels,
            self.vocab_size,
            self.d_model,
            self.max_audio_ctx,
            self.n_encoder_layers,
            self.max_text_ctx,
            self.n_decoder_layers,
        )
        .with_d_head(self.d_head)
    }
}

impl From<&WhisperApiConfig> for WhisperGeometry {
    fn from(cfg: &WhisperApiConfig) -> Self {
        Self {
            n_mels: cfg.n_mels,
            vocab_size: cfg.vocab_size,
            d_model: cfg.d_model,
            max_audio_ctx: cfg.max_audio_ctx,
            n_encoder_layers: cfg.n_encoder_layers,
            max_text_ctx: cfg.max_text_ctx,
            n_decoder_layers: cfg.n_decoder_layers,
            d_head: cfg.d_head,
        }
    }
}

impl WhisperApiConfig {
    /// The comparable core of this config.
    pub fn geometry(&self) -> WhisperGeometry {
        WhisperGeometry::from(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_round_trips_through_api_config() {
        let geometry = WhisperGeometry::openai(80, 51865, 512, 6, 6);
        let cfg = geometry.to_api_config();
        assert_eq!(cfg.geometry(), geometry);
        assert_eq!(WhisperGeometry::from(&cfg), geometry);
        assert_eq!(geometry.n_heads(), 8);
        assert_eq!(geometry.max_audio_ctx, 3000);
        assert_eq!(geometry.max_text_ctx, 448);
        assert_eq!(geometry.d_head, 64);
    }
}
