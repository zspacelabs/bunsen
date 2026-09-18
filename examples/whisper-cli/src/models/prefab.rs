//! Built-in Whisper prefabs: the public geometries, without weights.
//!
//! Upstream's `ModelDimensions` for every checkpoint `openai-whisper` ships,
//! typed in rather than scanned, so that a name means a shape before any
//! bytes are fetched &mdash; and so that a checkpoint can be checked against
//! the shape its name promised
//! ([`loader::check_geometry`](super::loader::check_geometry)).
//!
//! The vocabulary size is part of the geometry: an English-only `tiny.en`
//! (51864) and a multilingual `tiny` (51865) are different prefabs, as they
//! are different `ModelDimensions` upstream. `large-v1` and `large-v2` share
//! one prefab, `large`; `large-v3` widened the mel bank to 128 and added a
//! hundredth language; `large-v3-turbo` kept its encoder over a four-layer
//! decoder.
//!
//! What a prefab does *not* fix is what a checkpoint cannot report either:
//! the audio front end and the token layout. Those are upstream's for every
//! entry here, and [`WhisperApiConfig`]'s defaults say so.

use bunsen::{
    data::pretrained::{
        StaticPreFabConfig,
        StaticPreFabMap,
    },
    kits::speech::whisper::{
        WhisperApiConfig,
        blocks::{
            AUDIO_ENCODER_STRIDE,
            WHISPER_DEFAULT_D_MODEL,
        },
    },
};

/// The geometry a prefab fixes: the numbers
/// [`PytorchWhisperScanner`](bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner)
/// reads back from a checkpoint, and nothing a checkpoint cannot report.
///
/// A [`WhisperApiConfig`] carries more (the front end, the token layout)
/// and is not `PartialEq`; this is the comparable core of one.
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
const OPENAI_MAX_AUDIO_CTX: usize = 1500 * AUDIO_ENCODER_STRIDE;

/// Upstream's text context.
const OPENAI_MAX_TEXT_CTX: usize = 448;

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

// `whisper/model.py`'s `ModelDimensions`, per checkpoint. The multilingual
// vocabulary is 51865 (99 languages); `large-v3` is 51866 (a hundredth,
// Cantonese); English-only is 51864.
const TINY: WhisperGeometry = WhisperGeometry::openai(80, 51865, 384, 4, 4);
const TINY_EN: WhisperGeometry = WhisperGeometry::openai(80, 51864, 384, 4, 4);
const BASE: WhisperGeometry = WhisperGeometry::openai(80, 51865, 512, 6, 6);
const BASE_EN: WhisperGeometry = WhisperGeometry::openai(80, 51864, 512, 6, 6);
const SMALL: WhisperGeometry = WhisperGeometry::openai(80, 51865, 768, 12, 12);
const SMALL_EN: WhisperGeometry = WhisperGeometry::openai(80, 51864, 768, 12, 12);
const MEDIUM: WhisperGeometry = WhisperGeometry::openai(80, 51865, 1024, 24, 24);
const MEDIUM_EN: WhisperGeometry = WhisperGeometry::openai(80, 51864, 1024, 24, 24);
const LARGE: WhisperGeometry = WhisperGeometry::openai(80, 51865, 1280, 32, 32);
const LARGE_V3: WhisperGeometry = WhisperGeometry::openai(128, 51866, 1280, 32, 32);
const LARGE_V3_TURBO: WhisperGeometry = WhisperGeometry::openai(128, 51866, 1280, 32, 4);

/// The Whisper prefabs: every geometry `openai-whisper` ships, by name.
///
/// `weights` is `None` throughout. bunsen's
/// [`StaticPretrainedWeightsDescriptor`](bunsen::data::pretrained::StaticPretrainedWeightsDescriptor)
/// has a name and URLs but no digest, no format and no notion of a local
/// source, which a 3 GB checkpoint needs; the pretrained side lives in
/// [`pretrained`](super::pretrained) until that descriptor grows them.
pub static WHISPER_PREFABS: StaticPreFabMap<WhisperApiConfig> = StaticPreFabMap {
    name: "whisper",
    description: "OpenAI Whisper geometries, as `whisper.model.ModelDimensions` has them",
    items: &[
        &StaticPreFabConfig {
            name: "tiny",
            description: "d_model 384, 4 + 4 layers, multilingual",
            builder: || TINY.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "tiny.en",
            description: "d_model 384, 4 + 4 layers, English-only",
            builder: || TINY_EN.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "base",
            description: "d_model 512, 6 + 6 layers, multilingual",
            builder: || BASE.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "base.en",
            description: "d_model 512, 6 + 6 layers, English-only",
            builder: || BASE_EN.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "small",
            description: "d_model 768, 12 + 12 layers, multilingual",
            builder: || SMALL.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "small.en",
            description: "d_model 768, 12 + 12 layers, English-only",
            builder: || SMALL_EN.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "medium",
            description: "d_model 1024, 24 + 24 layers, multilingual",
            builder: || MEDIUM.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "medium.en",
            description: "d_model 1024, 24 + 24 layers, English-only",
            builder: || MEDIUM_EN.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "large",
            description: "d_model 1280, 32 + 32 layers, 80 mels, 99 languages (large-v1, large-v2)",
            builder: || LARGE.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "large-v3",
            description: "d_model 1280, 32 + 32 layers, 128 mels, 100 languages",
            builder: || LARGE_V3.to_api_config(),
            weights: None,
        },
        &StaticPreFabConfig {
            name: "large-v3-turbo",
            description: "d_model 1280, 32 + 4 layers, 128 mels, 100 languages",
            builder: || LARGE_V3_TURBO.to_api_config(),
            weights: None,
        },
    ],
};

/// The prefab a geometry belongs to, if any: the reverse lookup, for a
/// checkpoint that arrived as a path rather than a name.
pub fn prefab_for_geometry(
    geometry: &WhisperGeometry
) -> Option<&'static StaticPreFabConfig<WhisperApiConfig>> {
    WHISPER_PREFABS
        .items
        .iter()
        .copied()
        .find(|p| WhisperGeometry::from(&p.to_config()) == *geometry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_round_trips_through_api_config() {
        for prefab in WHISPER_PREFABS.items {
            let cfg = prefab.to_config();
            let geometry = WhisperGeometry::from(&cfg);
            assert_eq!(
                WhisperGeometry::from(&geometry.to_api_config()),
                geometry,
                "{}",
                prefab.name
            );
            assert_eq!(geometry.d_model % geometry.d_head, 0, "{}", prefab.name);
            assert_eq!(geometry.max_audio_ctx, 3000, "{}", prefab.name);
            assert_eq!(geometry.max_text_ctx, 448, "{}", prefab.name);
        }
    }

    #[test]
    fn test_prefab_names_are_unique() {
        let mut names: Vec<&str> = WHISPER_PREFABS.items.iter().map(|p| p.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), WHISPER_PREFABS.items.len());
    }

    #[test]
    fn test_prefab_for_geometry() {
        assert_eq!(
            prefab_for_geometry(&BASE).map(|p| p.name),
            Some("base"),
            "the base geometry names the base prefab",
        );
        assert_eq!(
            prefab_for_geometry(&LARGE_V3_TURBO).map(|p| p.name),
            Some("large-v3-turbo")
        );
        assert!(
            prefab_for_geometry(&WhisperGeometry::openai(80, 51865, 1000, 1, 1)).is_none(),
            "an unknown geometry names nothing",
        );
    }

    #[test]
    fn test_vocab_sizes_follow_the_language_variant() {
        for prefab in WHISPER_PREFABS.items {
            let vocab = prefab.to_config().vocab_size;
            if prefab.name.ends_with(".en") {
                assert_eq!(vocab, 51864, "{}", prefab.name);
            } else if prefab.name.starts_with("large-v3") {
                assert_eq!(vocab, 51866, "{}", prefab.name);
            } else {
                assert_eq!(vocab, 51865, "{}", prefab.name);
            }
        }
    }
}
