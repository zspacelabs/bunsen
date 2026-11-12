//! # Whisper prefabs

use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::speech::asr::whisper::{
        WhisperApiConfig,
        WhisperGeometry,
    },
};

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

impl WhisperApiConfig {
    /// The `tiny` prefab: `d_model` 384, 4 + 4 layers, multilingual.
    pub fn tiny() -> Self {
        TINY.to_api_config()
    }

    /// The `tiny.en` prefab: `d_model` 384, 4 + 4 layers, English-only.
    pub fn tiny_en() -> Self {
        TINY_EN.to_api_config()
    }

    /// The `base` prefab: `d_model` 512, 6 + 6 layers, multilingual.
    pub fn base() -> Self {
        BASE.to_api_config()
    }

    /// The `base.en` prefab: `d_model` 512, 6 + 6 layers, English-only.
    pub fn base_en() -> Self {
        BASE_EN.to_api_config()
    }

    /// The `small` prefab: `d_model` 768, 12 + 12 layers, multilingual.
    pub fn small() -> Self {
        SMALL.to_api_config()
    }

    /// The `small.en` prefab: `d_model` 768, 12 + 12 layers, English-only.
    pub fn small_en() -> Self {
        SMALL_EN.to_api_config()
    }

    /// The `medium` prefab: `d_model` 1024, 24 + 24 layers, multilingual.
    pub fn medium() -> Self {
        MEDIUM.to_api_config()
    }

    /// The `medium.en` prefab: `d_model` 1024, 24 + 24 layers, English-only.
    pub fn medium_en() -> Self {
        MEDIUM_EN.to_api_config()
    }

    /// The `large` prefab: `d_model` 1280, 32 + 32 layers, 80 mels, 99
    /// languages (large-v1, large-v2).
    pub fn large() -> Self {
        LARGE.to_api_config()
    }

    /// The `large-v3` prefab: `d_model` 1280, 32 + 32 layers, 128 mels, 100
    /// languages.
    pub fn large_v3() -> Self {
        LARGE_V3.to_api_config()
    }

    /// The `large-v3-turbo` prefab: `d_model` 1280, 32 + 4 layers, 128 mels,
    /// 100 languages.
    pub fn large_v3_turbo() -> Self {
        LARGE_V3_TURBO.to_api_config()
    }
}

/// The Whisper prefabs: every geometry `openai-whisper` ships, by name.
///
/// The public geometries, without weights: upstream's `ModelDimensions`
/// for every checkpoint `openai-whisper` ships, typed in rather than
/// scanned, so that a name means a shape before any bytes are fetched, and
/// so that a checkpoint can be checked against the shape its name
/// promised. A pretrained row names its prefab; [`WhisperGeometry::prefab`]
/// is the reverse lookup, for a checkpoint that arrived as a path.
///
/// The vocabulary size is part of the geometry: an English-only `tiny.en`
/// (51864) and a multilingual `tiny` (51865) are different prefabs, as they
/// are different `ModelDimensions` upstream. `large-v1` and `large-v2`
/// share one prefab, `large`; `large-v3` widened the mel bank to 128 and
/// added a hundredth language; `large-v3-turbo` kept its encoder over a
/// four-layer decoder.
///
/// What a prefab does *not* fix is what a checkpoint cannot report either:
/// the audio front end and the token layout. Those are upstream's for
/// every entry here, and [`WhisperApiConfig`]'s defaults say so.
impl Prefabs for WhisperApiConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("tiny", Self::tiny),
        ("tiny.en", Self::tiny_en),
        ("base", Self::base),
        ("base.en", Self::base_en),
        ("small", Self::small),
        ("small.en", Self::small_en),
        ("medium", Self::medium),
        ("medium.en", Self::medium_en),
        ("large", Self::large),
        ("large-v3", Self::large_v3),
        ("large-v3-turbo", Self::large_v3_turbo),
    ];
    const PREFAB_TABLE: &'static str = "whisper";
}

impl WhisperGeometry {
    /// The name of the prefab this geometry belongs to, if any: the
    /// reverse lookup in [`WhisperApiConfig`]'s [`Prefabs`], for a checkpoint
    /// that arrived as a path rather than a name.
    pub fn prefab(&self) -> Option<&'static str> {
        WhisperApiConfig::find_prefab(|cfg| cfg.geometry() == *self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_round_trips_through_api_config() {
        for (name, prefab) in WhisperApiConfig::PREFABS {
            let cfg = prefab();
            let geometry = cfg.geometry();
            assert_eq!(geometry.to_api_config().geometry(), geometry, "{}", name);
            assert_eq!(geometry.d_model % geometry.d_head, 0, "{}", name);
            assert_eq!(geometry.max_audio_ctx, 3000, "{}", name);
            assert_eq!(geometry.max_text_ctx, 448, "{}", name);
        }
    }

    #[test]
    fn test_prefab_names_are_unique() {
        let mut names = WhisperApiConfig::prefab_names();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), WhisperApiConfig::PREFABS.len());
        assert_eq!(WhisperApiConfig::PREFABS.len(), 11);
    }

    #[test]
    fn test_a_geometry_names_its_prefab() {
        assert_eq!(
            BASE.prefab(),
            Some("base"),
            "the base geometry names the base prefab",
        );
        assert_eq!(LARGE_V3_TURBO.prefab(), Some("large-v3-turbo"));
        assert!(
            WhisperGeometry::openai(80, 51865, 1000, 1, 1)
                .prefab()
                .is_none(),
            "an unknown geometry names nothing",
        );
    }

    #[test]
    fn test_vocab_sizes_follow_the_language_variant() {
        for (name, prefab) in WhisperApiConfig::PREFABS {
            let vocab = prefab().vocab_size;
            if name.ends_with(".en") {
                assert_eq!(vocab, 51864, "{}", name);
            } else if name.starts_with("large-v3") {
                assert_eq!(vocab, 51866, "{}", name);
            } else {
                assert_eq!(vocab, 51865, "{}", name);
            }
        }
    }
}
