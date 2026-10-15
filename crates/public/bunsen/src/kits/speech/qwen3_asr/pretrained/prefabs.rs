//! # Qwen3-ASR prefabs
use crate::{
    data::pretrained::{
        StaticPreFabConfig,
        StaticPreFabMap,
    },
    kits::speech::qwen3_asr::Qwen3AsrConfig,
};

/// The Qwen3-ASR prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched. `FermionResearch`'s Phonon-1 builds re-pack
/// `Qwen3-ASR-0.6B`'s weights without changing its shape, so every row is
/// `qwen3-asr-0.6b`.
pub static QWEN3_ASR_PREFABS: StaticPreFabMap<Qwen3AsrConfig> = StaticPreFabMap {
    name: "qwen3-asr",
    description: "Qwen3-ASR geometries, as transformers' Qwen3ASRConfig has them",
    items: &[&StaticPreFabConfig {
        name: "qwen3-asr-0.6b",
        description: "a 0.6B audio tower (896, 18 layers) and Qwen3-0.6B (1024, 28 layers)",
        builder: Qwen3AsrConfig::qwen3_asr_0_6b,
    }],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::qwen3_asr::blocks::Qwen3AsrMeta;

    #[test]
    fn test_prefabs() {
        let config = QWEN3_ASR_PREFABS
            .expect_lookup_prefab("qwen3-asr-0.6b")
            .to_config();
        assert_eq!(config.d_audio(), 896);
        assert_eq!(config.d_text(), 1024);
        assert!(QWEN3_ASR_PREFABS.try_lookup_prefab("qwen3-asr-9b").is_err());
    }
}
