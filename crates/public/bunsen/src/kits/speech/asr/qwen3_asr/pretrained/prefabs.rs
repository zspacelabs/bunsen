//! # Qwen3-ASR prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::speech::asr::qwen3_asr::Qwen3AsrConfig,
};

/// The Qwen3-ASR prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched. `FermionResearch`'s Phonon-1 builds re-pack
/// `Qwen3-ASR-0.6B`'s weights without changing its shape, so they are
/// `qwen3-asr-0.6b`.
impl Prefabs for Qwen3AsrConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("qwen3-asr-0.6b", Self::qwen3_asr_0_6b),
        ("qwen3-asr-1.7b", Self::qwen3_asr_1_7b),
    ];
    const PREFAB_TABLE: &'static str = "qwen3-asr";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::asr::qwen3_asr::blocks::Qwen3AsrMeta;

    #[test]
    fn test_prefabs() {
        let config = Qwen3AsrConfig::expect_prefab("qwen3-asr-0.6b");
        assert_eq!(config.d_audio(), 896);
        assert_eq!(config.d_text(), 1024);
        assert!(Qwen3AsrConfig::try_prefab("qwen3-asr-9b").is_err());
    }
}
