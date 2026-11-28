//! # Qwen3.5 prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::language::generation::qwen3_5::Qwen3_5ContractConfig,
};

/// The Qwen3.5 prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched; a `-Base` checkpoint has its chat sibling's geometry.
/// The `config.json`s are not committed: the validation crate checks each
/// prefab against its rows' files.
impl Prefabs for Qwen3_5ContractConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("qwen3.5-0.8b", Self::qwen3_5_0_8b),
        ("qwen3.5-2b", Self::qwen3_5_2b),
        ("qwen3.5-4b", Self::qwen3_5_4b),
        ("qwen3.5-9b", Self::qwen3_5_9b),
        ("qwen3.5-27b", Self::qwen3_5_27b),
    ];
    const PREFAB_TABLE: &'static str = "qwen3.5";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::models::language_decoder::LmDecoderMeta,
        errors::testing::ErrorMatcher,
        kits::language::generation::qwen3_5::Qwen3_5LayerKind,
    };

    /// Every prefab is a hybrid of every fourth layer full, a quarter of each
    /// head rotated; the 9B and 27B untie their heads.
    #[test]
    fn test_prefabs() {
        for (name, prefab) in Qwen3_5ContractConfig::PREFABS {
            let config = prefab();
            config.try_stack().unwrap();
            assert_eq!(config.rotary_dim(), 64, "{}", name);
            assert_eq!(config.layer_kind(3), Qwen3_5LayerKind::Full);
            assert_eq!(config.layer_kind(4), Qwen3_5LayerKind::Linear);
        }
        let small = Qwen3_5ContractConfig::expect_prefab("qwen3.5-0.8b");
        assert_eq!((small.d_model(), small.n_layers()), (1024, 24));
        assert!(small.tied_head);
        let nine = Qwen3_5ContractConfig::expect_prefab("qwen3.5-9b");
        assert_eq!((nine.n_kv_heads(), nine.linear_value_heads), (4, 32));
        assert!(!nine.tied_head);
        ErrorMatcher::lookup().assert_err(&Qwen3_5ContractConfig::try_prefab("qwen3.5-35b-a3b"));
    }
}
