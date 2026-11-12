//! # Qwen3.5 prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::language::qwen3_5::Qwen3_5Config,
};

/// The Qwen3.5 prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched; a `-Base` checkpoint has its chat sibling's geometry.
/// The `config.json`s are not committed: the validation crate checks each
/// prefab against its rows' files.
impl Prefabs for Qwen3_5Config {
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
        kits::language::qwen3_5::Qwen3_5LayerKind,
    };

    /// Every prefab is a hybrid of every fourth layer full, a quarter of each
    /// head rotated; the 9B and 27B untie their heads.
    #[test]
    fn test_prefabs() {
        for (name, prefab) in Qwen3_5Config::PREFABS {
            let config = prefab();
            config.stack().unwrap();
            assert_eq!(config.rotary_dim(), 64, "{}", name);
            assert_eq!(config.layer_kind(3), Qwen3_5LayerKind::Full);
            assert_eq!(config.layer_kind(4), Qwen3_5LayerKind::Linear);
        }
        let small = Qwen3_5Config::expect_prefab("qwen3.5-0.8b");
        assert_eq!((small.d_model(), small.n_layers()), (1024, 24));
        assert!(small.tied_head);
        let nine = Qwen3_5Config::expect_prefab("qwen3.5-9b");
        assert_eq!((nine.n_kv_heads(), nine.linear_value_heads), (4, 32));
        assert!(!nine.tied_head);
        assert!(Qwen3_5Config::try_prefab("qwen3.5-35b-a3b").is_err());
    }
}
