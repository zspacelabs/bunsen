//! # Qwen3 prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::language::qwen3::Qwen3Config,
};

/// The Qwen3 prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched. A `-Base` checkpoint has its chat model's geometry.
impl Prefabs for Qwen3Config {
    const PREFABS: PrefabTable<Self> = &[
        ("qwen3-0.6b", Self::qwen3_0_6b),
        ("qwen3-1.7b", Self::qwen3_1_7b),
        ("qwen3-4b", Self::qwen3_4b),
        ("qwen3-4b-2507", Self::qwen3_4b_2507),
        ("qwen3-8b", Self::qwen3_8b),
    ];
    const PREFAB_TABLE: &'static str = "qwen3";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::transformers::models::language_decoder::LmDecoderMeta;

    #[test]
    fn test_prefabs() {
        let config = Qwen3Config::expect_prefab("qwen3-4b");
        assert_eq!(config.d_model(), 2560);
        assert_eq!(config.n_layers(), 36);
        let config = Qwen3Config::expect_prefab("qwen3-8b");
        assert_eq!(config.d_model(), 4096);
        assert!(!config.tied_head);
        assert!(Qwen3Config::try_prefab("qwen3-14b").is_err());
    }
}
