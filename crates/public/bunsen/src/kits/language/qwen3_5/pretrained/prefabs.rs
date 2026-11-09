//! # Qwen3.5 prefabs
use crate::{
    data::pretrained::{
        StaticPreFabConfig,
        StaticPreFabMap,
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
pub static QWEN3_5_PREFABS: StaticPreFabMap<Qwen3_5Config> = StaticPreFabMap {
    name: "qwen3.5",
    description: "dense Qwen3.5 text geometries, as transformers' configs have them",
    items: &[
        &StaticPreFabConfig {
            name: "qwen3.5-0.8b",
            description: "Qwen3.5, 1024 wide, 24 layers (6 full), 8 query / 2 key-value heads, tied",
            builder: Qwen3_5Config::qwen3_5_0_8b,
        },
        &StaticPreFabConfig {
            name: "qwen3.5-2b",
            description: "Qwen3.5, 2048 wide, 24 layers (6 full), 8 query / 2 key-value heads, tied",
            builder: Qwen3_5Config::qwen3_5_2b,
        },
        &StaticPreFabConfig {
            name: "qwen3.5-4b",
            description: "Qwen3.5, 2560 wide, 32 layers (8 full), 16 query / 4 key-value heads, tied",
            builder: Qwen3_5Config::qwen3_5_4b,
        },
        &StaticPreFabConfig {
            name: "qwen3.5-9b",
            description: "Qwen3.5, 4096 wide, 32 layers (8 full), 16 query / 4 key-value heads",
            builder: Qwen3_5Config::qwen3_5_9b,
        },
        &StaticPreFabConfig {
            name: "qwen3.5-27b",
            description: "Qwen3.5, 5120 wide, 64 layers (16 full), 24 query / 4 key-value heads",
            builder: Qwen3_5Config::qwen3_5_27b,
        },
    ],
};

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
        for prefab in QWEN3_5_PREFABS.iter() {
            let config = prefab.to_config();
            config.stack().unwrap();
            assert_eq!(config.rotary_dim(), 64, "{}", prefab.name);
            assert_eq!(config.layer_kind(3), Qwen3_5LayerKind::Full);
            assert_eq!(config.layer_kind(4), Qwen3_5LayerKind::Linear);
        }
        let small = QWEN3_5_PREFABS
            .expect_lookup_prefab("qwen3.5-0.8b")
            .to_config();
        assert_eq!((small.d_model(), small.n_layers()), (1024, 24));
        assert!(small.tied_head);
        let nine = QWEN3_5_PREFABS
            .expect_lookup_prefab("qwen3.5-9b")
            .to_config();
        assert_eq!((nine.n_kv_heads(), nine.linear_value_heads), (4, 32));
        assert!(!nine.tied_head);
        assert!(
            QWEN3_5_PREFABS
                .try_lookup_prefab("qwen3.5-35b-a3b")
                .is_err()
        );
    }
}
