//! # Qwen3 prefabs
use crate::{
    data::pretrained::{
        StaticPreFabConfig,
        StaticPreFabMap,
    },
    kits::gpts::qwen3::Qwen3Config,
};

/// The Qwen3 prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched. A `-Base` checkpoint has its chat model's geometry.
pub static QWEN3_PREFABS: StaticPreFabMap<Qwen3Config> = StaticPreFabMap {
    name: "qwen3",
    description: "dense Qwen3 geometries, as transformers' Qwen3Config has them",
    items: &[
        &StaticPreFabConfig {
            name: "qwen3-0.6b",
            description: "1024 wide, 28 layers, 16 query / 8 key-value heads",
            builder: Qwen3Config::qwen3_0_6b,
        },
        &StaticPreFabConfig {
            name: "qwen3-1.7b",
            description: "2048 wide, 28 layers, 16 query / 8 key-value heads",
            builder: Qwen3Config::qwen3_1_7b,
        },
        &StaticPreFabConfig {
            name: "qwen3-4b",
            description: "2560 wide, 36 layers, 32 query / 8 key-value heads",
            builder: Qwen3Config::qwen3_4b,
        },
        &StaticPreFabConfig {
            name: "qwen3-4b-2507",
            description: "qwen3-4b with a rotary base of 5e6",
            builder: Qwen3Config::qwen3_4b_2507,
        },
        &StaticPreFabConfig {
            name: "qwen3-8b",
            description: "4096 wide, 36 layers, 32 query / 8 key-value heads, a head of its own",
            builder: Qwen3Config::qwen3_8b,
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::gpts::qwen3::Qwen3Meta;

    #[test]
    fn test_prefabs() {
        let config = QWEN3_PREFABS.expect_lookup_prefab("qwen3-4b").to_config();
        assert_eq!(config.d_model(), 2560);
        assert_eq!(config.n_layers(), 36);
        let config = QWEN3_PREFABS.expect_lookup_prefab("qwen3-8b").to_config();
        assert_eq!(config.d_model(), 4096);
        assert!(!config.tied_head);
        assert!(QWEN3_PREFABS.try_lookup_prefab("qwen3-14b").is_err());
    }
}
