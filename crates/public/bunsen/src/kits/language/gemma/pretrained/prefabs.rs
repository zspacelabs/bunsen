//! # Gemma prefabs
use crate::{
    data::pretrained::{
        StaticPreFabConfig,
        StaticPreFabMap,
    },
    kits::language::gemma::GemmaConfig,
};

/// The Gemma prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched; a checkpoint's `-it` (or `-pt`) sibling has its
/// geometry. The `config.json`s are not committed: the validation crate
/// checks each prefab against its rows' files.
pub static GEMMA_PREFABS: StaticPreFabMap<GemmaConfig> = StaticPreFabMap {
    name: "gemma",
    description: "Gemma 1, 2 and 3 text geometries, as transformers' configs have them",
    items: &[
        &StaticPreFabConfig {
            name: "gemma-3-270m",
            description: "Gemma 3, 640 wide, 18 layers, 4 query / 1 key-value heads",
            builder: GemmaConfig::gemma_3_270m,
        },
        &StaticPreFabConfig {
            name: "gemma-3-1b",
            description: "Gemma 3, 1152 wide, 26 layers, 4 query / 1 key-value heads",
            builder: GemmaConfig::gemma_3_1b,
        },
        &StaticPreFabConfig {
            name: "gemma-3-4b",
            description: "Gemma 3, 2560 wide, 34 layers, 8 query / 4 key-value heads, the text model",
            builder: GemmaConfig::gemma_3_4b,
        },
        &StaticPreFabConfig {
            name: "gemma-2-2b",
            description: "Gemma 2, 2304 wide, 26 layers, 8 query / 4 key-value heads",
            builder: GemmaConfig::gemma_2_2b,
        },
        &StaticPreFabConfig {
            name: "gemma-2-9b",
            description: "Gemma 2, 3584 wide, 42 layers, 16 query / 8 key-value heads",
            builder: GemmaConfig::gemma_2_9b,
        },
        &StaticPreFabConfig {
            name: "gemma-2b",
            description: "Gemma 1, 2048 wide, 18 layers, 8 query / 1 key-value heads",
            builder: GemmaConfig::gemma_2b,
        },
        &StaticPreFabConfig {
            name: "gemma-7b",
            description: "Gemma 1, 3072 wide, 28 layers, 16 heads",
            builder: GemmaConfig::gemma_7b,
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::models::LmDecoderMeta,
        kits::language::gemma::GemmaLayerKind,
    };

    /// Each generation's prefabs carry its features: Gemma 3's pattern and
    /// two tables, Gemma 2's alternation and caps, Gemma 1's plain stack.
    #[test]
    fn test_prefabs() {
        let g3 = GEMMA_PREFABS.expect_lookup_prefab("gemma-3-4b").to_config();
        assert_eq!((g3.d_model(), g3.n_layers()), (2560, 34));
        assert_eq!(g3.layer_kind(5), GemmaLayerKind::Full);
        assert_eq!(g3.layer_kind(6), GemmaLayerKind::Sliding);
        assert!(g3.qk_norm && g3.post_norms && g3.rope_local_theta.is_some());
        let g2 = GEMMA_PREFABS.expect_lookup_prefab("gemma-2-2b").to_config();
        assert_eq!(g2.layer_kind(0), GemmaLayerKind::Sliding);
        assert_eq!(g2.layer_kind(1), GemmaLayerKind::Full);
        assert_eq!(
            (g2.attn_softcap, g2.final_softcap),
            (Some(50.0), Some(30.0))
        );
        let g1 = GEMMA_PREFABS.expect_lookup_prefab("gemma-7b").to_config();
        assert!(!g1.post_norms && g1.layer_kinds.is_empty());
        for prefab in GEMMA_PREFABS.iter() {
            prefab.to_config().stack().unwrap();
        }
        assert!(GEMMA_PREFABS.try_lookup_prefab("gemma-3-27b").is_err());
    }
}
