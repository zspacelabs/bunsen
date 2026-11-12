//! # Gemma prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
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
impl Prefabs for GemmaConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("gemma-3-270m", Self::gemma_3_270m),
        ("gemma-3-1b", Self::gemma_3_1b),
        ("gemma-3-4b", Self::gemma_3_4b),
        ("gemma-2-2b", Self::gemma_2_2b),
        ("gemma-2-9b", Self::gemma_2_9b),
        ("gemma-2b", Self::gemma_2b),
        ("gemma-7b", Self::gemma_7b),
    ];
    const PREFAB_TABLE: &'static str = "gemma";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::models::language_decoder::LmDecoderMeta,
        kits::language::gemma::GemmaLayerKind,
    };

    /// Each generation's prefabs carry its features: Gemma 3's pattern and
    /// two tables, Gemma 2's alternation and caps, Gemma 1's plain stack.
    #[test]
    fn test_prefabs() {
        let g3 = GemmaConfig::expect_prefab("gemma-3-4b");
        assert_eq!((g3.d_model(), g3.n_layers()), (2560, 34));
        assert_eq!(g3.layer_kind(5), GemmaLayerKind::Full);
        assert_eq!(g3.layer_kind(6), GemmaLayerKind::Sliding);
        assert!(g3.qk_norm && g3.post_norms && g3.rope_local_theta.is_some());
        let g2 = GemmaConfig::expect_prefab("gemma-2-2b");
        assert_eq!(g2.layer_kind(0), GemmaLayerKind::Sliding);
        assert_eq!(g2.layer_kind(1), GemmaLayerKind::Full);
        assert_eq!(
            (g2.attn_softcap, g2.final_softcap),
            (Some(50.0), Some(30.0))
        );
        let g1 = GemmaConfig::expect_prefab("gemma-7b");
        assert!(!g1.post_norms && g1.layer_kinds.is_empty());
        for (_, prefab) in GemmaConfig::PREFABS {
            prefab().stack().unwrap();
        }
        assert!(GemmaConfig::try_prefab("gemma-3-27b").is_err());
    }
}
