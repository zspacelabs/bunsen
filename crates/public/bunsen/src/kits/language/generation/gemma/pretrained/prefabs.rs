//! # Gemma prefabs
use crate::{
    blocks::transformers::embedding::RopeScaling,
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::language::generation::gemma::GemmaContractConfig,
};

/// The prefab geometries, by name: each a [`GemmaContractConfig`] of one
/// checkpoint.
impl GemmaContractConfig {
    /// A Gemma 3: the sandwich, QK-norm, a `query_pre_attn_scalar` of 256,
    /// five sliding layers to one full one over a window of `window`, the
    /// local table at 10^4 and the global one at 10^6.
    fn gemma3(
        n_vocab: usize,
        d_model: usize,
        n_layers: usize,
        heads: [usize; 3],
        d_ff: usize,
        window: usize,
    ) -> Self {
        let [n_heads, n_kv_heads, head_dim] = heads;
        Self::new(
            n_vocab, d_model, n_layers, n_heads, n_kv_heads, head_dim, d_ff,
        )
        .with_post_norms(true)
        .with_qk_norm(true)
        .with_query_pre_attn_scalar(Some(256.0))
        .with_layer_kinds(Self::every_nth_full(n_layers, 6))
        .with_sliding_window(Some(window))
        .with_rope_theta(1_000_000.0)
        .with_rope_local_theta(Some(10_000.0))
    }

    /// A Gemma 2: the sandwich, a `query_pre_attn_scalar` of 256, sliding
    /// and full layers alternating over a window of 4096, one table at
    /// 10^4, soft-caps of 50 on the scores and 30 on the logits.
    fn gemma2(
        d_model: usize,
        n_layers: usize,
        heads: [usize; 3],
        d_ff: usize,
    ) -> Self {
        let [n_heads, n_kv_heads, head_dim] = heads;
        Self::new(
            256_000, d_model, n_layers, n_heads, n_kv_heads, head_dim, d_ff,
        )
        .with_post_norms(true)
        .with_query_pre_attn_scalar(Some(256.0))
        .with_layer_kinds(Self::every_nth_full(n_layers, 2))
        .with_sliding_window(Some(4096))
        .with_attn_softcap(Some(50.0))
        .with_final_softcap(Some(30.0))
    }

    /// `gemma-3-270m`'s geometry: 18 layers of width 640, 4 query heads and
    /// one key/value head of width 256, a window of 512.
    pub fn gemma_3_270m() -> Self {
        Self::gemma3(262_144, 640, 18, [4, 1, 256], 2048, 512)
    }

    /// `gemma-3-1b`'s geometry: 26 layers of width 1152, 4 query heads and
    /// one key/value head of width 256, a window of 512.
    pub fn gemma_3_1b() -> Self {
        Self::gemma3(262_144, 1152, 26, [4, 1, 256], 6912, 512)
    }

    /// `gemma-3-4b`'s text model: 34 layers of width 2560, 8 query and 4
    /// key/value heads of width 256, a window of 1024, its global table
    /// scaled linearly by 8, and 262208 tokens (the image tokens' rows
    /// among them).
    pub fn gemma_3_4b() -> Self {
        Self::gemma3(262_208, 2560, 34, [8, 4, 256], 10240, 1024)
            .with_rope_scaling(RopeScaling::Linear { factor: 8.0 })
    }

    /// `gemma-2-2b`'s geometry: 26 layers of width 2304, 8 query and 4
    /// key/value heads of width 256.
    pub fn gemma_2_2b() -> Self {
        Self::gemma2(2304, 26, [8, 4, 256], 9216)
    }

    /// `gemma-2-9b`'s geometry: 42 layers of width 3584, 16 query and 8
    /// key/value heads of width 256.
    pub fn gemma_2_9b() -> Self {
        Self::gemma2(3584, 42, [16, 8, 256], 14336)
    }

    /// `gemma-2b`'s geometry (Gemma 1): 18 layers of width 2048, 8 query
    /// heads and one key/value head of width 256.
    pub fn gemma_2b() -> Self {
        Self::new(256_000, 2048, 18, 8, 1, 256, 16384)
    }

    /// `gemma-7b`'s geometry (Gemma 1): 28 layers of width 3072, 16 heads of
    /// width 256.
    pub fn gemma_7b() -> Self {
        Self::new(256_000, 3072, 28, 16, 16, 256, 24576)
    }
}

/// The Gemma prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched; a checkpoint's `-it` (or `-pt`) sibling has its
/// geometry. The `config.json`s are not committed: the validation crate
/// checks each prefab against its rows' files.
impl Prefabs for GemmaContractConfig {
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
        errors::testing::ErrorMatcher,
        kits::language::generation::gemma::GemmaLayerKind,
    };

    /// Each generation's prefabs carry its features: Gemma 3's pattern and
    /// two tables, Gemma 2's alternation and caps, Gemma 1's plain stack.
    #[test]
    fn test_prefabs() {
        let g3 = GemmaContractConfig::expect_prefab("gemma-3-4b");
        assert_eq!((g3.d_model(), g3.n_layers()), (2560, 34));
        assert_eq!(g3.layer_kind(5), GemmaLayerKind::Full);
        assert_eq!(g3.layer_kind(6), GemmaLayerKind::Sliding);
        assert!(g3.qk_norm && g3.post_norms && g3.rope_local_theta.is_some());
        let g2 = GemmaContractConfig::expect_prefab("gemma-2-2b");
        assert_eq!(g2.layer_kind(0), GemmaLayerKind::Sliding);
        assert_eq!(g2.layer_kind(1), GemmaLayerKind::Full);
        assert_eq!(
            (g2.attn_softcap, g2.final_softcap),
            (Some(50.0), Some(30.0))
        );
        let g1 = GemmaContractConfig::expect_prefab("gemma-7b");
        assert!(!g1.post_norms && g1.layer_kinds.is_empty());
        for (_, prefab) in GemmaContractConfig::PREFABS {
            prefab().try_stack().unwrap();
        }
        ErrorMatcher::lookup().assert_err(&GemmaContractConfig::try_prefab("gemma-3-27b"));
    }
}
