//! # Qwen3.5 prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::language::generation::qwen3_5::Qwen3_5ContractConfig,
};

/// The prefab geometries, by name: each a [`Qwen3_5ContractConfig`] of one
/// checkpoint.
impl Qwen3_5ContractConfig {
    /// A published dense Qwen3.5's text model: 248320 tokens, full-attention
    /// heads of width 256 over `heads` (query, key/value), 16 linear key
    /// heads and `linear_value_heads` value heads of width 128, every fourth
    /// layer full, a rotary base of 10^7 over a quarter of each head.
    fn published(
        d_model: usize,
        n_layers: usize,
        heads: [usize; 2],
        d_ff: usize,
        linear_value_heads: usize,
        tied_head: bool,
    ) -> Self {
        let [n_heads, n_kv_heads] = heads;
        Self::new(
            248_320,
            d_model,
            n_layers,
            n_heads,
            n_kv_heads,
            256,
            d_ff,
            16,
            linear_value_heads,
            128,
            128,
        )
        .with_tied_head(tied_head)
        .with_every_nth_full()
    }

    /// `Qwen3.5-0.8B`'s geometry: 24 layers of width 1024, 8 query and 2
    /// key/value heads, 16 linear value heads, a tied head.
    pub fn qwen3_5_0_8b() -> Self {
        Self::published(1024, 24, [8, 2], 3584, 16, true)
    }

    /// `Qwen3.5-2B`'s geometry: 24 layers of width 2048, 8 query and 2
    /// key/value heads, 16 linear value heads, a tied head.
    pub fn qwen3_5_2b() -> Self {
        Self::published(2048, 24, [8, 2], 6144, 16, true)
    }

    /// `Qwen3.5-4B`'s geometry: 32 layers of width 2560, 16 query and 4
    /// key/value heads, 32 linear value heads, a tied head.
    pub fn qwen3_5_4b() -> Self {
        Self::published(2560, 32, [16, 4], 9216, 32, true)
    }

    /// `Qwen3.5-9B`'s geometry: 32 layers of width 4096, 16 query and 4
    /// key/value heads, 32 linear value heads, a head of its own.
    pub fn qwen3_5_9b() -> Self {
        Self::published(4096, 32, [16, 4], 12288, 32, false)
    }

    /// `Qwen3.5-27B`'s geometry: 64 layers of width 5120, 24 query and 4
    /// key/value heads, 48 linear value heads, a head of its own.
    pub fn qwen3_5_27b() -> Self {
        Self::published(5120, 64, [24, 4], 17408, 48, false)
    }
}

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
