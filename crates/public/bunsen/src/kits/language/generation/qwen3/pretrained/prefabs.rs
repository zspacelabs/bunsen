//! # Qwen3 prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::language::generation::qwen3::Qwen3ContractConfig,
};

/// The prefab geometries, by name: each a [`Qwen3ContractConfig`] of one
/// checkpoint.
impl Qwen3ContractConfig {
    /// The text model of `Qwen3-ASR-0.6B` (`Qwen3-0.6B`'s geometry): 151936
    /// tokens, 28 layers of width 1024, 16 query and 8 key/value heads of
    /// width 128, a 3072-wide feed-forward, and a rotary base of 10^6.
    pub fn qwen3_0_6b() -> Self {
        Self::new(151_936, 1024, 28, 16, 8, 128, 3072)
    }

    /// `Qwen3-1.7B`'s geometry: the text model of `Qwen3-ASR-1.7B`.
    pub fn qwen3_1_7b() -> Self {
        Self::new(151_936, 2048, 28, 16, 8, 128, 6144)
    }

    /// `Qwen3-4B`'s geometry: 36 layers of width 2560, 32 query and 8
    /// key/value heads of width 128, and a 9728-wide feed-forward.
    pub fn qwen3_4b() -> Self {
        Self::new(151_936, 2560, 36, 32, 8, 128, 9728)
    }

    /// The geometry of `Qwen3-4B-Instruct-2507` and `Qwen3-4B-Thinking-2507`:
    /// [`qwen3_4b`](Self::qwen3_4b)'s, with a rotary base of 5 * 10^6.
    pub fn qwen3_4b_2507() -> Self {
        Self::qwen3_4b().with_rope_theta(5_000_000.0)
    }

    /// `Qwen3-8B`'s geometry: 36 layers of width 4096, 32 query and 8
    /// key/value heads of width 128, a 12288-wide feed-forward, and an output
    /// head of its own.
    pub fn qwen3_8b() -> Self {
        Self::new(151_936, 4096, 36, 32, 8, 128, 12288).with_tied_head(false)
    }
}

/// The Qwen3 prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched. A `-Base` checkpoint has its chat model's geometry.
impl Prefabs for Qwen3ContractConfig {
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
    use crate::{
        blocks::transformers::models::language_decoder::LmDecoderMeta,
        errors::testing::ErrorMatcher,
    };

    #[test]
    fn test_prefabs() {
        let config = Qwen3ContractConfig::expect_prefab("qwen3-4b");
        assert_eq!(config.d_model(), 2560);
        assert_eq!(config.n_layers(), 36);
        let config = Qwen3ContractConfig::expect_prefab("qwen3-8b");
        assert_eq!(config.d_model(), 4096);
        assert!(!config.tied_head);
        ErrorMatcher::lookup().assert_err(&Qwen3ContractConfig::try_prefab("qwen3-14b"));
    }
}
