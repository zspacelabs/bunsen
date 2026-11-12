//! # nanochat prefabs
use crate::{
    data::pretrained::{
        PrefabTable,
        Prefabs,
    },
    kits::language::nanochat::NanoChatGptContractConfig,
};

/// A nanochat of upstream's geometry rule at `depth`: `depth * 64` wide,
/// heads of 128 (upstream's `aspect_ratio` 64 and `head_dim` 128), as many
/// KV heads as query heads, a context of 2048 and `vocab_size` tokens.
fn upstream_geometry(
    depth: usize,
    vocab_size: usize,
) -> NanoChatGptContractConfig {
    let n_embed = depth * 64;
    NanoChatGptContractConfig::new()
        .with_init_seq_len(2048)
        .with_max_seq_len_factor(10)
        .with_vocab_size(vocab_size)
        .with_n_layer(depth)
        .with_n_head(n_embed / 128)
        .with_n_kv_head(n_embed / 128)
        .with_n_embed(n_embed)
}

impl NanoChatGptContractConfig {
    /// The `nanochat-d20` prefab: upstream's 2025 model at depth 20, 1280
    /// wide, 10 heads, 65536 tokens.
    pub fn nanochat_d20() -> Self {
        upstream_geometry(20, 65536)
    }

    /// The `nanochat-d18-2026` prefab: upstream's 2026 model at depth 18,
    /// 1152 wide, 9 heads, 32768 tokens, every layer's window long.
    pub fn nanochat_d18_2026() -> Self {
        upstream_geometry(18, 32768).with_upstream_2026("L")
    }
}

/// The nanochat prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched; a checkpoint's own meta still gives the geometry it
/// loads into.
impl Prefabs for NanoChatGptContractConfig {
    const PREFABS: PrefabTable<Self> = &[
        ("nanochat-d20", Self::nanochat_d20),
        ("nanochat-d18-2026", Self::nanochat_d18_2026),
    ];
    const PREFAB_TABLE: &'static str = "nanochat";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::language::nanochat::NanoChatGptMeta;

    #[test]
    fn test_prefabs() {
        let config = NanoChatGptContractConfig::expect_prefab("nanochat-d20");
        assert_eq!(
            (config.n_embed(), config.n_layer(), config.n_head()),
            (1280, 20, 10)
        );
        assert!(!config.residual_lambdas);

        let config = NanoChatGptContractConfig::expect_prefab("nanochat-d18-2026");
        assert_eq!(
            (config.n_embed(), config.n_layer(), config.n_head()),
            (1152, 18, 9)
        );
        assert_eq!(config.window_pattern.as_deref(), Some("L"));
        assert!(NanoChatGptContractConfig::try_prefab("nanochat-d32").is_err());
    }
}
