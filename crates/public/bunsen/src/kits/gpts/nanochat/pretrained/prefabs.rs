//! # nanochat prefabs
use crate::{
    data::pretrained::{
        StaticPreFabConfig,
        StaticPreFabMap,
    },
    kits::gpts::nanochat::NanoChatGptContractConfig,
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

/// Upstream's 2025 d20: 20 layers, 1280 wide, 10 heads, 65536 tokens.
fn nanochat_d20() -> NanoChatGptContractConfig {
    upstream_geometry(20, 65536)
}

/// Upstream's 2026 d18: 18 layers, 1152 wide, 9 heads, 32768 tokens, every
/// layer's window long.
fn nanochat_d18_2026() -> NanoChatGptContractConfig {
    upstream_geometry(18, 32768).with_upstream_2026("L")
}

/// The nanochat prefabs: the geometries bunsen's rows instantiate, by name,
/// without weights.
///
/// A pretrained row names its prefab, so a name means a shape before any
/// bytes are fetched; a checkpoint's own meta still gives the geometry it
/// loads into.
pub static NANOCHAT_PREFABS: StaticPreFabMap<NanoChatGptContractConfig> = StaticPreFabMap {
    name: "nanochat",
    description: "nanochat geometries of upstream's 2025 and 2026 models",
    items: &[
        &StaticPreFabConfig {
            name: "nanochat-d20",
            description: "upstream's 2025 model at depth 20: 1280 wide, 10 heads, 65536 tokens",
            builder: nanochat_d20,
        },
        &StaticPreFabConfig {
            name: "nanochat-d18-2026",
            description: "upstream's 2026 model at depth 18: 1152 wide, 9 heads, 32768 tokens, long windows",
            builder: nanochat_d18_2026,
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::gpts::nanochat::NanoChatGptMeta;

    #[test]
    fn test_prefabs() {
        let config = NANOCHAT_PREFABS
            .expect_lookup_prefab("nanochat-d20")
            .to_config();
        assert_eq!(
            (config.n_embed(), config.n_layer(), config.n_head()),
            (1280, 20, 10)
        );
        assert!(!config.residual_lambdas);

        let config = NANOCHAT_PREFABS
            .expect_lookup_prefab("nanochat-d18-2026")
            .to_config();
        assert_eq!(
            (config.n_embed(), config.n_layer(), config.n_head()),
            (1152, 18, 9)
        );
        assert_eq!(config.window_pattern.as_deref(), Some("L"));
        assert!(NANOCHAT_PREFABS.try_lookup_prefab("nanochat-d32").is_err());
    }
}
