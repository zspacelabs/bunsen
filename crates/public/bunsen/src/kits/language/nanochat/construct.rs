//! A loaded nanochat model, with the files a decode reads beside it.
use super::NanoChatGpt;
use crate::kits::support::{
    construct::{
        LmConstruct,
        LmPolicy,
    },
    model_config::{
        GenerationPolicy,
        SamplingPolicy,
    },
};

/// The key of the state dict, `model_<step>.pt`, in a nanochat resource set.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the checkpoint's meta, `meta_<step>.json`, in a nanochat
/// resource set.
pub const META: &str = "meta";

/// The key of the tokenizer, `tokenizer.pkl` (a pickled `tiktoken`
/// `Encoding`), in a [`NanoChatConstruct`]'s files and a nanochat resource
/// construct.
pub const TOKENIZER: &str = "tokenizer";

/// The key of the tokenizer's byte lengths, `token_bytes.pt` (one per token,
/// for upstream's bits-per-byte metric), in a [`NanoChatConstruct`]'s files and
/// a nanochat resource set.
pub const TOKEN_BYTES: &str = "token_bytes";

/// How a nanochat pretrained is run: the [`LmPolicy`] its presets patch,
/// over [`nanochat_default_policy`].
pub type NanoChatPolicy = LmPolicy;

/// The sampling upstream's chat scripts reply under: temperature 0.6, top-k
/// 50 (nanochat's checkpoints carry no generation config).
pub const NANOCHAT_SAMPLING: SamplingPolicy = SamplingPolicy {
    temperature: 0.6,
    top_k: Some(50),
    top_p: None,
    min_p: None,
};

/// The policy a nanochat pretrained resolves its presets over: its
/// generation [`NANOCHAT_SAMPLING`]. Its stop tokens are its tokenizer's
/// (`<|assistant_end|>`, `<|bos|>`), which the kit does not read.
pub fn nanochat_default_policy() -> NanoChatPolicy {
    LmPolicy {
        generation: GenerationPolicy {
            sampling: Some(NANOCHAT_SAMPLING),
            ..Default::default()
        },
    }
}

/// A nanochat model, the positions it was trained on (its meta's
/// `sequence_len`; the model's rotary table reaches ten times it), its
/// policy, and its checkpoint's other files, unread: what a pretrained loads
/// to ([`LmConstruct`]). The tokenizer, [`TOKENIZER`], is the caller's to
/// read.
///
/// [`default_nanochat_factory`](super::pretrained::default_nanochat_factory)
/// builds one from a name.
pub type NanoChatConstruct = LmConstruct<NanoChatGpt>;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::language::nanochat::NanoChatGptContractConfig,
        support::testing::cpu_device,
    };

    #[test]
    fn test_files_by_key() {
        let model: NanoChatGpt = NanoChatGptContractConfig::new()
            .with_vocab_size(16)
            .with_n_layer(1)
            .with_n_head(2)
            .with_n_kv_head(2)
            .with_n_embed(8)
            .with_init_seq_len(4)
            .init(&cpu_device());
        let bundle = NanoChatConstruct::new(model)
            .with_max_positions(Some(4))
            .with_file(TOKENIZER, "/x/tokenizer.pkl");
        assert_eq!(bundle.max_positions, Some(4));
        assert_eq!(bundle.tokenizer(), Some(Path::new("/x/tokenizer.pkl")));
        assert_eq!(bundle.file(TOKEN_BYTES), None);
        assert_eq!(
            nanochat_default_policy().generation.sampling,
            Some(NANOCHAT_SAMPLING)
        );
    }
}
