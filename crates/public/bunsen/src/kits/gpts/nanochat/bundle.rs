//! A loaded nanochat model, with the files a decode reads beside it.
use std::{
    collections::BTreeMap,
    path::{
        Path,
        PathBuf,
    },
};

use super::{
    NanoChatGpt,
    weights::{
        NanoChatGeneration,
        NanoChatMeta,
    },
};

/// The key of the state dict, `model_<step>.pt`, in a nanochat resource map.
pub const CHECKPOINT: &str = "checkpoint";

/// The key of the checkpoint's meta, `meta_<step>.json`, in a nanochat
/// resource map.
pub const META: &str = "meta";

/// The key of the tokenizer, `tokenizer.pkl` (a pickled `tiktoken`
/// `Encoding`), in a [`NanoChatBundle`]'s files and a nanochat resource map.
pub const TOKENIZER: &str = "tokenizer";

/// The key of the tokenizer's byte lengths, `token_bytes.pt` (one per token,
/// for upstream's bits-per-byte metric), in a [`NanoChatBundle`]'s files and
/// a nanochat resource map.
pub const TOKEN_BYTES: &str = "token_bytes";

/// A nanochat model, which of upstream's models it is, its meta, and its
/// checkpoint's other files, unread: what a pretrained loads to.
///
/// The kit reads only what builds the model: ids in, logits out. The
/// tokenizer is the caller's to read, through [`tokenizer`](Self::tokenizer).
/// The model decodes through a
/// [`NanoChatDecodeState`](super::NanoChatDecodeState) the caller makes from
/// it.
///
/// [`default_nanochat_factory`](super::pretrained::default_nanochat_factory)
/// builds one from a name.
#[derive(Debug)]
pub struct NanoChatBundle {
    /// The model.
    pub model: NanoChatGpt,

    /// Which of upstream's models the checkpoint is.
    pub generation: NanoChatGeneration,

    /// The checkpoint's meta.
    pub meta: NanoChatMeta,

    /// The checkpoint's files, local, by resource key: [`TOKENIZER`] and
    /// [`TOKEN_BYTES`] among them, where it has them.
    pub files: BTreeMap<String, PathBuf>,
}

impl NanoChatBundle {
    /// A bundle of `model`, of `generation`, with its `meta`, and no files.
    pub fn new(
        model: NanoChatGpt,
        generation: NanoChatGeneration,
        meta: NanoChatMeta,
    ) -> Self {
        Self {
            model,
            generation,
            meta,
            files: BTreeMap::new(),
        }
    }

    /// Adds the file `path` under `key`.
    pub fn with_file(
        mut self,
        key: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        self.files.insert(key.into(), path.into());
        self
    }

    /// The file under `key`, when the checkpoint has it.
    pub fn file(
        &self,
        key: &str,
    ) -> Option<&Path> {
        self.files.get(key).map(PathBuf::as_path)
    }

    /// The `tokenizer.pkl`, under [`TOKENIZER`].
    pub fn tokenizer(&self) -> Option<&Path> {
        self.file(TOKENIZER)
    }

    /// The `token_bytes.pt`, under [`TOKEN_BYTES`].
    pub fn token_bytes(&self) -> Option<&Path> {
        self.file(TOKEN_BYTES)
    }

    /// The context the checkpoint was trained on, in tokens: its meta's
    /// `sequence_len`. The model's rotary table reaches ten times it.
    pub fn sequence_len(&self) -> usize {
        self.meta.model_config.sequence_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::gpts::nanochat::{
            NanoChatGptContractConfig,
            weights::NanoChatModelConfig,
        },
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
        let meta = NanoChatMeta {
            step: Some(1),
            model_config: NanoChatModelConfig {
                sequence_len: 4,
                vocab_size: 16,
                n_layer: 1,
                n_head: 2,
                n_kv_head: 2,
                n_embd: 8,
                window_pattern: None,
            },
        };
        let bundle = NanoChatBundle::new(model, NanoChatGeneration::Upstream2025, meta)
            .with_file(TOKENIZER, "/x/tokenizer.pkl");
        assert_eq!(bundle.sequence_len(), 4);
        assert_eq!(bundle.tokenizer(), Some(Path::new("/x/tokenizer.pkl")));
        assert_eq!(bundle.token_bytes(), None);
    }
}
