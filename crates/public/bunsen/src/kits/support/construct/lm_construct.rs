//! A loaded language model, with the files a decode reads beside it.
use alloc::collections::BTreeMap;
use std::path::{
    Path,
    PathBuf,
};

use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    data::catalog::presets::PresetSelection,
    kits::support::model_config::GenerationPolicy,
};

/// How a language model's pretrained is run: the config its presets and
/// probes patch.
///
/// Its `generation` is the [`GenerationPolicy`] a chat driver reads: the
/// stop tokens and sampling its `generation_config.json` names, and the
/// template's kwargs its presets set (Qwen3's `thinking-mode`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LmPolicy {
    /// How it generates a reply.
    pub generation: GenerationPolicy,
}

/// The key of the tokenizer, `tokenizer.json`, in an [`LmConstruct`]'s files
/// and a language model's resource bundle.
pub const TOKENIZER: &str = "tokenizer";

/// The key of the tokenizer's config, `tokenizer_config.json` (its added
/// tokens, and the chat template where it holds one), in an
/// [`LmConstruct`]'s files and a language model's resource bundle.
pub const TOKENIZER_CONFIG: &str = "tokenizer_config";

/// The key of the generation defaults, `generation_config.json` (the stop
/// tokens and the sampling parameters), in an [`LmConstruct`]'s files and a
/// language model's resource bundle.
pub const GENERATION_CONFIG: &str = "generation_config";

/// The key of a chat template kept in a file of its own,
/// `chat_template.jinja`, in an [`LmConstruct`]'s files and a language
/// model's resource bundle.
pub const CHAT_TEMPLATE: &str = "chat_template";

/// A language model, the positions it was trained on, and its checkpoint's
/// other files, unread: what a language model's pretrained loads to (a
/// Qwen3's, a Gemma's).
///
/// The kit reads only what builds the model: ids in, logits out, and the
/// [`LmPolicy`] its presets resolved to. The tokenizer and its config (its
/// added tokens and chat template) are the caller's to read, through
/// [`tokenizer`](Self::tokenizer) and
/// [`tokenizer_config`](Self::tokenizer_config). The model decodes
/// through a [`KVCache`](crate::ops::transformers::attention::KVCache) the
/// caller makes from it
/// ([`KVCacheModel`](crate::ops::transformers::attention::KVCacheModel)).
///
/// A kit's factory builds one from a name
/// ([`default_qwen3_factory`](crate::kits::language::qwen3::pretrained::default_qwen3_factory),
/// say).
#[derive(Debug)]
pub struct LmConstruct<M> {
    /// The model.
    pub model: M,

    /// The positions the checkpoint was trained on
    /// (`max_position_embeddings`), when it says.
    pub max_positions: Option<usize>,

    /// The checkpoint's files, local, by resource key: [`TOKENIZER`],
    /// [`TOKENIZER_CONFIG`] and [`GENERATION_CONFIG`] among them, where it
    /// has them.
    pub files: BTreeMap<String, PathBuf>,

    /// How it is run: what its presets resolved to.
    pub policy: LmPolicy,

    /// The presets applied, and the lookup's patch.
    pub presets: PresetSelection,
}

impl<M> LmConstruct<M> {
    /// A construct of `model` alone: no trained length, no files.
    pub fn new(model: M) -> Self {
        Self {
            model,
            max_positions: None,
            files: BTreeMap::new(),
            policy: LmPolicy::default(),
            presets: PresetSelection::default(),
        }
    }

    /// Sets the policy, and the presets that resolved it.
    pub fn with_policy(
        self,
        policy: LmPolicy,
        presets: PresetSelection,
    ) -> Self {
        Self {
            policy,
            presets,
            ..self
        }
    }

    /// Sets the positions the checkpoint was trained on.
    pub fn with_max_positions(
        self,
        max_positions: Option<usize>,
    ) -> Self {
        Self {
            max_positions,
            ..self
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

    /// The `tokenizer.json`, under [`TOKENIZER`].
    pub fn tokenizer(&self) -> Option<&Path> {
        self.file(TOKENIZER)
    }

    /// The `tokenizer_config.json`, under [`TOKENIZER_CONFIG`].
    pub fn tokenizer_config(&self) -> Option<&Path> {
        self.file(TOKENIZER_CONFIG)
    }

    /// The `generation_config.json`, under [`GENERATION_CONFIG`].
    pub fn generation_config(&self) -> Option<&Path> {
        self.file(GENERATION_CONFIG)
    }

    /// The `chat_template.jinja`, under [`CHAT_TEMPLATE`], when the template
    /// is a file of its own.
    pub fn chat_template(&self) -> Option<&Path> {
        self.file(CHAT_TEMPLATE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::language::qwen3::{
            Qwen3Config,
            Qwen3Model,
        },
        support::testing::cpu_device,
    };

    #[test]
    fn test_files_by_key() {
        let model: Qwen3Model = Qwen3Config::new(40, 12, 1, 3, 1, 6, 20).init(&cpu_device());
        let bundle = LmConstruct::new(model)
            .with_max_positions(Some(512))
            .with_file(TOKENIZER, "/x/tokenizer.json");
        assert_eq!(bundle.max_positions, Some(512));
        assert_eq!(bundle.tokenizer(), Some(Path::new("/x/tokenizer.json")));
        assert_eq!(bundle.tokenizer_config(), None);
        assert_eq!(bundle.generation_config(), None);
        assert_eq!(bundle.chat_template(), None);
    }
}
