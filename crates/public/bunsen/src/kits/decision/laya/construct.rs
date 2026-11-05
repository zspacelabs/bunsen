//! A loaded Laya model, with the files a prompt reads beside it.
use std::{
    collections::BTreeMap,
    path::{
        Path,
        PathBuf,
    },
};

use super::{
    LayaAgentConfig,
    LayaModel,
};
pub use crate::kits::support::construct::{
    TOKENIZER,
    TOKENIZER_CONFIG,
};

/// A Laya model, its head's `rl_agent_config.json` (its prompt budgets and
/// calibration), and its checkpoint's other files, unread: what a Laya
/// pretrained loads to.
///
/// [`default_laya_factory`](super::pretrained::default_laya_factory) builds
/// one from a name. Its [`TOKENIZER`] is the `tokenizer.json` a caller's
/// [`Tokenizer`](crate::kits::support::tokens::Tokenizer) reads, and its
/// [`TOKENIZER_CONFIG`] names the specials
/// ([`LayaSpecials::from_tokenizer_config`](super::LayaSpecials::from_tokenizer_config)).
#[derive(Debug)]
pub struct LayaConstruct {
    /// The model.
    pub model: LayaModel,

    /// The head's config: the prompt budgets and the temperatures.
    pub agent: LayaAgentConfig,

    /// The checkpoint's files, local, by resource key.
    pub files: BTreeMap<String, PathBuf>,
}

impl LayaConstruct {
    /// A construct of `model` and `agent`, with no files.
    pub fn new(
        model: LayaModel,
        agent: LayaAgentConfig,
    ) -> Self {
        Self {
            model,
            agent,
            files: BTreeMap::new(),
        }
    }

    /// With the file `path` under `key`.
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
}
