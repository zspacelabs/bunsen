//! The config file's typed mirror.

use std::path::Path;

use serde::Deserialize;

use super::config::NanoChatGeneration;
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
        sys_at,
    },
    kits::language::generation::nanochat::NanoChatGptContractConfig,
};

/// A nanochat checkpoint's `meta_<step>.json`: the parts bunsen reads. The
/// rest (the training run's settings, its losses, a chat template some
/// uploads add) is ignored.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct NanoChatMetaMirror {
    /// The training step the checkpoint is of, when the meta says.
    #[serde(default)]
    pub step: Option<u64>,

    /// The model's geometry, as upstream's `GPTConfig`.
    pub model_config: NanoChatModelConfigMirror,
}

/// Upstream's `GPTConfig`, as a checkpoint's meta writes it: upstream's
/// names.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct NanoChatModelConfigMirror {
    /// The training context, in tokens; the rotary table is ten times it.
    pub sequence_len: usize,

    /// The vocabulary size, before any padding.
    pub vocab_size: usize,

    /// The number of layers.
    pub n_layer: usize,

    /// The number of query heads.
    pub n_head: usize,

    /// The number of key/value heads.
    pub n_kv_head: usize,

    /// The model width.
    pub n_embd: usize,

    /// The 2026 model's sliding-window pattern (`"SSSL"`, `"L"`); absent
    /// in the 2025 model's meta.
    #[serde(default)]
    pub window_pattern: Option<String>,
}

impl NanoChatMetaMirror {
    /// Parses a `meta_<step>.json`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it is not a nanochat meta.
    pub fn from_json(json: &str) -> BunsenResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            BunsenError::invalid_resource("not a nanochat checkpoint meta").with_cause(e)
        })
    }

    /// Reads and parses the `meta_<step>.json` at `path`.
    ///
    /// # Errors
    /// As [`sys_at`] when the file cannot be read; as
    /// [`from_json`](Self::from_json) when it is not a nanochat meta.
    pub fn from_path(path: &Path) -> BunsenResult<Self> {
        let json = std::fs::read_to_string(path).map_err(sys_at("read", path))?;
        Self::from_json(&json).with_context(|| path.display().to_string())
    }
}

impl NanoChatMetaMirror {
    /// Which of upstream's models the checkpoint whose state dict holds `names`
    /// and whose meta this is: the 2026 one when the state dict has its
    /// parameters and the meta its window pattern, the 2025 one when it has
    /// neither.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) when it has
    /// some of the 2026 parameters and not others, or when the state dict and
    /// the meta disagree: a checkpoint of one of upstream's commits between the
    /// two, which bunsen does not model.
    pub fn generation<S: AsRef<str>>(
        &self,
        names: &[S],
    ) -> BunsenResult<NanoChatGeneration> {
        let has = |name: &str| names.iter().any(|n| n.as_ref() == name);
        let present: Vec<&str> = UPSTREAM_2026_MARKERS
            .iter()
            .copied()
            .filter(|m| has(m))
            .collect();
        let value_embeds = names
            .iter()
            .any(|n| n.as_ref().starts_with("value_embeds."));
        let windowed = self.model_config.window_pattern.is_some();
        match (present.len(), value_embeds, windowed) {
            (0, false, false) => Ok(NanoChatGeneration::Upstream2025),
            (n, true, true) if n == UPSTREAM_2026_MARKERS.len() => {
                Ok(NanoChatGeneration::Upstream2026)
            }
            _ => Err(BunsenError::unsupported(format!(
                "a nanochat checkpoint between upstream's 2025 and 2026 models: it has {} of \
                 {UPSTREAM_2026_MARKERS:?}, {} value embeddings, and {} window pattern",
                if present.is_empty() {
                    "none".to_string()
                } else {
                    format!("{present:?}")
                },
                if value_embeds { "" } else { "no" },
                if windowed { "a" } else { "no" },
            ))),
        }
    }
}

impl NanoChatMetaMirror {
    /// The config of a checkpoint of `generation` whose meta is `meta`: its
    /// geometry, `sequence_len` the initial sequence length (the rotary table
    /// ten times it, as upstream's), and the generation's options.
    pub fn gpt_config(
        &self,
        generation: NanoChatGeneration,
    ) -> NanoChatGptContractConfig {
        let m = &self.model_config;
        let config = NanoChatGptContractConfig::new()
            .with_init_seq_len(m.sequence_len)
            .with_max_seq_len_factor(10)
            .with_vocab_size(m.vocab_size)
            .with_n_layer(m.n_layer)
            .with_n_head(m.n_head)
            .with_n_kv_head(m.n_kv_head)
            .with_n_embed(m.n_embd);
        match generation {
            NanoChatGeneration::Upstream2025 => config,
            NanoChatGeneration::Upstream2026 => {
                config.with_upstream_2026(m.window_pattern.clone().unwrap_or_else(|| "L".into()))
            }
        }
    }
}

/// The state-dict names only upstream's 2026 model has: one per 2026 part
/// that carries a parameter.
pub(super) const UPSTREAM_2026_MARKERS: [&str; 5] = [
    "resid_lambdas",
    "x0_lambdas",
    "smear_gate.weight",
    "smear_lambda",
    "backout_lambda",
];
