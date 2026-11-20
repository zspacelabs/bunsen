//! # nanochat as upstream's checkpoint manager saves it
//!
//! Upstream nanochat (`karpathy/nanochat`) writes a step as two files: the
//! model's state dict, `model_<step>.pt` (`torch.save`), and its
//! `meta_<step>.json`, whose `model_config` is the model's geometry. As
//! bunsen's [`NanoChatGpt`](crate::kits::language::nanochat::NanoChatGpt):
//!
//! - **Mirrors:** [`NanoChatMetaMirror`] (the meta) and
//!   [`NanoChatModelConfigMirror`] (its `model_config`, upstream's
//!   `GPTConfig`); [`generation`](NanoChatMetaMirror::generation) tells from
//!   the state dict's names which of upstream's models it is
//!   ([`NanoChatGeneration`]), and
//!   [`gpt_config`](NanoChatMetaMirror::gpt_config) is the config for both.
//! - **Names:** [`nanochat_gpt_rekey`].
//! - **Loading** (feature `store`): `load_nanochat_gpt_tensors`, and
//!   `load_nanochat_gpt_from_paths`, a meta and a state dict; with `cache`,
//!   `load_nanochat_gpt`, a resource set's.

use std::path::Path;

use serde::Deserialize;

use crate::{
    burner::module::Rekey,
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
        sys_at,
    },
    kits::language::nanochat::NanoChatGptContractConfig,
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

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

/// Which of upstream's models a checkpoint is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoChatGeneration {
    /// Upstream's model as of late 2025 (`karpathy/nanochat` at
    /// `a088b7a6`): the policy's defaults.
    Upstream2025,

    /// Upstream's 2026 model (`a3ca42a6`): the 2025 one with
    /// [`with_upstream_2026`](crate::kits::language::nanochat::NanoChatGptContractConfig::with_upstream_2026).
    Upstream2026,
}

/// The state-dict names only upstream's 2026 model has: one per 2026 part
/// that carries a parameter.
const UPSTREAM_2026_MARKERS: [&str; 5] = [
    "resid_lambdas",
    "x0_lambdas",
    "smear_gate.weight",
    "smear_lambda",
    "backout_lambda",
];

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

/// Upstream's state-dict names onto a
/// [`NanoChatGpt`](crate::kits::language::nanochat::NanoChatGpt)'s module
/// paths: the `transformer.` prefix dropped; onto its language model's (`lm.`),
/// the embedding `wte` its `lm.embedding`, the blocks `h.{i}` the stack's
/// `lm.stack.layers.{i}` (the feed-forward's `c_fc` / `c_proj` its
/// `linear1` / `linear2`, the value gate `ve_gate` its `value_gate`), and the
/// head `lm_head` its projection, `lm.head.proj`; everything else
/// (`attn.c_q`, the lambdas, `value_embeds.{i}`, `smear_gate`) bunsen names
/// as upstream does.
///
/// Unlike the wrappers' rekeys, it names the model's paths, not its
/// language model's alone: the 2026 additions sit beside the language model.
pub fn nanochat_gpt_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("lm_head.", "lm.head.proj.")
        .mount(
            "transformer.",
            "",
            Rekey::keep_rest()
                .rename("wte.", "lm.embedding.")
                .mount_indexed(
                    "h.",
                    "lm.stack.layers.",
                    Rekey::keep_rest()
                        .rename("mlp.c_fc.", "mlp.linear1.")
                        .rename("mlp.c_proj.", "mlp.linear2.")
                        .rename("attn.ve_gate.", "attn.value_gate."),
                ),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    fn meta(window: Option<&str>) -> NanoChatMetaMirror {
        NanoChatMetaMirror {
            step: Some(1),
            model_config: NanoChatModelConfigMirror {
                sequence_len: 2048,
                vocab_size: 65536,
                n_layer: 20,
                n_head: 10,
                n_kv_head: 10,
                n_embd: 1280,
                window_pattern: window.map(str::to_string),
            },
        }
    }

    /// The pinned checkpoints' metas parse, the extra keys some uploads add
    /// ignored.
    #[test]
    fn test_meta_from_json() {
        let m = NanoChatMetaMirror::from_json(
            r#"{"step": 650, "val_loss": 1.07, "model_config": {"sequence_len": 2048,
            "vocab_size": 65536, "n_layer": 20, "n_head": 10, "n_kv_head": 10, "n_embd": 1280,
            "bos_token": "<|bos|>", "chat_template": "..."}}"#,
        )
        .unwrap();
        assert_eq!(m.step, Some(650));
        assert_eq!(
            m,
            NanoChatMetaMirror {
                step: Some(650),
                ..meta(None)
            }
        );

        let err = NanoChatMetaMirror::from_json(r#"{"step": 1}"#).unwrap_err();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource).assert(&err);
    }

    /// The generation follows the state dict and the meta together, and a
    /// checkpoint between the two is refused.
    #[test]
    fn test_generation() {
        let v2025 = ["transformer.wte.weight", "lm_head.weight"];
        assert_eq!(
            meta(None).generation(&v2025).unwrap(),
            NanoChatGeneration::Upstream2025
        );
        let mut v2026: Vec<&str> = v2025.to_vec();
        v2026.extend(UPSTREAM_2026_MARKERS);
        v2026.push("value_embeds.1.weight");
        assert_eq!(
            meta(Some("L")).generation(&v2026).unwrap(),
            NanoChatGeneration::Upstream2026
        );

        let unsupported = ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("between upstream's 2025 and 2026 models");
        // The meta says 2026, the state dict 2025; and the other way round.
        unsupported.assert(&meta(Some("L")).generation(&v2025).unwrap_err());
        unsupported.assert(&meta(None).generation(&v2026).unwrap_err());
        // Some 2026 parts and not others: an intermediate upstream commit.
        let partial = ["transformer.wte.weight", "resid_lambdas", "x0_lambdas"];
        unsupported.assert(&meta(Some("L")).generation(&partial).unwrap_err());
    }

    /// The config takes the meta's geometry, the rotary table ten times its
    /// context, and the generation's options.
    #[test]
    fn test_config() {
        let c = meta(None).gpt_config(NanoChatGeneration::Upstream2025);
        assert_eq!(
            (
                c.init_seq_len,
                c.max_seq_len_factor,
                c.vocab_size,
                c.n_layer,
                c.n_embed
            ),
            (2048, 10, 65536, 20, 1280)
        );
        assert!(!c.residual_lambdas && c.window_pattern.is_none() && c.rope_base == 10000.0);

        let c = meta(Some("SSSL")).gpt_config(NanoChatGeneration::Upstream2026);
        assert_eq!(c.window_pattern.as_deref(), Some("SSSL"));
        assert!(c.residual_lambdas && c.value_embeds && c.backout && c.rope_base == 100000.0);
    }

    /// Upstream's names land where bunsen keeps each part.
    #[test]
    fn test_nanochat_gpt_rekey() {
        let r = nanochat_gpt_rekey();
        for (from, to) in [
            ("transformer.wte.weight", "lm.embedding.weight"),
            (
                "transformer.h.3.attn.c_q.weight",
                "lm.stack.layers.3.attn.c_q.weight",
            ),
            (
                "transformer.h.3.attn.c_proj.weight",
                "lm.stack.layers.3.attn.c_proj.weight",
            ),
            (
                "transformer.h.0.mlp.c_fc.weight",
                "lm.stack.layers.0.mlp.linear1.weight",
            ),
            (
                "transformer.h.0.mlp.c_proj.weight",
                "lm.stack.layers.0.mlp.linear2.weight",
            ),
            (
                "transformer.h.17.attn.ve_gate.weight",
                "lm.stack.layers.17.attn.value_gate.weight",
            ),
            ("lm_head.weight", "lm.head.proj.weight"),
            ("value_embeds.17.weight", "value_embeds.17.weight"),
            ("smear_gate.weight", "smear_gate.weight"),
            ("resid_lambdas", "resid_lambdas"),
        ] {
            assert_eq!(r.apply(from).as_deref(), Some(to), "{from}");
        }
    }
}
