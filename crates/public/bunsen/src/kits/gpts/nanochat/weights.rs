//! Loading upstream nanochat's own checkpoints into a [`NanoChatGpt`].
//!
//! Upstream's checkpoint manager writes a step as two files: the model's
//! state dict, `model_<step>.pt` (`torch.save`), and its `meta_<step>.json`,
//! whose `model_config` is the model's geometry. [`NanoChatMeta`] reads the
//! meta, [`nanochat_generation`] tells from the state dict's names which of
//! upstream's models it is, [`nanochat_config`] builds the config for both,
//! and [`load_pt_tensors`] loads the state dict through [`pt_rekey`].

use std::path::Path;

use burn::{
    store::{
        ModuleSnapshot,
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::{
        Device,
        TensorData,
    },
};
use serde::Deserialize;

use super::{
    NanoChatGpt,
    NanoChatGptContractConfig,
};
use crate::{
    burner::{
        module::{
            ModuleInit,
            Rekey,
        },
        tensor::param_float_dtype,
    },
    data::pretrained::{
        NamedTensorsApplied,
        apply_rekeyed_tensors_as,
        read_pytorch_tensors,
    },
    errors::{
        BunsenError,
        BunsenResult,
        ResultContext,
        sys_at,
    },
};

/// A nanochat checkpoint's `meta_<step>.json`: the parts bunsen reads. The
/// rest (the training run's settings, its losses, a chat template some
/// uploads add) is ignored.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct NanoChatMeta {
    /// The training step the checkpoint is of, when the meta says.
    #[serde(default)]
    pub step: Option<u64>,

    /// The model's geometry, as upstream's `GPTConfig`.
    pub model_config: NanoChatModelConfig,
}

/// Upstream's `GPTConfig`, as a checkpoint's meta writes it: upstream's
/// names.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct NanoChatModelConfig {
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

impl NanoChatMeta {
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
    pub fn from_file(path: &Path) -> BunsenResult<Self> {
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
    /// [`with_upstream_2026`](NanoChatGptContractConfig::with_upstream_2026).
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

/// Which of upstream's models the checkpoint whose state dict holds `names`
/// and whose meta is `meta` is: the 2026 one when the state dict has its
/// parameters and the meta its window pattern, the 2025 one when it has
/// neither.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) when it has
/// some of the 2026 parameters and not others, or when the state dict and
/// the meta disagree: a checkpoint of one of upstream's commits between the
/// two, which bunsen does not model.
pub fn nanochat_generation<S: AsRef<str>>(
    meta: &NanoChatMeta,
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
    let windowed = meta.model_config.window_pattern.is_some();
    match (present.len(), value_embeds, windowed) {
        (0, false, false) => Ok(NanoChatGeneration::Upstream2025),
        (n, true, true) if n == UPSTREAM_2026_MARKERS.len() => Ok(NanoChatGeneration::Upstream2026),
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

/// The config of a checkpoint of `generation` whose meta is `meta`: its
/// geometry, `sequence_len` the initial sequence length (the rotary table
/// ten times it, as upstream's), and the generation's options.
pub fn nanochat_config(
    meta: &NanoChatMeta,
    generation: NanoChatGeneration,
) -> NanoChatGptContractConfig {
    let m = &meta.model_config;
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

/// Upstream's state-dict names onto a [`NanoChatGpt`]'s module paths: the
/// `transformer.` prefix dropped, the feed-forward's `c_fc` / `c_proj` its
/// `linear1` / `linear2`, and the value gate `ve_gate` its `value_gate`;
/// everything else (`attn.c_q`, `lm_head`, the lambdas, `value_embeds.{i}`,
/// `smear_gate`) bunsen names as upstream does.
pub fn pt_rekey() -> Rekey {
    Rekey::keep_rest().mount(
        "transformer.",
        "",
        Rekey::keep_rest().mount_indexed(
            "h.",
            "h.",
            Rekey::keep_rest()
                .rename("mlp.c_fc.", "mlp.linear1.")
                .rename("mlp.c_proj.", "mlp.linear2.")
                .rename("attn.ve_gate.", "attn.value_gate."),
        ),
    )
}

/// Loads an upstream nanochat state dict (`tensors`, under upstream's names)
/// into `model` by [`pt_rekey`], every floating tensor cast to the dtype
/// `model` was built in: a bf16 model takes bf16 tensors as they are.
///
/// Upstream's norms carry no learned scale, and bunsen's do: each norm's
/// scale loads as ones, which is upstream's norm exactly, in any dtype.
///
/// # Errors
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) naming any
/// tensor the model has no parameter for; otherwise as
/// [`apply_rekeyed_tensors_as`].
pub fn load_pt_tensors(
    model: &mut NanoChatGpt,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let dtype = param_float_dtype(&model.wte().weight);
    let mut tensors: Vec<PackTensor> = tensors.into_iter().collect();
    let scales: Vec<PackTensor> = model
        .collect(None, None, true)
        .into_iter()
        .filter(|t| t.name.ends_with(".gamma") && t.name.contains("norm"))
        .map(|t| {
            let n: usize = t.shape.iter().product();
            bridge::from_data(
                TensorData::new(vec![1.0f32; n], t.shape.clone()),
                t.name.clone(),
                None,
            )
        })
        .collect();
    tensors.extend(scales);
    let applied = apply_rekeyed_tensors_as(model, tensors, &pt_rekey(), Some(dtype))?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::unsupported(format!(
            "a nanochat state dict with tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}

/// An upstream nanochat checkpoint, its `meta_<step>.json` at `meta` and its
/// `model_<step>.pt` at `model`, as a [`NanoChatGpt`] on `device`: its
/// config from the meta and the state dict's names
/// ([`nanochat_generation`], [`nanochat_config`]), built, and loaded
/// ([`load_pt_tensors`]).
///
/// # Errors
/// As [`NanoChatMeta::from_file`],
/// [`read_pytorch_tensors`], [`nanochat_generation`] and
/// [`load_pt_tensors`]; each names the file it is about.
pub fn load_pt_checkpoint(
    meta: &Path,
    model: &Path,
    device: &Device,
) -> BunsenResult<(NanoChatGpt, NanoChatGeneration)> {
    let meta_data = NanoChatMeta::from_file(meta)?;
    let tensors = read_pytorch_tensors(model)?;
    let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    let generation = nanochat_generation(&meta_data, &names)
        .with_context(|| format!("{} and {}", meta.display(), model.display()))?;
    let mut gpt: NanoChatGpt = nanochat_config(&meta_data, generation).try_init(device)?;
    load_pt_tensors(&mut gpt, tensors).with_context(|| model.display().to_string())?;
    Ok((gpt, generation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    fn meta(window: Option<&str>) -> NanoChatMeta {
        NanoChatMeta {
            step: Some(1),
            model_config: NanoChatModelConfig {
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
        let m = NanoChatMeta::from_json(
            r#"{"step": 650, "val_loss": 1.07, "model_config": {"sequence_len": 2048,
            "vocab_size": 65536, "n_layer": 20, "n_head": 10, "n_kv_head": 10, "n_embd": 1280,
            "bos_token": "<|bos|>", "chat_template": "..."}}"#,
        )
        .unwrap();
        assert_eq!(m.step, Some(650));
        assert_eq!(
            m,
            NanoChatMeta {
                step: Some(650),
                ..meta(None)
            }
        );

        let err = NanoChatMeta::from_json(r#"{"step": 1}"#).unwrap_err();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource).assert(&err);
    }

    /// The generation follows the state dict and the meta together, and a
    /// checkpoint between the two is refused.
    #[test]
    fn test_generation() {
        let v2025 = ["transformer.wte.weight", "lm_head.weight"];
        assert_eq!(
            nanochat_generation(&meta(None), &v2025).unwrap(),
            NanoChatGeneration::Upstream2025
        );
        let mut v2026: Vec<&str> = v2025.to_vec();
        v2026.extend(UPSTREAM_2026_MARKERS);
        v2026.push("value_embeds.1.weight");
        assert_eq!(
            nanochat_generation(&meta(Some("L")), &v2026).unwrap(),
            NanoChatGeneration::Upstream2026
        );

        let unsupported = ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("between upstream's 2025 and 2026 models");
        // The meta says 2026, the state dict 2025; and the other way round.
        unsupported.assert(&nanochat_generation(&meta(Some("L")), &v2025).unwrap_err());
        unsupported.assert(&nanochat_generation(&meta(None), &v2026).unwrap_err());
        // Some 2026 parts and not others: an intermediate upstream commit.
        let partial = ["transformer.wte.weight", "resid_lambdas", "x0_lambdas"];
        unsupported.assert(&nanochat_generation(&meta(Some("L")), &partial).unwrap_err());
    }

    /// The config takes the meta's geometry, the rotary table ten times its
    /// context, and the generation's options.
    #[test]
    fn test_config() {
        let c = nanochat_config(&meta(None), NanoChatGeneration::Upstream2025);
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
        assert!(!c.residual_lambdas && c.window_pattern.is_none() && c.rope_base == 10000);

        let c = nanochat_config(&meta(Some("SSSL")), NanoChatGeneration::Upstream2026);
        assert_eq!(c.window_pattern.as_deref(), Some("SSSL"));
        assert!(c.residual_lambdas && c.value_embeds && c.backout && c.rope_base == 100000);
    }

    /// Upstream's names land where bunsen keeps each part.
    #[test]
    fn test_pt_rekey() {
        let r = pt_rekey();
        for (from, to) in [
            ("transformer.wte.weight", "wte.weight"),
            ("transformer.h.3.attn.c_q.weight", "h.3.attn.c_q.weight"),
            (
                "transformer.h.3.attn.c_proj.weight",
                "h.3.attn.c_proj.weight",
            ),
            ("transformer.h.0.mlp.c_fc.weight", "h.0.mlp.linear1.weight"),
            (
                "transformer.h.0.mlp.c_proj.weight",
                "h.0.mlp.linear2.weight",
            ),
            (
                "transformer.h.17.attn.ve_gate.weight",
                "h.17.attn.value_gate.weight",
            ),
            ("lm_head.weight", "lm_head.weight"),
            ("value_embeds.17.weight", "value_embeds.17.weight"),
            ("smear_gate.weight", "smear_gate.weight"),
            ("resid_lambdas", "resid_lambdas"),
        ] {
            assert_eq!(r.apply(from).as_deref(), Some(to), "{from}");
        }
    }
}
