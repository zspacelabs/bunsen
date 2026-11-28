//! The config conversion.

use std::path::Path;

use crate::{
    errors::BunsenResult,
    kits::{
        language::generation::qwen3::Qwen3Config,
        support::transformers::interop::{
            TransformersConfigJson,
            TransformersDecoderSection,
            config_from_path,
            config_from_str,
        },
    },
};

/// The [`Qwen3Config`] of a `transformers` `Qwen3ForCausalLM` `config.json`:
/// what an `hf:` repo's geometry is, any dense Qwen3's.
///
/// As [`qwen3_config`] over the whole file, which must say `model_type`
/// `qwen3`: a mixture of experts (`qwen3_moe`) is refused. The head is
/// untied when the file does not say, as `transformers`' `Qwen3Config`
/// defaults it.
///
/// # Errors
/// As [`config_from_str`] and [`qwen3_config`].
pub fn qwen3_config_from_str(json: &str) -> BunsenResult<Qwen3Config> {
    config_from_str(json, |t| qwen3_config(t, false))
}

/// [`qwen3_config_from_str`] of the `config.json` at `path`; every error
/// names the file.
///
/// # Errors
/// As [`config_from_path`] and [`qwen3_config`].
pub fn qwen3_config_from_path(path: &Path) -> BunsenResult<Qwen3Config> {
    config_from_path(path, |t| qwen3_config(t, false))
}

/// The [`Qwen3Config`] of a `transformers` Qwen3 text config: a whole
/// `Qwen3ForCausalLM` `config.json`, or the `text_config` section of a model
/// a Qwen3 is the text model of. `tie_default` is whether the head is tied
/// when the section does not say.
///
/// The sizes, the norm epsilon, the rotary base and whether the head is tied
/// come from the section. What bunsen's model fixes is checked: `model_type`
/// `qwen3`, `SiLU`, no attention bias, no sliding window, and the default
/// rotary embedding (no scaling).
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the section lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Qwen3 is not.
pub fn qwen3_config(
    t: &TransformersConfigJson,
    tie_default: bool,
) -> BunsenResult<Qwen3Config> {
    t.expect_str("model_type", "qwen3")?;
    let d = TransformersDecoderSection::read(t)?;
    t.expect_eq("hidden_act", d.hidden_act.as_str(), "silu")?;
    t.expect_eq("attention_bias", d.attention_bias, false)?;
    t.expect_eq("sliding_window", d.sliding_window, None)?;
    t.expect_eq("rope_parameters.rope_type", d.rope_type.as_str(), "default")?;
    Ok(Qwen3Config::new(
        t.usize("vocab_size")?,
        d.hidden_size,
        d.num_hidden_layers,
        d.num_attention_heads,
        d.num_key_value_heads,
        d.head_dim,
        d.intermediate_size,
    )
    .with_rms_eps(d.rms_norm_eps)
    .with_rope_theta(d.rope_theta)
    .with_tied_head(t.bool_or("tie_word_embeddings", tie_default)?))
}
