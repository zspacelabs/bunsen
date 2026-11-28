//! The config conversion.

use std::path::Path;

use crate::{
    blocks::{
        signals::DwStridingSubsamplingConfig,
        transformers::encoders::conformer::{
            ConformerBlockConfig,
            ConformerEncoderConfig,
        },
    },
    errors::BunsenResult,
    kits::{
        speech::asr::parakeet::ParakeetTdtConfig,
        support::transformers::interop::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

/// The [`ParakeetTdtConfig`] of a parsed `transformers` `ParakeetForTDT`
/// `config.json`: what an `hf:` repo's geometry is.
///
/// The encoder's sizes, the prediction network's width and depth, the
/// vocabulary and the durations come from the file. What bunsen's model fixes
/// is checked: a TDT model (`model_type` `parakeet_tdt`), 128 mel bins (the
/// front end's), and depthwise-striding subsampling of kernel 3 and stride 2.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Parakeet is not (a CTC model,
/// another subsampling).
pub fn parakeet_tdt_config(c: &TransformersConfigJson) -> BunsenResult<ParakeetTdtConfig> {
    c.expect_str("model_type", "parakeet_tdt")?;
    let e = c.section("encoder_config")?;
    e.expect_eq("num_mel_bins", e.usize("num_mel_bins")?, 128)?;
    e.expect_eq(
        "subsampling_conv_kernel_size",
        e.usize("subsampling_conv_kernel_size")?,
        3,
    )?;
    e.expect_eq(
        "subsampling_conv_stride",
        e.usize("subsampling_conv_stride")?,
        2,
    )?;
    let factor = e.usize("subsampling_factor")?;
    if !factor.is_power_of_two() || factor < 2 {
        return Err(crate::errors::BunsenError::unsupported(format!(
            "config.json: `encoder_config.subsampling_factor` is {factor}, not a power of two"
        )));
    }
    let d_model = e.usize("hidden_size")?;
    let encoder = ConformerEncoderConfig::new(
        DwStridingSubsamplingConfig::new(128, e.usize("subsampling_conv_channels")?, d_model)
            .with_n_stages(factor.trailing_zeros() as usize),
        ConformerBlockConfig::new(
            d_model,
            e.usize("num_attention_heads")?,
            e.usize("intermediate_size")?,
            e.usize("conv_kernel_size")?,
        )
        .with_bias(e.bool_or("attention_bias", true)?)
        .with_conv_bias(e.bool_or("convolution_bias", true)?),
        e.usize("num_hidden_layers")?,
    )
    .with_scale_input(e.bool_or("scale_input", false)?);
    Ok(ParakeetTdtConfig::new(
        encoder,
        c.usize("vocab_size")?,
        c.usize("decoder_hidden_size")?,
    )
    .with_n_pred_layers(c.usize("num_decoder_layers")?)
    .with_durations(c.usizes("durations")?))
}

/// [`parakeet_tdt_config`] of a `config.json`'s text.
///
/// # Errors
/// As [`config_from_str`] and [`parakeet_tdt_config`].
pub fn parakeet_tdt_config_from_str(json: &str) -> BunsenResult<ParakeetTdtConfig> {
    config_from_str(json, parakeet_tdt_config)
}

/// [`parakeet_tdt_config`] of the `config.json` at `path`; every error names
/// the file.
///
/// # Errors
/// As [`config_from_path`] and [`parakeet_tdt_config`].
pub fn parakeet_tdt_config_from_path(path: &Path) -> BunsenResult<ParakeetTdtConfig> {
    config_from_path(path, parakeet_tdt_config)
}
