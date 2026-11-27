//! # Parakeet in `transformers`' names
//!
//! A `ParakeetForTDT` checkpoint as bunsen's
//! [`ParakeetTdt`](crate::kits::speech::asr::parakeet::ParakeetTdt), the
//! names every Parakeet checkpoint here is read under:
//!
//! - **Config:** [`parakeet_tdt_config`] of a parsed `config.json`, and
//!   [`parakeet_tdt_config_from_str`] / [`parakeet_tdt_config_from_path`].
//! - **Names:** [`parakeet_tdt_rekey`], over the encoder's ([`encoder_rekey`]:
//!   [`subsampling_rekey`], [`encoder_block_rekey`]).
//! - **Loading** (feature `store`): `load_parakeet_tdt_tensors`; with `cache`,
//!   `load_parakeet_tdt`, a resource set's checkpoint and `tokenizer.json`
//!   vocabulary.

use std::path::Path;

use crate::{
    blocks::{
        signals::{
            DwStridingFlattenOrder,
            DwStridingSubsamplingConfig,
            DwStridingSubsamplingMeta,
        },
        transformers::encoders::conformer::{
            ConformerBlockConfig,
            ConformerEncoderConfig,
        },
    },
    burner::module::Rekey,
    errors::BunsenResult,
    kits::{
        speech::asr::parakeet::ParakeetTdtConfig,
        support::interop::transformers::{
            TransformersConfigJson,
            config_from_path,
            config_from_str,
        },
    },
};

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

/// `transformers`' `ParakeetEncoderSubsampling` names onto a
/// [`DwStridingSubsampling`](crate::blocks::signals::DwStridingSubsampling)'s,
/// relative to it: the flat `layers.{k}` by
/// [`sequential_rekey`](DwStridingSubsamplingMeta::sequential_rekey),
/// everything else as it is, and the projection's weight permuted when
/// `striding` flattens otherwise than `transformers` does (channels outer).
pub fn subsampling_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    striding.reflatten_from(
        DwStridingFlattenOrder::ChannelsFreqs,
        striding.sequential_rekey("layers."),
    )
}

/// `transformers`' `ParakeetEncoderBlock` names onto a
/// [`ConformerBlock`](crate::blocks::transformers::encoders::conformer::ConformerBlock)'s,
/// relative to it: the attention's projections and positional biases, the
/// convolution module's depthwise convolution and norm, and the sublayers'
/// norms, renamed; everything else as it is.
pub fn encoder_block_rekey() -> Rekey {
    Rekey::keep_rest()
        .rename("self_attn.q_proj.", "attn.query.")
        .rename("self_attn.k_proj.", "attn.key.")
        .rename("self_attn.v_proj.", "attn.value.")
        .rename("self_attn.o_proj.", "attn.output.")
        .rename("self_attn.relative_k_proj.", "attn.pos.")
        .exact("self_attn.bias_u", "attn.pos_bias_u")
        .exact("self_attn.bias_v", "attn.pos_bias_v")
        .rename("conv.depthwise_conv.", "conv.depthwise.conv.")
        .rename("conv.norm.", "conv.depthwise.norm.")
        .rename("norm_feed_forward1.", "feed_forward1_norm.")
        .rename("norm_self_att.", "attn_norm.")
        .rename("norm_conv.", "conv_norm.")
        .rename("norm_feed_forward2.", "feed_forward2_norm.")
        .rename("norm_out.", "out_norm.")
        .rename("self_attn.", "attn.")
}

/// `transformers`' `ParakeetEncoder` names onto the
/// [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)
/// a [`ConformerEncoderConfig`] builds: [`subsampling_rekey`] and
/// [`encoder_block_rekey`] mounted where they sit, the batch-norm counters
/// dropped, and everything else as it is.
pub fn encoder_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .mount("subsampling.", "subsampling.", subsampling_rekey(striding))
        .mount_indexed("layers.", "transformer.layers.", encoder_block_rekey())
}

/// `transformers`' `ParakeetForTDT` state-dict names onto a
/// [`ParakeetTdt`](crate::kits::speech::asr::parakeet::ParakeetTdt)'s module
/// paths: the encoder by [`encoder_rekey`], the prediction network's
/// `nn.LSTM` by
/// [`lstm_stack_rekey`](crate::kits::support::interop::torch::lstm_stack_rekey),
/// the batch-norm counters dropped, and everything else as it is.
///
/// `striding` is the encoder's subsampling, a config or a built module: its
/// stage count and flatten order fix its map.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`Rekey::apply_to`] resolves to `gamma` / `beta`.
pub fn parakeet_tdt_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .mount("encoder.", "encoder.", encoder_rekey(striding))
        .mount(
            "decoder.lstm.",
            "decoder.lstm.",
            crate::kits::support::interop::torch::lstm_stack_rekey(),
        )
        .rename("decoder.decoder_projector.", "decoder.projector.")
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::assert_config_eq;

    #[test]
    fn test_parakeet_tdt_rekey() {
        for (hf, bunsen) in [
            (
                "encoder.subsampling.layers.0.weight",
                Some("encoder.subsampling.stem.conv.weight"),
            ),
            (
                "encoder.subsampling.layers.2.bias",
                Some("encoder.subsampling.stages.0.depthwise.conv.bias"),
            ),
            (
                "encoder.subsampling.layers.3.weight",
                Some("encoder.subsampling.stages.0.pointwise.conv.weight"),
            ),
            (
                "encoder.subsampling.layers.5.weight",
                Some("encoder.subsampling.stages.1.depthwise.conv.weight"),
            ),
            (
                "encoder.subsampling.layers.6.bias",
                Some("encoder.subsampling.stages.1.pointwise.conv.bias"),
            ),
            ("encoder.subsampling.layers.8.weight", None),
            (
                "encoder.subsampling.linear.weight",
                Some("encoder.subsampling.linear.weight"),
            ),
            (
                "encoder.layers.3.self_attn.q_proj.weight",
                Some("encoder.transformer.layers.3.attn.query.weight"),
            ),
            (
                "encoder.layers.3.self_attn.relative_k_proj.weight",
                Some("encoder.transformer.layers.3.attn.pos.weight"),
            ),
            (
                "encoder.layers.3.self_attn.bias_u",
                Some("encoder.transformer.layers.3.attn.pos_bias_u"),
            ),
            (
                "encoder.layers.3.conv.depthwise_conv.weight",
                Some("encoder.transformer.layers.3.conv.depthwise.conv.weight"),
            ),
            (
                "encoder.layers.3.conv.norm.running_var",
                Some("encoder.transformer.layers.3.conv.depthwise.norm.running_var"),
            ),
            ("encoder.layers.3.conv.norm.num_batches_tracked", None),
            (
                "encoder.layers.3.conv.pointwise_conv1.weight",
                Some("encoder.transformer.layers.3.conv.pointwise_conv1.weight"),
            ),
            (
                "encoder.layers.3.norm_out.weight",
                Some("encoder.transformer.layers.3.out_norm.weight"),
            ),
            (
                "decoder.lstm.weight_ih_l1",
                Some("decoder.lstm.1.features.weight"),
            ),
            (
                "decoder.lstm.bias_hh_l0",
                Some("decoder.lstm.0.hidden.bias"),
            ),
            (
                "decoder.decoder_projector.bias",
                Some("decoder.projector.bias"),
            ),
            ("joint.head.weight", Some("joint.head.weight")),
            ("encoder_projector.weight", Some("encoder_projector.weight")),
        ] {
            assert_eq!(
                parakeet_tdt_rekey(&DwStridingSubsamplingConfig::new(128, 256, 1024))
                    .apply(hf)
                    .as_deref(),
                bunsen,
                "{hf}"
            );
        }
    }

    /// `nvidia/parakeet-tdt-0.6b-v3`'s `config.json` is the built-in
    /// geometry; a CTC model's is refused by its `model_type`.
    #[test]
    fn test_parakeet_tdt_config() {
        const V3: &str = r#"{"architectures": ["ParakeetForTDT"], "blank_token_id": 8192,
            "decoder_hidden_size": 640, "durations": [0, 1, 2, 3, 4],
            "encoder_config": {"attention_bias": false, "conv_kernel_size": 9,
                "convolution_bias": false, "hidden_size": 1024, "intermediate_size": 4096,
                "model_type": "parakeet_encoder", "num_attention_heads": 8,
                "num_hidden_layers": 24, "num_mel_bins": 128, "scale_input": false,
                "subsampling_conv_channels": 256, "subsampling_conv_kernel_size": 3,
                "subsampling_conv_stride": 2, "subsampling_factor": 8},
            "model_type": "parakeet_tdt", "num_decoder_layers": 2, "vocab_size": 8193}"#;
        let config = parakeet_tdt_config_from_str(V3).unwrap();
        assert_config_eq("the config", &config, &ParakeetTdtConfig::tdt_0_6b_v3());

        let ctc = V3.replace("parakeet_tdt", "parakeet_ctc");
        crate::errors::testing::ErrorMatcher::unsupported()
            .message_contains("`model_type`")
            .assert_err(&parakeet_tdt_config_from_str(&ctc));
        let mels = V3.replace("\"num_mel_bins\": 128", "\"num_mel_bins\": 80");
        crate::errors::testing::ErrorMatcher::unsupported()
            .message_contains("`encoder_config.num_mel_bins` is 80")
            .assert_err(&parakeet_tdt_config_from_str(&mels));
    }
}
