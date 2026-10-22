//! Loading `transformers`-named tensors into a [`ParakeetTdt`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::blocks::{
    ParakeetTdt,
    ParakeetTdtConfig,
};
use crate::{
    blocks::{
        rnn::lstm::FusedLstm,
        signals::{
            DwStridingSubsamplingConfig,
            DwStridingSubsamplingMeta,
        },
        transformers::encoders::conformer::{
            ConformerBlockConfig,
            ConformerEncoderConfig,
        },
    },
    burner::module::Rekey,
    data::pretrained::{
        HfConfigJson,
        NamedTensorsApplied,
        apply_rekeyed_tensors,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// `transformers`' `ParakeetForTDT` state-dict names onto a
/// [`ParakeetTdt`]'s module paths: the encoder by
/// [`ConformerEncoderConfig::transformers_rekey`], the prediction network's
/// `nn.LSTM` by [`FusedLstm::torch_stack_rekey`], the batch-norm counters
/// dropped, and everything else as it is.
///
/// `striding` is the encoder's subsampling, a config or a built module: its
/// stage count and flatten order fix its map.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`apply_rekeyed_tensors`] resolves to `gamma` / `beta`.
pub fn hf_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::keep_rest()
        .drop_suffix(".num_batches_tracked")
        .mount(
            "encoder.",
            "encoder.",
            ConformerEncoderConfig::transformers_rekey(striding),
        )
        .mount(
            "decoder.lstm.",
            "decoder.lstm.",
            FusedLstm::torch_stack_rekey(),
        )
        .rename("decoder.decoder_projector.", "decoder.projector.")
}

/// Loads tensors named as `transformers`' `ParakeetForTDT` state dict into
/// `model`: [`apply_rekeyed_tensors`] under [`hf_rekey`], at `model`'s
/// subsampling.
///
/// # Errors
/// As [`apply_rekeyed_tensors`];
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// `model`'s encoder does not subsample by depthwise striding.
pub fn load_hf_tensors(
    model: &mut ParakeetTdt,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let striding = model.encoder.subsampling.dw_striding().ok_or_else(|| {
        BunsenError::invalid_resource("a Parakeet encoder subsamples by depthwise striding")
    })?;
    let rekey = hf_rekey(striding);
    apply_rekeyed_tensors(model, tensors, &rekey)
}

/// The [`ParakeetTdtConfig`] of a `transformers` `ParakeetForTDT`
/// `config.json`: what an `hf:` repo's geometry is.
///
/// The encoder's sizes, the prediction network's width and depth, the
/// vocabulary and the durations come from the file. What bunsen's model fixes
/// is checked: a TDT model (`model_type` `parakeet_tdt`), 128 mel bins (the
/// front end's), and depthwise-striding subsampling of kernel 3 and stride 2.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Parakeet is not (a CTC model,
/// another subsampling).
pub fn parakeet_config_from_hf(json: &str) -> BunsenResult<ParakeetTdtConfig> {
    let c = HfConfigJson::parse(json)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_rekey() {
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
                Some("encoder.transformer.layers.3.self_attn.query.weight"),
            ),
            (
                "encoder.layers.3.self_attn.relative_k_proj.weight",
                Some("encoder.transformer.layers.3.self_attn.pos.weight"),
            ),
            (
                "encoder.layers.3.self_attn.bias_u",
                Some("encoder.transformer.layers.3.self_attn.pos_bias_u"),
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
                Some("encoder.transformer.layers.3.norm_out.weight"),
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
                hf_rekey(&DwStridingSubsamplingConfig::new(128, 256, 1024))
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
    fn test_parakeet_config_from_hf() {
        const V3: &str = r#"{"architectures": ["ParakeetForTDT"], "blank_token_id": 8192,
            "decoder_hidden_size": 640, "durations": [0, 1, 2, 3, 4],
            "encoder_config": {"attention_bias": false, "conv_kernel_size": 9,
                "convolution_bias": false, "hidden_size": 1024, "intermediate_size": 4096,
                "model_type": "parakeet_encoder", "num_attention_heads": 8,
                "num_hidden_layers": 24, "num_mel_bins": 128, "scale_input": false,
                "subsampling_conv_channels": 256, "subsampling_conv_kernel_size": 3,
                "subsampling_conv_stride": 2, "subsampling_factor": 8},
            "model_type": "parakeet_tdt", "num_decoder_layers": 2, "vocab_size": 8193}"#;
        let config = parakeet_config_from_hf(V3).unwrap();
        assert_eq!(
            format!("{config:?}"),
            format!("{:?}", ParakeetTdtConfig::tdt_0_6b_v3())
        );

        let ctc = V3.replace("parakeet_tdt", "parakeet_ctc");
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("`model_type`")
            .assert_err(&parakeet_config_from_hf(&ctc));
        let mels = V3.replace("\"num_mel_bins\": 128", "\"num_mel_bins\": 80");
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Unsupported)
            .message_contains("`encoder_config.num_mel_bins` is 80")
            .assert_err(&parakeet_config_from_hf(&mels));
    }
}
