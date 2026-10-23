//! Loading `transformers`- and `NeMo`-named tensors into a [`Sortformer`].
use burn::store::burn_pack::Tensor as PackTensor;

use super::blocks::{
    NemotronDiarizationConfig,
    Sortformer,
};
use crate::{
    blocks::signals::{
        DwStridingSubsamplingMeta,
        FeatureStackingMeta,
        StackOrder,
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
    kits::speech::nemo::{
        conformer_encoder_rekey,
        transformer_encoder_rekey,
    },
};

/// `transformers`' `Nemotron3DiarizationEmbedder` names onto a
/// [`FeatureStacking`](crate::blocks::signals::FeatureStacking)'s, relative
/// to it: `projection.*`, its weight permuted when `stacking` stacks
/// otherwise than `transformers` does (frames outer). Anything else is
/// dropped.
pub fn embedder_rekey(stacking: &impl FeatureStackingMeta) -> Rekey {
    stacking.restack_from(
        StackOrder::FramesFeatures,
        Rekey::drop_rest().rename("projection.", "projection."),
    )
}

/// `transformers`' `Nemotron3DiarizationEncoderLayer` names onto a
/// [`RopeEncoderBlock`](crate::blocks::transformers::encoders::rope::RopeEncoderBlock)'s,
/// relative to it: `layer_norm{1,2}` its norms, `self_attn`'s
/// `{q,k,v,o}_proj` the attention, `mlp.fc{1,2}` the MLP. Anything else is
/// dropped.
pub fn encoder_layer_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("layer_norm1.", "attn_norm.")
        .rename("layer_norm2.", "mlp_norm.")
        .rename("self_attn.q_proj.", "attn.query.")
        .rename("self_attn.k_proj.", "attn.key.")
        .rename("self_attn.v_proj.", "attn.value.")
        .rename("self_attn.o_proj.", "attn.output.")
        .rename("mlp.fc1.", "mlp.linear1.")
        .rename("mlp.fc2.", "mlp.linear2.")
}

/// `transformers`' `Nemotron3DiarizationEncoder` names onto a
/// [`RopeTransformerEncoder`](crate::blocks::transformers::encoders::rope::RopeTransformerEncoder)'s,
/// relative to it: `input_layer_norm` the embedding norm, `layer_norm` the
/// final one, each `layers.{n}` by [`encoder_layer_rekey`]. Anything else
/// (the rotary table's `inv_freq`, a buffer) is dropped.
pub fn encoder_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("input_layer_norm.", "embed_norm.")
        .rename("layer_norm.", "final_norm.")
        .mount_indexed("layers.", "layers.", encoder_layer_rekey())
}

/// `transformers`' `Nemotron3DiarizationForAudioFrameClassification`
/// state-dict names onto a [`Sortformer`]'s module paths: the audio tower's
/// embedder by [`embedder_rekey`], its encoder by [`encoder_rekey`], the
/// projection, upsample and head by prefix. Anything else is dropped.
///
/// `stacking` is the encoder's front end, a config or a built module: its
/// stack order fixes its map.
///
/// Normalization parameters keep their `weight` / `bias` spellings, which
/// [`apply_rekeyed_tensors`] resolves to `gamma` / `beta`.
pub fn nemotron_hf_rekey(stacking: &impl FeatureStackingMeta) -> Rekey {
    Rekey::drop_rest()
        .exact("silence_embeds", "silence")
        .mount(
            "model.audio_tower.embedder.",
            "encoder.subsampling.",
            embedder_rekey(stacking),
        )
        .mount(
            "model.audio_tower.",
            "encoder.transformer.",
            encoder_rekey(),
        )
        .rename("model.proj.", "projection.")
        .rename("model.upsampler.conv.", "upsample.conv.")
        .rename("classifier.dense.", "head.dense.")
        .rename("classifier.out_proj.", "head.out.")
}

/// Loads tensors named as `transformers`'
/// `Nemotron3DiarizationForAudioFrameClassification` state dict into
/// `model`: [`apply_rekeyed_tensors`] under [`nemotron_hf_rekey`], at
/// `model`'s front end.
///
/// # Errors
/// As [`apply_rekeyed_tensors`];
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// `model`'s encoder does not subsample by feature stacking.
pub fn load_nemotron_hf_tensors(
    model: &mut Sortformer,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let stacking = model.encoder.subsampling.stacking().ok_or_else(|| {
        BunsenError::invalid_resource("a Nemotron encoder subsamples by feature stacking")
    })?;
    let rekey = nemotron_hf_rekey(stacking);
    apply_rekeyed_tensors(model, tensors, &rekey)
}

/// `NeMo` `SortformerEncLabelModel` state-dict names, in Sortformer v2's
/// layout, onto a [`Sortformer`]'s module paths: the `FastConformer` by
/// [`conformer_encoder_rekey`], the `transformer_encoder` by
/// [`transformer_encoder_rekey`], the projection and head by
/// prefix. Anything else is dropped: `num_batches_tracked`, the
/// featurizer's tables, and the unused `hidden_to_spks`.
///
/// `striding` is the encoder's subsampling, a config or a built module: its
/// stage count and flatten order fix its map.
pub fn sortformer_v2_nemo_rekey(striding: &impl DwStridingSubsamplingMeta) -> Rekey {
    Rekey::drop_rest()
        .mount("encoder.", "encoder.", conformer_encoder_rekey(striding))
        .mount(
            "transformer_encoder.",
            "transformer.",
            transformer_encoder_rekey(),
        )
        .rename("sortformer_modules.encoder_proj.", "projection.")
        .rename("sortformer_modules.first_hidden_to_hidden.", "head.dense.")
        .rename("sortformer_modules.single_hidden_to_spks.", "head.out.")
}

/// Loads tensors named as a `NeMo` Sortformer v2 state dict (a `.nemo`'s
/// `model_weights.ckpt`) into `model`: [`apply_rekeyed_tensors`] under
/// [`sortformer_v2_nemo_rekey`], at `model`'s subsampling.
///
/// # Errors
/// As [`apply_rekeyed_tensors`];
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// `model`'s encoder does not subsample by depthwise striding.
pub fn load_sortformer_v2_nemo_tensors(
    model: &mut Sortformer,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let striding = model.encoder.subsampling.dw_striding().ok_or_else(|| {
        BunsenError::invalid_resource("a Sortformer v2 encoder subsamples by depthwise striding")
    })?;
    let rekey = sortformer_v2_nemo_rekey(striding);
    apply_rekeyed_tensors(model, tensors, &rekey)
}

/// The [`NemotronDiarizationConfig`] of a `transformers`
/// `Nemotron3DiarizationForAudioFrameClassification` `config.json`: what an
/// `hf:` repo's geometry is.
///
/// The encoder's and the head's sizes come from the file. What bunsen's model
/// fixes is checked: the model type, a GELU feed-forward of a whole multiple
/// of the width, rotary embeddings over the whole head, an integer base, and
/// the head reading the encoder's width at the encoder's subsampling.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the field, when the file lacks one or does not parse;
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// field and its value, for a model bunsen's Sortformer is not.
pub fn nemotron_config_from_hf(json: &str) -> BunsenResult<NemotronDiarizationConfig> {
    let c = HfConfigJson::parse(json)?;
    c.expect_str("model_type", "nemotron3_diarization")?;
    let a = c.section("audio_config")?;
    a.expect_str("hidden_act", "gelu")?;
    let d_model = a.usize("hidden_size")?;
    let intermediate = a.usize("intermediate_size")?;
    if d_model == 0 || !intermediate.is_multiple_of(d_model) {
        return Err(BunsenError::unsupported(format!(
            "config.json: `audio_config.intermediate_size` {intermediate} is not a whole \
             multiple of `hidden_size` {d_model}"
        )));
    }
    let rope = a.section("rope_parameters")?;
    a.expect_eq(
        "partial_rotary_factor",
        a.f64("partial_rotary_factor")?,
        1.0,
    )?;
    let theta = rope.f64("rope_theta")?;
    if theta.fract() != 0.0 || theta < 1.0 {
        return Err(BunsenError::unsupported(format!(
            "config.json: `audio_config.rope_parameters.rope_theta` {theta} is not a whole number"
        )));
    }
    let factor = a.usize("subsampling_factor")?;

    let h = c.section("head_config")?;
    h.expect_eq("audio_hidden_size", h.usize("audio_hidden_size")?, d_model)?;
    h.expect_eq("subsampling_factor", h.usize("subsampling_factor")?, factor)?;

    Ok(NemotronDiarizationConfig::new()
        .with_n_mels(a.usize("num_mel_bins")?)
        .with_d_model(d_model)
        .with_n_layers(a.usize("num_hidden_layers")?)
        .with_n_heads(a.usize("num_attention_heads")?)
        .with_ff_expansion(intermediate / d_model)
        .with_subsampling_factor(factor)
        .with_rope_base(theta as usize)
        .with_d_hidden(h.usize("hidden_size")?)
        .with_n_speakers(h.usize("num_speakers")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paths() {
        for (hf, bunsen) in [
            ("silence_embeds", "silence"),
            (
                "model.audio_tower.embedder.projection.weight",
                "encoder.subsampling.projection.weight",
            ),
            (
                "model.audio_tower.input_layer_norm.bias",
                "encoder.transformer.embed_norm.bias",
            ),
            (
                "model.audio_tower.layer_norm.weight",
                "encoder.transformer.final_norm.weight",
            ),
            (
                "model.audio_tower.layers.30.self_attn.o_proj.bias",
                "encoder.transformer.layers.30.attn.output.bias",
            ),
            (
                "model.audio_tower.layers.0.mlp.fc2.weight",
                "encoder.transformer.layers.0.mlp.linear2.weight",
            ),
            ("model.upsampler.conv.weight", "upsample.conv.weight"),
            ("classifier.out_proj.bias", "head.out.bias"),
        ] {
            assert_eq!(
                nemotron_hf_rekey(&crate::blocks::signals::FeatureStackingConfig::new(
                    128, 512
                ))
                .apply(hf)
                .as_deref(),
                Some(bunsen),
                "{hf}"
            );
        }
        assert_eq!(
            nemotron_hf_rekey(&crate::blocks::signals::FeatureStackingConfig::new(
                128, 512
            ))
            .apply("model.audio_tower.rotary_emb.inv_freq"),
            None
        );
    }

    #[test]
    fn test_v2_paths() {
        for (nemo, bunsen) in [
            (
                "encoder.pre_encode.conv.0.weight",
                Some("encoder.subsampling.stem.conv.weight"),
            ),
            (
                "encoder.layers.16.self_attn.linear_q.bias",
                Some("encoder.transformer.layers.16.attn.query.bias"),
            ),
            ("encoder.layers.0.conv.batch_norm.num_batches_tracked", None),
            (
                "transformer_encoder.layers.17.first_sub_layer.out_projection.weight",
                Some("transformer.layers.17.attn.output.weight"),
            ),
            (
                "sortformer_modules.encoder_proj.weight",
                Some("projection.weight"),
            ),
            (
                "sortformer_modules.first_hidden_to_hidden.bias",
                Some("head.dense.bias"),
            ),
            (
                "sortformer_modules.single_hidden_to_spks.weight",
                Some("head.out.weight"),
            ),
            ("sortformer_modules.hidden_to_spks.weight", None),
            ("preprocessor.featurizer.fb", None),
        ] {
            assert_eq!(
                sortformer_v2_nemo_rekey(
                    &crate::blocks::signals::DwStridingSubsamplingConfig::new(128, 256, 512)
                )
                .apply(nemo)
                .as_deref(),
                bunsen,
                "{nemo}"
            );
        }
    }

    /// `Nemotron-3-Diarization`'s own `config.json` is its geometry.
    #[test]
    fn test_nemotron_config_from_hf() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/sortformer/nemotron_config.json"
        ));
        let config = nemotron_config_from_hf(json).unwrap();
        let want = NemotronDiarizationConfig::nemotron_3_diarization();
        assert_eq!(config.n_mels, want.n_mels);
        assert_eq!(config.d_model, want.d_model);
        assert_eq!(config.n_layers, want.n_layers);
        assert_eq!(config.n_heads, want.n_heads);
        assert_eq!(config.ff_expansion, want.ff_expansion);
        assert_eq!(config.subsampling_factor, want.subsampling_factor);
        assert_eq!(config.rope_base, want.rope_base);
        assert_eq!(config.d_hidden, want.d_hidden);
        assert_eq!(config.n_speakers, want.n_speakers);

        let other = json.replace("\"nemotron3_diarization\"", "\"something_else\"");
        assert!(nemotron_config_from_hf(&other).is_err());
        let relu = json.replace("\"gelu\"", "\"relu\"");
        assert!(nemotron_config_from_hf(&relu).is_err());
    }
}
