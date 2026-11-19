//! Loading the 12 Hz codec's `decoder.*` tensors into a
//! [`Qwen3TtsCodecDecoder`].
//!
//! `Qwen/Qwen3-TTS-Tokenizer-12Hz`'s `model.safetensors` (and each model
//! repository's `speech_tokenizer/model.safetensors`, the same file) holds
//! the encoder under `encoder.` and the decoder under `decoder.`, named as
//! `qwen_tts`' `Qwen3TTSTokenizerV2Decoder` names its modules, without
//! weight norm. The names map onto the modules' ([`qwen3_tts_codec_rekey`]):
//! the transformer's layers onto a decoder stack's, the vocoder's flat list
//! onto its input, blocks and output. The quantizer's input projections are
//! the encoder's side, and are not read.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    Qwen3TtsCodecDecoder,
    Qwen3TtsCodecDecoderContractConfig,
};
use crate::{
    blocks::transformers::decoders::causal::{
        CausalDecoderBlock,
        DecoderLayerKeys,
    },
    burner::module::Rekey,
    data::formats::NamedTensorsApplied,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The prefix the decoder's tensors sit under in the codec's file.
pub const QWEN3_TTS_CODEC_DECODER_PREFIX: &str = "decoder.";

/// A transformer layer's names onto a decoder block's canonical ones:
/// Llama's (`input_layernorm`, `post_attention_layernorm`, `self_attn`'s
/// `{q,k,v,o}_proj`, the `SwiGLU`'s projections as they are), and the layer
/// scales, `self_attn_layer_scale.scale` and `mlp_layer_scale.scale`.
pub fn qwen3_tts_codec_layer_rekey() -> Rekey {
    CausalDecoderBlock::layer_rekey(&DecoderLayerKeys::llama(
        "input_layernorm",
        "post_attention_layernorm",
    ))
    .exact("self_attn_layer_scale.scale", "attn_layer_scale")
    .exact("mlp_layer_scale.scale", "mlp_layer_scale")
}

/// The decoder's names, without [`QWEN3_TTS_CODEC_DECODER_PREFIX`], onto a
/// [`Qwen3TtsCodecDecoder`]'s canonical ones.
pub fn qwen3_tts_codec_rekey(config: &Qwen3TtsCodecDecoderContractConfig) -> Rekey {
    let codebooks = Rekey::keep_rest().mount_indexed(
        "vq.layers.",
        "codebooks.",
        Rekey::keep_rest().rename("_codebook.", ""),
    );
    let rekey = Rekey::drop_rest()
        .mount(
            "quantizer.rvq_first.",
            "quantizer.rvq_first.",
            codebooks.clone(),
        )
        .mount("quantizer.rvq_rest.", "quantizer.rvq_rest.", codebooks)
        .rename("pre_conv.", "pre_conv.")
        .rename("pre_transformer.input_proj.", "pre_transformer.input_proj.")
        .rename(
            "pre_transformer.output_proj.",
            "pre_transformer.output_proj.",
        )
        .rename("pre_transformer.norm.", "pre_transformer.stack.norm.")
        .mount_indexed(
            "pre_transformer.layers.",
            "pre_transformer.stack.layers.",
            qwen3_tts_codec_layer_rekey(),
        )
        .mount_indexed(
            "upsample.",
            "upsample.",
            Rekey::drop_rest()
                .rename("0.", "up.")
                .rename("1.dwconv.conv.", "block.dwconv.")
                .rename("1.", "block."),
        );
    let blocks = config.upsample_rates.len();
    let rekey = (0..blocks).fold(rekey.rename("decoder.0.", "decoder.input."), |rekey, i| {
        let units = (0..super::QWEN3_TTS_VOCODER_DILATIONS.len()).fold(
            Rekey::drop_rest()
                .rename("0.", "act.")
                .rename("1.", "upsample."),
            |units, u| units.rename(format!("{}.", u + 2), format!("residuals.{u}.")),
        );
        rekey.mount(
            format!("decoder.{}.block.", i + 1),
            format!("decoder.blocks.{i}."),
            units,
        )
    });
    rekey
        .rename(format!("decoder.{}.", blocks + 1), "decoder.output_act.")
        .rename(format!("decoder.{}.", blocks + 2), "decoder.output.")
}

/// Whether a decoder tensor (its name without the prefix) is the encoder's
/// side, which the decoder does not read: a quantizer's input projection.
fn is_encode_side(name: &str) -> bool {
    name.starts_with("quantizer.") && name.ends_with(".input_proj.weight")
}

/// Loads the codec file's tensors (`decoder.*`; the encoder's and any others
/// are dropped) into `model`, built from `config`: the names mapped
/// ([`qwen3_tts_codec_rekey`]), every floating tensor in `dtype`, or, when
/// `None`, in the dtype `model` was built in.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`Rekey::load`]; and naming any `decoder.*` tensor the model has no
/// parameter for.
pub fn load_qwen3_tts_codec_tensors(
    model: &mut Qwen3TtsCodecDecoder,
    config: &Qwen3TtsCodecDecoderContractConfig,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let prefix = QWEN3_TTS_CODEC_DECODER_PREFIX;
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| {
            t.name
                .strip_prefix(prefix)
                .is_some_and(|name| !is_encode_side(name))
        })
        .collect();
    let rekey = Rekey::drop_rest()
        .mount(prefix, "", qwen3_tts_codec_rekey(config))
        .then(Rekey::keep_rest().mount_canonical("", model));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "Qwen3-TTS codec decoder tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}
