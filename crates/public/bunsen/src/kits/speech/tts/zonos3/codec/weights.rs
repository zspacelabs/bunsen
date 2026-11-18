//! Loading `Zyphra/ZONOS3-Mini-hf`'s `codec.*` tensors into a
//! [`Zonos3Codec`].
//!
//! The file names the decoder as the `-hf` port's `Zonos3MiniCodec` does,
//! under `codec.`, weight norm already fused. Each trunk layer's packed
//! `qkv` is split by rows first ([`zonos3_codec_row_splits`]); the names
//! then map onto the modules' ([`zonos3_codec_rekey`]): the stage list's
//! odd entries are the parameterless patch-ups, so stage `2 i` is
//! [`Zonos3Codec::stages`]`[i]`.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    store::burn_pack::Tensor as PackTensor,
    tensor::FloatDType,
};

use super::{
    Zonos3Codec,
    Zonos3CodecConfig,
};
use crate::{
    burner::module::{
        Rekey,
        TensorLoadRowSplit,
    },
    data::formats::{
        NamedTensorsApplied,
        split_rows,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The prefix the codec decoder's tensors sit under in the `-hf` file.
pub const ZONOS3_CODEC_PREFIX: &str = "codec.";

/// Each trunk layer's packed `qkv`, `[3 W, W]`, split by rows into `q`,
/// `k` and `v`.
pub fn zonos3_codec_row_splits() -> [TensorLoadRowSplit; 1] {
    [TensorLoadRowSplit::equal(
        ".qkv.weight",
        [".q.weight", ".k.weight", ".v.weight"],
    )]
}

/// A trunk layer's names, after [`zonos3_codec_row_splits`], onto a decoder
/// block's canonical ones: `norm1` and `norm2` its two norms, `q`, `k`, `v`
/// and `attention_out` its attention's projections, `ffn.0` and `ffn.3` its
/// feed-forward's two linears, `attention_scale` and `ffn_scale` its layer
/// scales.
pub fn zonos3_codec_layer_rekey() -> Rekey {
    Rekey::drop_rest()
        .rename("norm1.", "attn_norm.")
        .rename("norm2.", "mlp_norm.")
        .rename("q.", "attn.c_q.")
        .rename("k.", "attn.c_k.")
        .rename("v.", "attn.c_v.")
        .rename("attention_out.", "attn.c_proj.")
        .rename("ffn.0.", "mlp.linear1.")
        .rename("ffn.3.", "mlp.linear2.")
        .exact("attention_scale", "attn_layer_scale")
        .exact("ffn_scale", "mlp_layer_scale")
}

/// The decoder's names, after [`zonos3_codec_row_splits`] and without
/// [`ZONOS3_CODEC_PREFIX`], onto a [`Zonos3Codec`]'s canonical ones.
pub fn zonos3_codec_rekey(config: &Zonos3CodecConfig) -> Rekey {
    let semantic = config.semantic_depth;
    let rekey = Rekey::drop_rest()
        .rename("quantizers.", "quantizers.")
        .rename(format!("semantic_head.{semantic}."), "semantic_proj.")
        .rename("semantic_head.", "semantic_blocks.")
        .rename("head.blocks.", "head_blocks.")
        .rename("head.norm.", "head_norm.")
        .rename("head.outs.", "head_outs.")
        .rename("head.mixer.", "mixer.")
        .exact("to_external_kernel", "resampler.weight");
    (0..config.stages.len()).fold(rekey, |rekey, i| {
        let from = format!("stages.{}.", 2 * i);
        let to = format!("stages.{i}.");
        rekey
            .rename(format!("{from}input_proj."), format!("{to}input_proj."))
            .rename(format!("{from}output_proj."), format!("{to}output_proj."))
            .mount_indexed(
                format!("{from}layers."),
                format!("{to}stack.layers."),
                zonos3_codec_layer_rekey(),
            )
    })
}

/// Loads the `-hf` file's tensors (`codec.*`; any others are dropped) into
/// `model`, built from `config`: the packed projections split
/// ([`zonos3_codec_row_splits`]), the names mapped
/// ([`zonos3_codec_rekey`]), every floating tensor in `dtype`, or, when
/// `None`, in the dtype `model` was built in.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) as
/// [`split_rows`] and [`Rekey::load`]; and naming any `codec.*` tensor the
/// model has no parameter for.
pub fn load_zonos3_codec_tensors(
    model: &mut Zonos3Codec,
    config: &Zonos3CodecConfig,
    tensors: impl IntoIterator<Item = PackTensor>,
    dtype: Option<FloatDType>,
) -> BunsenResult<NamedTensorsApplied> {
    let tensors: Vec<PackTensor> = tensors
        .into_iter()
        .filter(|t| t.name.starts_with(ZONOS3_CODEC_PREFIX))
        .collect();
    let tensors = split_rows(tensors, &zonos3_codec_row_splits())?;
    let rekey = Rekey::drop_rest()
        .mount(ZONOS3_CODEC_PREFIX, "", zonos3_codec_rekey(config))
        .then(Rekey::keep_rest().mount_canonical("", model));
    let applied = rekey.load(model, tensors, dtype)?;
    if !applied.unused.is_empty() {
        return Err(BunsenError::invalid_resource(format!(
            "ZONOS3 codec tensors the model has no parameter for: {}",
            applied.unused.join(", ")
        )));
    }
    Ok(applied)
}
