//! The codec decoder's `config.json` section and `-hf` names.

use alloc::{
    format,
    vec::Vec,
};

use serde::Deserialize;

use crate::{
    burner::module::{
        Rekey,
        TensorLoadRowSplit,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::tts::zonos3::codec::{
        Zonos3CodecConfig,
        Zonos3CodecStageConfig,
        Zonos3CodecStageMeta,
    },
};

/// One trunk stage of `config.json`'s `codec` section, under upstream's
/// names.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Zonos3CodecStageMirror {
    /// The causal sliding window, in frames, the frame's own included.
    pub context_frames: usize,
    /// The stage's width.
    pub d_model: usize,
    /// Its feed-forward's hidden width.
    pub dim_feedforward: usize,
    /// The width it reads.
    pub input_dimension: usize,
    /// Its attention heads.
    pub num_heads: usize,
    /// Its layers.
    pub num_layers: usize,
    /// The width it writes; `None`, its own (no output projection).
    pub output_dimension: Option<usize>,
}

/// The `codec` section of `Zyphra/ZONOS3-Mini-hf`'s `config.json` (the
/// `keep2_plain` decoder), under upstream's names.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Zonos3CodecMirror {
    /// Each codebook vector's width.
    pub codebook_dim: usize,
    /// Codes per codebook.
    pub codebook_size: usize,
    /// The branch mixer's taps.
    pub fir_kernel: usize,
    /// The iSTFT head's `ConvNeXt` blocks.
    pub head_depth: usize,
    /// Their hidden width.
    pub head_hidden: usize,
    /// The iSTFT hop, at the internal rate.
    pub head_hop: usize,
    /// Output samples per frame of codes.
    pub hop: usize,
    /// The latent's width.
    pub latent_dim: usize,
    /// The iSTFT branches' FFT sizes.
    pub n_ffts: Vec<usize>,
    /// The resampler's output phases.
    pub resample_new: usize,
    /// Its input stride.
    pub resample_orig: usize,
    /// Its kernel's half-width beyond the stride.
    pub resample_width: usize,
    /// The codebooks after the first.
    pub residual_codebooks: usize,
    /// The semantic head's `ConvNeXt` blocks.
    pub semantic_depth: usize,
    /// Their hidden width.
    pub semantic_hidden: usize,
    /// The first codebook's projection width, the semantic head's.
    pub semantic_width: usize,
    /// The trunk's stages.
    pub stages: Vec<Zonos3CodecStageMirror>,
}

impl Zonos3CodecMirror {
    /// The [`Zonos3CodecConfig`] of the section.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// stage whose feed-forward is not a whole multiple of its width, whose
    /// width is not a whole number of heads, or whose input is not its
    /// predecessor's output halved by the patch-up; or an even mixer.
    pub fn codec_config(&self) -> BunsenResult<Zonos3CodecConfig> {
        let json = self;
        let mut stages = Vec::with_capacity(json.stages.len());
        let mut reads = json.latent_dim;
        for (i, s) in json.stages.iter().enumerate() {
            if !s.dim_feedforward.is_multiple_of(s.d_model)
                || !s.d_model.is_multiple_of(s.num_heads)
            {
                return Err(BunsenError::unsupported(format!(
                    "codec stage {i}: d_model {}, {} heads, dim_feedforward {}",
                    s.d_model, s.num_heads, s.dim_feedforward
                )));
            }
            if s.input_dimension != reads {
                return Err(BunsenError::unsupported(format!(
                    "codec stage {i} reads {}, after {reads}",
                    s.input_dimension
                )));
            }
            let stage = Zonos3CodecStageConfig::new(
                s.input_dimension,
                s.d_model,
                s.num_layers,
                s.num_heads,
                s.dim_feedforward / s.d_model,
                s.context_frames,
            )
            .with_output_width(s.output_dimension);
            reads = stage.out_width() / 2;
            stages.push(stage);
        }
        if json.fir_kernel.is_multiple_of(2) {
            return Err(BunsenError::unsupported(format!(
                "an even mixer of {} taps",
                json.fir_kernel
            )));
        }
        Ok(Zonos3CodecConfig::new(
            json.codebook_size,
            json.codebook_dim,
            json.residual_codebooks,
            json.semantic_width,
            json.semantic_depth,
            json.semantic_hidden,
            json.latent_dim,
            stages,
            json.head_depth,
            json.head_hidden,
            json.n_ffts.clone(),
            json.head_hop,
            json.fir_kernel,
            json.resample_orig,
            json.resample_new,
            json.resample_width,
            json.hop,
        ))
    }
}

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
/// [`ZONOS3_CODEC_PREFIX`], onto a
/// [`Zonos3Codec`](crate::kits::speech::tts::zonos3::codec::Zonos3Codec)'s
/// canonical ones.
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
