//! The speaker encoder's `config.json` section and tensor names.

use alloc::vec::Vec;

use serde::Deserialize;

use crate::{
    burner::module::{
        Indices,
        Rekey,
        TensorLoadTransform,
    },
    errors::BunsenResult,
    kits::speech::tts::zonos3::blocks::{
        Zonos3MectBlockSpec,
        Zonos3SpeakerEncoderConfig,
        Zonos3SpeakerEncoderMeta,
        Zonos3SpeakerFrontEndConfig,
    },
};

/// The `speaker_encoder` section of `config.json`, under upstream's names:
/// the MECT-B2 network and its front end.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Zonos3SpeakerEncoderMirror {
    /// The pooling attention's hidden width.
    pub bottleneck_dim: usize,
    /// The embedding's width.
    pub embed_dim: usize,
    /// The mel bins the network reads.
    pub feat_dim: usize,
    /// A frame's length, in milliseconds.
    pub frame_length_ms: f64,
    /// The shift between frames, in milliseconds.
    pub frame_shift_ms: f64,
    /// The blocks' transformer width.
    pub hidden_dim: usize,
    /// The head's channels.
    pub in_channels: usize,
    /// The longest clip read, in seconds.
    pub max_seconds: f64,
    /// The token soft-MoE's experts.
    pub num_experts: usize,
    /// The filterbank's mel bins.
    pub num_mel_bins: usize,
    /// The rate the encoder reads, in Hz.
    pub sample_rate: usize,
    /// Each stage's blocks, `[res_blocks, channels, stride_freq,
    /// stride_time, expansion]`.
    pub stages: Vec<Vec<[usize; 5]>>,
    /// The clip's scale before the filterbank.
    pub waveform_scale: f64,
}

impl Zonos3SpeakerEncoderMirror {
    /// The encoder's config: the section's values, upstream's constants
    /// (4 heads, a 113-tap position convolution) and its resample
    /// (`torchaudio`'s `sinc_interp_hann` at its defaults).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) as
    /// [`Zonos3SpeakerEncoderConfig::validate`].
    pub fn speaker_config(&self) -> BunsenResult<Zonos3SpeakerEncoderConfig> {
        let stages = self
            .stages
            .iter()
            .map(|stage| {
                stage
                    .iter()
                    .map(
                        |&[res_blocks, channels, stride_freq, stride_time, expansion]| {
                            Zonos3MectBlockSpec {
                                res_blocks,
                                channels,
                                stride_freq,
                                stride_time,
                                expansion,
                            }
                        },
                    )
                    .collect()
            })
            .collect();
        let config = Zonos3SpeakerEncoderConfig::new(
            self.in_channels,
            self.feat_dim,
            self.hidden_dim,
            self.num_experts,
            self.bottleneck_dim,
            self.embed_dim,
            stages,
            Zonos3SpeakerFrontEndConfig::new(
                self.sample_rate,
                self.max_seconds,
                self.waveform_scale,
                self.num_mel_bins,
                self.frame_length_ms,
                self.frame_shift_ms,
            ),
        );
        config.validate()?;
        Ok(config)
    }
}

/// The prefix the speaker encoder's tensors sit under in the `-hf` file.
pub const ZONOS3_SPEAKER_PREFIX: &str = "speaker_encoder.";

/// One MECT block's names onto a
/// [`Zonos3MectBlock`](crate::kits::speech::tts::zonos3::blocks::Zonos3MectBlock)'s:
/// the `Sequential` indices of upstream's shortcut, reduce, projection and
/// position convolution onto named parts, and the attention's projections
/// onto burn's.
pub fn zonos3_mect_block_rekey() -> Rekey {
    let res_block = Rekey::keep_rest()
        .rename("shortcut.0.", "shortcut.conv.")
        .rename("shortcut.1.", "shortcut.bn.");
    Rekey::drop_rest()
        .mount_indexed("cnn_block.", "cnn.", res_block)
        .rename("reduce_conv.0.", "reduce.conv.")
        .rename("reduce_conv.1.", "reduce.bn.")
        .rename("reduce_project.0.", "reduce_project.")
        .rename("reduce_project.1.", "reduce_norm.")
        .rename("pos_conv.pos_conv.0.", "pos_conv.depthwise.")
        .rename("pos_conv.pos_conv.1.", "pos_conv.bn.")
        .rename("pos_conv.pos_conv.3.", "pos_conv.pointwise.")
        .rename(
            "transformer.attention.q_proj.",
            "transformer.attention.query.",
        )
        .rename(
            "transformer.attention.k_proj.",
            "transformer.attention.key.",
        )
        .rename(
            "transformer.attention.v_proj.",
            "transformer.attention.value.",
        )
        .rename(
            "transformer.attention.out_proj.",
            "transformer.attention.output.",
        )
        .rename("transformer.layer_norm.", "transformer.layer_norm.")
        .rename(
            "transformer.feed_forward.intermediate_dense.token_moe.",
            "transformer.moe.",
        )
        .rename(
            "transformer.feed_forward.output_dense.",
            "transformer.output_dense.",
        )
        .rename(
            "transformer.final_layer_norm.",
            "transformer.final_layer_norm.",
        )
        .rename("expansion_project.0.", "expansion_project.")
}

/// The encoder's names, without [`ZONOS3_SPEAKER_PREFIX`], onto a
/// [`Zonos3SpeakerEncoder`](crate::kits::speech::tts::zonos3::blocks::Zonos3SpeakerEncoder)'s
/// built from `config`: upstream's `mect_block{s}` (from 1) onto
/// `stages.{s - 1}`; the stage mix's `[1, n, D, 1]` as `[n, D]`; the
/// pooling's kernel-1 convolutions, `[out, in, 1]`, as its MLP's linears'
/// `[out, in]` (which the store's adapter transposes).
pub fn zonos3_speaker_rekey(config: &Zonos3SpeakerEncoderConfig) -> Rekey {
    let (n, width, bottleneck) = (config.n_stages(), config.width(), config.bottleneck_dim);
    Rekey::drop_rest()
        .rename("feature_extractor.head.0.", "head_conv.")
        .rename("feature_extractor.head.1.", "head_bn.")
        .mount_indexed_with(
            "feature_extractor.mect_block",
            "stages.",
            Indices::From(1),
            Rekey::drop_rest().mount_indexed("", "", zonos3_mect_block_rekey()),
        )
        .exact("feature_extractor.dense3.inputs_weights", "stage_mix")
        .rename("encoding_layers.", "pooling.attention.")
        .rename("bn.", "bn.")
        .rename("linear.", "linear.")
        .transform("stage_mix", TensorLoadTransform::reshape([n, width]))
        .transform(
            "pooling.attention.linear1.weight",
            TensorLoadTransform::reshape([bottleneck, width]),
        )
        .transform(
            "pooling.attention.linear2.weight",
            TensorLoadTransform::reshape([width, bottleneck]),
        )
}
