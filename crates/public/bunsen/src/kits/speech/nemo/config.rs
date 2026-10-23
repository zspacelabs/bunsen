//! Typed sections of a `NeMo` YAML config.
use burn::nn::activation::ActivationConfig;
use serde::Deserialize;

use super::{
    FeatureNormalization,
    NemoMelFrontEndConfig,
};
use crate::{
    blocks::{
        signals::DwStridingSubsamplingConfig,
        transformers::encoders::{
            conformer::{
                ConformerBlockConfig,
                ConformerEncoderConfig,
            },
            residual::{
                NormPlacement,
                ResidualEncoderAttentionBlockConfig,
            },
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming the
/// section's field and its value.
fn unsupported(
    section: &str,
    field: &str,
    value: impl std::fmt::Debug,
    why: &str,
) -> BunsenError {
    BunsenError::unsupported(format!("NeMo config: `{section}.{field}` {value:?}: {why}"))
}

/// Checks that a section instantiates `want` (its `_target_` is `want`, or
/// ends with `.{want}`).
fn expect_target(
    section: &str,
    target: &str,
    want: &[&str],
) -> BunsenResult<()> {
    let ok = want
        .iter()
        .any(|w| target == *w || target.ends_with(&format!(".{w}")));
    if ok {
        Ok(())
    } else {
        Err(unsupported(
            section,
            "_target_",
            target,
            &format!("not {}", want.join(" or ")),
        ))
    }
}

/// A `preprocessor` section: `AudioToMelSpectrogramPreprocessor`'s
/// arguments, named as `NeMo` names them and each defaulting as its
/// constructor defaults it.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NemoPreprocessorConfig {
    /// The class `NeMo` instantiates.
    #[serde(rename = "_target_")]
    pub target: String,

    /// In Hz.
    #[serde(default = "defaults::sample_rate")]
    pub sample_rate: usize,

    /// The analysis window, in seconds.
    #[serde(default = "defaults::window_size")]
    pub window_size: f64,

    /// The hop, in seconds.
    #[serde(default = "defaults::window_stride")]
    pub window_stride: f64,

    /// The window function.
    #[serde(default = "defaults::window")]
    pub window: String,

    /// The mel channel count.
    #[serde(default = "defaults::features")]
    pub features: usize,

    /// The FFT size; the window's length rounded up to a power of two when
    /// absent.
    #[serde(default)]
    pub n_fft: Option<usize>,

    /// `per_feature`, or `NA` for none.
    #[serde(default = "defaults::normalize")]
    pub normalize: String,

    /// The pre-emphasis coefficient; `null` for none.
    #[serde(default = "defaults::preemph")]
    pub preemph: Option<f64>,

    /// The spectrum's power.
    #[serde(default = "defaults::mag_power")]
    pub mag_power: f64,

    /// Whether the mel energies are logged.
    #[serde(default = "defaults::truth")]
    pub log: bool,

    /// How the log is kept finite: `add` or `clamp`.
    #[serde(default = "defaults::log_zero_guard_type")]
    pub log_zero_guard_type: String,

    /// The value added (or clamped to).
    #[serde(default = "defaults::log_zero_guard_value")]
    pub log_zero_guard_value: f64,

    /// Frames stacked into one feature.
    #[serde(default = "defaults::one")]
    pub frame_splicing: usize,

    /// The lowest mel frequency, in Hz.
    #[serde(default)]
    pub lowfreq: f64,

    /// The highest mel frequency, in Hz; Nyquist when absent.
    #[serde(default)]
    pub highfreq: Option<f64>,

    /// Whether frames are centred without padding (`exact_pad`).
    #[serde(default)]
    pub exact_pad: bool,
}

impl NemoPreprocessorConfig {
    /// The [`NemoMelFrontEndConfig`] this section describes, its tables in
    /// f32: a checkpoint that stores them in bf16 sets
    /// [`bf16_tables`](NemoMelFrontEndConfig::bf16_tables) itself.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the field and its value, for a setting bunsen's front end does not
    /// compute: a window other than Hann, an FFT size other than the
    /// window's rounded up to a power of two, a power other than 2, no log or
    /// a clamping guard, frame splicing, a band other than `[0, nyquist]`,
    /// exact padding, or a normalization other than `per_feature` and `NA`.
    pub fn front_end(&self) -> BunsenResult<NemoMelFrontEndConfig> {
        const S: &str = "preprocessor";
        expect_target(S, &self.target, &["AudioToMelSpectrogramPreprocessor"])?;
        let rate = self.sample_rate as f64;
        let win_length = (self.window_size * rate).round() as usize;
        let hop = (self.window_stride * rate).round() as usize;
        if self.window != "hann" {
            return Err(unsupported(S, "window", &self.window, "only hann"));
        }
        let n_fft = win_length.next_power_of_two();
        if self.n_fft.is_some_and(|n| n != n_fft) {
            return Err(unsupported(
                S,
                "n_fft",
                self.n_fft,
                &format!("the front end pads its {win_length}-sample window to {n_fft}"),
            ));
        }
        if self.mag_power != 2.0 {
            return Err(unsupported(S, "mag_power", self.mag_power, "only 2"));
        }
        if !self.log {
            return Err(unsupported(S, "log", self.log, "only logged mels"));
        }
        if self.log_zero_guard_type != "add" {
            return Err(unsupported(
                S,
                "log_zero_guard_type",
                &self.log_zero_guard_type,
                "only add",
            ));
        }
        if self.frame_splicing != 1 {
            return Err(unsupported(
                S,
                "frame_splicing",
                self.frame_splicing,
                "only 1",
            ));
        }
        if self.lowfreq != 0.0 {
            return Err(unsupported(S, "lowfreq", self.lowfreq, "only 0"));
        }
        if self.highfreq.is_some_and(|f| f != rate / 2.0) {
            return Err(unsupported(S, "highfreq", self.highfreq, "only nyquist"));
        }
        if self.exact_pad {
            return Err(unsupported(S, "exact_pad", self.exact_pad, "only false"));
        }
        let normalize = match self.normalize.as_str() {
            "per_feature" => FeatureNormalization::PerFeature,
            "NA" => FeatureNormalization::None,
            other => return Err(unsupported(S, "normalize", other, "only per_feature or NA")),
        };
        Ok(NemoMelFrontEndConfig::new()
            .with_sample_rate(self.sample_rate)
            .with_win_length(win_length)
            .with_hop(hop)
            .with_n_mels(self.features)
            .with_preemphasis(self.preemph.unwrap_or(0.0))
            .with_log_guard(self.log_zero_guard_value)
            .with_normalize(normalize))
    }
}

/// An `encoder` section for `ConformerEncoder`: the `FastConformer`'s
/// arguments, named as `NeMo` names them and each defaulting as its
/// constructor defaults it.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NemoConformerEncoderConfig {
    /// The class `NeMo` instantiates.
    #[serde(rename = "_target_")]
    pub target: String,

    /// The mel channels in.
    pub feat_in: usize,

    /// The Conformer blocks.
    pub n_layers: usize,

    /// The model width.
    pub d_model: usize,

    /// The subsampling: `dw_striding`.
    #[serde(default = "defaults::subsampling")]
    pub subsampling: String,

    /// The time reduction: a power of two.
    #[serde(default = "defaults::subsampling_factor")]
    pub subsampling_factor: usize,

    /// The subsampling's channels; `-1` for `d_model`.
    #[serde(default = "defaults::minus_one")]
    pub subsampling_conv_channels: i64,

    /// The subsampling's kernel.
    #[serde(default = "defaults::three")]
    pub subsampling_conv_kernel_size: usize,

    /// The subsampling's stride per stage.
    #[serde(default = "defaults::two")]
    pub subsampling_conv_stride: usize,

    /// Causal (left-padded) subsampling.
    #[serde(default)]
    pub causal_downsampling: bool,

    /// The feed-forward's width over `d_model`.
    #[serde(default = "defaults::four")]
    pub ff_expansion_factor: usize,

    /// The attention: `rel_pos`.
    #[serde(default = "defaults::rel_pos")]
    pub self_attention_model: String,

    /// The attention heads.
    #[serde(default = "defaults::four")]
    pub n_heads: usize,

    /// The attention's left and right context, in frames; `-1` for all.
    #[serde(default = "defaults::full_context")]
    pub att_context_size: Vec<i64>,

    /// Whether the subsampled input is scaled by `sqrt(d_model)`.
    #[serde(default = "defaults::truth")]
    pub xscaling: bool,

    /// Whether each layer has its own position biases.
    #[serde(default = "defaults::truth")]
    pub untie_biases: bool,

    /// The convolution module's depthwise kernel.
    #[serde(default = "defaults::conv_kernel_size")]
    pub conv_kernel_size: usize,

    /// The convolution module's norm: `batch_norm`.
    #[serde(default = "defaults::batch_norm")]
    pub conv_norm_type: String,

    /// The convolution's context; `null` for symmetric.
    #[serde(default)]
    pub conv_context_size: Option<Vec<i64>>,
}

impl NemoConformerEncoderConfig {
    /// The [`ConformerEncoderConfig`] this section describes.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the field and its value, for an encoder bunsen's `FastConformer` is
    /// not: subsampling other than depthwise-striding at kernel 3 and stride
    /// 2 by a power of two, causal subsampling, attention other than full
    /// relative-position with untied biases, a convolution norm other than
    /// batch norm, or an asymmetric convolution context.
    pub fn conformer(&self) -> BunsenResult<ConformerEncoderConfig> {
        const S: &str = "encoder";
        expect_target(S, &self.target, &["ConformerEncoder"])?;
        if self.subsampling != "dw_striding" {
            return Err(unsupported(
                S,
                "subsampling",
                &self.subsampling,
                "only dw_striding",
            ));
        }
        let factor = self.subsampling_factor;
        if factor < 2 || !factor.is_power_of_two() {
            return Err(unsupported(
                S,
                "subsampling_factor",
                factor,
                "a power of two from 2",
            ));
        }
        if self.subsampling_conv_kernel_size != 3 {
            return Err(unsupported(
                S,
                "subsampling_conv_kernel_size",
                self.subsampling_conv_kernel_size,
                "only 3",
            ));
        }
        if self.subsampling_conv_stride != 2 {
            return Err(unsupported(
                S,
                "subsampling_conv_stride",
                self.subsampling_conv_stride,
                "only 2",
            ));
        }
        if self.causal_downsampling {
            return Err(unsupported(S, "causal_downsampling", true, "only false"));
        }
        if self.self_attention_model != "rel_pos" {
            return Err(unsupported(
                S,
                "self_attention_model",
                &self.self_attention_model,
                "only rel_pos",
            ));
        }
        if self.att_context_size.iter().any(|&c| c != -1) {
            return Err(unsupported(
                S,
                "att_context_size",
                &self.att_context_size,
                "only full context, [-1, -1]",
            ));
        }
        if !self.untie_biases {
            return Err(unsupported(S, "untie_biases", false, "only true"));
        }
        if self.conv_norm_type != "batch_norm" {
            return Err(unsupported(
                S,
                "conv_norm_type",
                &self.conv_norm_type,
                "only batch_norm",
            ));
        }
        let half = (self.conv_kernel_size as i64 - 1) / 2;
        if let Some(context) = &self.conv_context_size
            && context != &[half, half]
        {
            return Err(unsupported(
                S,
                "conv_context_size",
                context,
                "only symmetric",
            ));
        }
        let channels = match self.subsampling_conv_channels {
            -1 => self.d_model,
            c if c > 0 => c as usize,
            c => {
                return Err(unsupported(
                    S,
                    "subsampling_conv_channels",
                    c,
                    "positive or -1",
                ));
            }
        };
        let subsampling = DwStridingSubsamplingConfig::new(self.feat_in, channels, self.d_model)
            .with_n_stages(factor.trailing_zeros() as usize);
        let block = ConformerBlockConfig::new(
            self.d_model,
            self.n_heads,
            self.d_model * self.ff_expansion_factor,
            self.conv_kernel_size,
        );
        Ok(
            ConformerEncoderConfig::new(subsampling, block, self.n_layers)
                .with_scale_input(self.xscaling),
        )
    }
}

/// A `transformer_encoder` section for `NeMo`'s
/// `transformer.transformer_encoders.TransformerEncoder`: the arguments,
/// named as `NeMo` names them and each defaulting as its constructor defaults
/// it.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NemoTransformerEncoderConfig {
    /// The class `NeMo` instantiates.
    #[serde(rename = "_target_")]
    pub target: String,

    /// The layers.
    pub num_layers: usize,

    /// The model width.
    pub hidden_size: usize,

    /// The feed-forward's hidden width.
    pub inner_size: usize,

    /// The attention heads.
    #[serde(default = "defaults::one")]
    pub num_attention_heads: usize,

    /// The feed-forward's activation: `relu` or `gelu`.
    #[serde(default = "defaults::relu")]
    pub hidden_act: String,

    /// Whether each layer normalizes before its sublayers, rather than after
    /// each residual sum: [`NormPlacement::Pre`] or [`NormPlacement::Post`].
    #[serde(default)]
    pub pre_ln: bool,

    /// Whether a pre-norm stack normalizes once more after its last layer;
    /// a post-norm stack never does.
    #[serde(default = "defaults::truth")]
    pub pre_ln_final_layer_norm: bool,

    /// Whether attention is causal.
    #[serde(default)]
    pub mask_future: bool,
}

impl NemoTransformerEncoderConfig {
    /// The layer this section stacks, as a
    /// [`ResidualEncoderAttentionBlockConfig`]: every projection with a
    /// bias, as `NeMo`'s `MultiHeadAttention` and `PositionWiseFF` have.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the field and its value, for causal attention, an activation other
    /// than `relu` and `gelu`, or a width the heads or the feed-forward do
    /// not divide.
    pub fn block(&self) -> BunsenResult<ResidualEncoderAttentionBlockConfig> {
        const S: &str = "transformer_encoder";
        expect_target(
            S,
            &self.target,
            &[
                "transformer.transformer_encoders.TransformerEncoder",
                "transformer.TransformerEncoder",
            ],
        )?;
        if self.mask_future {
            return Err(unsupported(S, "mask_future", true, "only bidirectional"));
        }
        let activation = match self.hidden_act.as_str() {
            "relu" => ActivationConfig::Relu,
            "gelu" => ActivationConfig::Gelu,
            other => return Err(unsupported(S, "hidden_act", other, "only relu or gelu")),
        };
        let (d, heads) = (self.hidden_size, self.num_attention_heads);
        if heads == 0 || !d.is_multiple_of(heads) {
            return Err(unsupported(
                S,
                "num_attention_heads",
                heads,
                &format!("does not divide hidden_size {d}"),
            ));
        }
        if d == 0 || !self.inner_size.is_multiple_of(d) {
            return Err(unsupported(
                S,
                "inner_size",
                self.inner_size,
                &format!("not a whole multiple of hidden_size {d}"),
            ));
        }
        Ok(ResidualEncoderAttentionBlockConfig::new(d)
            .with_d_head(d / heads)
            .with_expansion_factor(self.inner_size / d)
            .with_key_bias(true)
            .with_norm_placement(if self.pre_ln {
                NormPlacement::Pre
            } else {
                NormPlacement::Post
            })
            .with_activation(activation))
    }

    /// Whether the stack ends with a norm: `NeMo` adds it only to a pre-norm
    /// stack ([`pre_ln`](Self::pre_ln) and
    /// [`pre_ln_final_layer_norm`](Self::pre_ln_final_layer_norm)).
    pub fn final_norm(&self) -> bool {
        self.pre_ln && self.pre_ln_final_layer_norm
    }
}

/// `NeMo`'s constructor defaults.
mod defaults {
    pub fn sample_rate() -> usize {
        16000
    }
    pub fn window_size() -> f64 {
        0.02
    }
    pub fn window_stride() -> f64 {
        0.01
    }
    pub fn window() -> String {
        "hann".into()
    }
    pub fn features() -> usize {
        64
    }
    pub fn normalize() -> String {
        "per_feature".into()
    }
    pub fn preemph() -> Option<f64> {
        Some(0.97)
    }
    pub fn mag_power() -> f64 {
        2.0
    }
    pub fn truth() -> bool {
        true
    }
    pub fn log_zero_guard_type() -> String {
        "add".into()
    }
    pub fn log_zero_guard_value() -> f64 {
        2f64.powi(-24)
    }
    pub fn one() -> usize {
        1
    }
    pub fn two() -> usize {
        2
    }
    pub fn three() -> usize {
        3
    }
    pub fn four() -> usize {
        4
    }
    pub fn minus_one() -> i64 {
        -1
    }
    pub fn subsampling() -> String {
        "striding".into()
    }
    pub fn subsampling_factor() -> usize {
        4
    }
    pub fn rel_pos() -> String {
        "rel_pos".into()
    }
    pub fn full_context() -> Vec<i64> {
        vec![-1, -1]
    }
    pub fn conv_kernel_size() -> usize {
        31
    }
    pub fn batch_norm() -> String {
        "batch_norm".into()
    }
    pub fn relu() -> String {
        "relu".into()
    }
}
