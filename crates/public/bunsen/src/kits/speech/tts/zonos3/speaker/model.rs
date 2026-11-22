//! The MECT-B2 speaker encoder and its parts.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    module::{
        Module,
        Param,
    },
    nn::{
        BatchNorm,
        BatchNormConfig,
        Linear,
        LinearConfig,
        PaddingConfig1d,
        PaddingConfig2d,
        attention::{
            MhaInput,
            MultiHeadAttention,
            MultiHeadAttentionConfig,
        },
        conv::{
            Conv1d,
            Conv1dConfig,
            Conv2d,
            Conv2dConfig,
        },
    },
    prelude::*,
    tensor::{
        DType,
        activation::{
            gelu,
            gelu_approximate,
            relu,
            softmax,
        },
    },
};

use super::{
    Zonos3MectBlockGeometry,
    Zonos3SpeakerEncoderConfig,
    Zonos3SpeakerEncoderMeta,
};
use crate::{
    blocks::{
        norm::{
            BunsenLayerNormConfig,
            LayerNorm,
        },
        pooling::{
            AttentiveStatsPooling,
            AttentiveStatsPoolingConfig,
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::signal::{
        KaldiFbank,
        KaldiFbankMeta,
    },
    support::audio::HannSincResampler,
};

/// A 2-d convolution, `[k, k]`, `padding` on every side.
fn conv2d(
    [cin, cout]: [usize; 2],
    kernel: usize,
    stride: [usize; 2],
    groups: usize,
    bias: bool,
    device: &Device,
) -> Conv2d {
    let pad = kernel / 2;
    Conv2dConfig::new([cin, cout], [kernel, kernel])
        .with_stride(stride)
        .with_groups(groups)
        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
        .with_bias(bias)
        .init(device)
}

/// A 1-d convolution over time, `kernel` taps, `kernel / 2` padding each
/// side.
fn conv1d(
    [cin, cout]: [usize; 2],
    kernel: usize,
    groups: usize,
    device: &Device,
) -> Conv1d {
    Conv1dConfig::new(cin, cout, kernel)
        .with_groups(groups)
        .with_padding(PaddingConfig1d::Explicit(kernel / 2, kernel / 2))
        .init(device)
}

/// A batch norm over `channels`, `1e-5` (`PyTorch`'s).
fn batch_norm(
    channels: usize,
    device: &Device,
) -> BatchNorm {
    BatchNormConfig::new(channels).init(device)
}

/// A layer norm over `width`, `1e-6` (upstream's).
fn layer_norm(
    width: usize,
    device: &Device,
) -> LayerNorm {
    BunsenLayerNormConfig::new(width)
        .with_epsilon(1e-6)
        .init(device)
}

/// A residual block's strided 1x1 shortcut: a convolution and its norm.
#[derive(Module, Debug)]
pub struct Zonos3MectShortcut {
    /// `[c_out, c_in, 1, 1]`, strided, no bias.
    pub conv: Conv2d,

    /// Its norm.
    pub bn: BatchNorm,
}

/// One residual block (`AntSpeaker`'s `BasicResBlock`): two grouped 3x3
/// convolutions, each followed by a pointwise one, a norm and a `ReLU`, plus
/// the input, through a 1x1 shortcut where the block strides or changes
/// channels.
///
/// The grouped convolutions' groups are the block's input channels, for
/// both: the second is grouped over its input's channels too.
#[derive(Module, Debug)]
pub struct Zonos3MectResBlock {
    /// The first 3x3, `c_in -> c_out` in `c_in` groups, strided, no bias.
    pub conv1: Conv2d,

    /// Pointwise, `c_out -> c_out`.
    pub conv_pw1: Conv2d,

    /// Its norm.
    pub bn1: BatchNorm,

    /// The second 3x3, `c_out -> c_out` in `c_in` groups, no bias.
    pub conv2: Conv2d,

    /// Pointwise, `c_out -> c_out`.
    pub conv_pw2: Conv2d,

    /// Its norm.
    pub bn2: BatchNorm,

    /// The shortcut; `None`, the input itself.
    pub shortcut: Option<Zonos3MectShortcut>,
}

impl Zonos3MectResBlock {
    fn new(
        [cin, cout]: [usize; 2],
        stride: [usize; 2],
        device: &Device,
    ) -> Self {
        let shortcut = (stride != [1, 1] || cin != cout).then(|| Zonos3MectShortcut {
            conv: Conv2dConfig::new([cin, cout], [1, 1])
                .with_stride(stride)
                .with_bias(false)
                .init(device),
            bn: batch_norm(cout, device),
        });
        Self {
            conv1: conv2d([cin, cout], 3, stride, cin, false, device),
            conv_pw1: conv2d([cout, cout], 1, [1, 1], 1, true, device),
            bn1: batch_norm(cout, device),
            conv2: conv2d([cout, cout], 3, [1, 1], cin, false, device),
            conv_pw2: conv2d([cout, cout], 1, [1, 1], 1, true, device),
            bn2: batch_norm(cout, device),
            shortcut,
        }
    }

    /// `x`, `[B, c_in, f_in, t_in]`, to `[B, c_out, f, t]`.
    pub fn forward(
        &self,
        x: Tensor<4>,
    ) -> Tensor<4> {
        let out = relu(
            self.bn1
                .forward(self.conv_pw1.forward(self.conv1.forward(x.clone()))),
        );
        let out = relu(
            self.bn2
                .forward(self.conv_pw2.forward(self.conv2.forward(out))),
        );
        let skip = match &self.shortcut {
            Some(shortcut) => shortcut.bn.forward(shortcut.conv.forward(x)),
            None => x,
        };
        out + skip
    }
}

/// The reduce convolution of an expanding block: 1x1 back to its output
/// channels, and its norm.
#[derive(Module, Debug)]
pub struct Zonos3MectReduce {
    /// `[c, c_res, 1, 1]`, with a bias.
    pub conv: Conv2d,

    /// Its norm.
    pub bn: BatchNorm,
}

/// The position convolution: a depthwise convolution over time, a norm, an
/// exact GELU, and a pointwise convolution.
#[derive(Module, Debug)]
pub struct Zonos3MectPosConv {
    /// Depthwise, `[H, 1, k]`, `k / 2` zeros each side.
    pub depthwise: Conv1d,

    /// Its norm.
    pub bn: BatchNorm,

    /// Pointwise, `[H, H, 1]`.
    pub pointwise: Conv1d,
}

impl Zonos3MectPosConv {
    /// `x`, `[B, H, T]`, to `[B, H, T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        self.pointwise
            .forward(gelu(self.bn.forward(self.depthwise.forward(x))))
    }
}

/// The token soft mixture of experts: every expert on every frame, mixed by
/// a softmax gate (`AntSpeaker`'s `TokenSMoE`, no routing).
#[derive(Module, Debug)]
pub struct Zonos3MectTokenSoftMoe {
    /// The gate, `H -> E`, no bias.
    pub gate: Linear,

    /// The experts, packed, `H -> E H`, no bias: expert `e`'s output is
    /// columns `e H .. (e + 1) H`.
    pub experts: Linear,
}

impl Zonos3MectTokenSoftMoe {
    /// `x`, `[B, T, H]`: `Σ_e softmax(gate(x))_e gelu_tanh(expert_e(x))`,
    /// `[B, T, H]`; the gate's softmax in f32.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [batch, frames, width] = x.dims();
        let experts = self.gate.weight.dims()[1];
        let dtype = x.dtype();
        let weights = softmax(self.gate.forward(x.clone()).cast(DType::F32), 2).cast(dtype);
        let outputs =
            gelu_approximate(self.experts.forward(x)).reshape([batch, frames, experts, width]);
        (outputs * weights.unsqueeze_dim(3))
            .sum_dim(2)
            .reshape([batch, frames, width])
    }
}

/// A block's transformer layer, post-norm: `ln(x + attention(x))`, then
/// `ln(x + output_dense(gelu_tanh(moe(x))))`.
#[derive(Module, Debug)]
pub struct Zonos3MectTransformerLayer {
    /// Self-attention, no mask, no positions.
    pub attention: MultiHeadAttention,

    /// After the attention's residual.
    pub layer_norm: LayerNorm,

    /// The feed-forward's mixture.
    pub moe: Zonos3MectTokenSoftMoe,

    /// The feed-forward's output, `H -> H`.
    pub output_dense: Linear,

    /// After the feed-forward's residual.
    pub final_layer_norm: LayerNorm,
}

impl Zonos3MectTransformerLayer {
    /// `x`, `[B, H, T]` channels first, to `[B, H, T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let x = x.swap_dims(1, 2);
        let attended = self
            .attention
            .forward(MhaInput::self_attn(x.clone()))
            .context;
        let x = self.layer_norm.forward(x + attended);
        let ff = self
            .output_dense
            .forward(gelu_approximate(self.moe.forward(x.clone())));
        self.final_layer_norm.forward(x + ff).swap_dims(1, 2)
    }
}

/// One MECT block: residual convolutions over `[c, f, t]`, then, over the
/// flattened `c f` channels at the strided rate, a projection to `H`, the
/// position convolution and a transformer layer, a projection back with a
/// residual, repeated to the input's rate, plus the input.
#[derive(Module, Debug)]
pub struct Zonos3MectBlock {
    /// The residual blocks.
    pub cnn: Vec<Zonos3MectResBlock>,

    /// The reduce convolution, when the block expands.
    pub reduce: Option<Zonos3MectReduce>,

    /// `D -> H`, `[H, D, 1]`.
    pub reduce_project: Conv1d,

    /// Over the projection's channels.
    pub reduce_norm: LayerNorm,

    /// The position convolution.
    pub pos_conv: Zonos3MectPosConv,

    /// The transformer layer.
    pub transformer: Zonos3MectTransformerLayer,

    /// `H -> D`, `[D, H, 1]`.
    pub expansion_project: Conv1d,

    /// The time stride the transformer runs at, repeated back.
    pub scale_time: usize,
}

impl Zonos3MectBlock {
    fn new(
        block: &Zonos3MectBlockGeometry,
        config: &Zonos3SpeakerEncoderConfig,
        device: &Device,
    ) -> BunsenResult<Self> {
        let spec = block.spec;
        let (width, hidden) = (config.width(), config.hidden_dim);
        let res = block.res_channels();
        let cnn = (0..spec.res_blocks)
            .map(|i| {
                let (cin, stride) = if i == 0 {
                    (block.in_channels, [spec.stride_freq, spec.stride_time])
                } else {
                    (res, [1, 1])
                };
                Zonos3MectResBlock::new([cin, res], stride, device)
            })
            .collect();
        let reduce = (spec.expansion > 1).then(|| Zonos3MectReduce {
            conv: conv2d([res, spec.channels], 1, [1, 1], 1, true, device),
            bn: batch_norm(spec.channels, device),
        });
        Ok(Self {
            cnn,
            reduce,
            reduce_project: conv1d([width, hidden], 1, 1, device),
            reduce_norm: layer_norm(hidden, device),
            pos_conv: Zonos3MectPosConv {
                depthwise: conv1d([hidden, hidden], config.pos_conv_kernel, hidden, device),
                bn: batch_norm(hidden, device),
                pointwise: conv1d([hidden, hidden], 1, 1, device),
            },
            transformer: Zonos3MectTransformerLayer {
                attention: MultiHeadAttentionConfig::new(hidden, config.heads)
                    .with_dropout(0.0)
                    .init(device),
                layer_norm: layer_norm(hidden, device),
                moe: Zonos3MectTokenSoftMoe {
                    gate: LinearConfig::new(hidden, config.num_experts)
                        .with_bias(false)
                        .init(device),
                    experts: LinearConfig::new(hidden, config.num_experts * hidden)
                        .with_bias(false)
                        .init(device),
                },
                output_dense: LinearConfig::new(hidden, hidden).init(device),
                final_layer_norm: layer_norm(hidden, device),
            },
            expansion_project: conv1d([hidden, width], 1, 1, device),
            scale_time: spec.stride_time,
        })
    }

    /// `x`, `[B, c_in, f_in, T]`, to `[B, c, f, T]`, with `c_in f_in = c f =
    /// D`.
    pub fn forward(
        &self,
        x: Tensor<4>,
    ) -> Tensor<4> {
        let [batch, c_in, f_in, frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "c_in", "f_in", "frames"],
            &x,
            &["batch", "c_in", "f_in", "frames"],
        );
        let width = c_in * f_in;
        let out = self.cnn.iter().fold(x.clone(), |h, block| block.forward(h));
        let out = match &self.reduce {
            Some(reduce) => reduce.bn.forward(reduce.conv.forward(out)),
            None => out,
        };
        let [_, c, f, strided] = out.dims();
        // Frequency-major, `[f, c]` flattened, as upstream's `permute(0, 2, 1,
        // 3)`.
        let flat = out.permute([0, 2, 1, 3]).reshape([batch, f * c, strided]);
        let h = self.reduce_project.forward(flat.clone());
        let h = self.reduce_norm.forward(h.swap_dims(1, 2)).swap_dims(1, 2);
        let h = self
            .transformer
            .forward(h.clone() + self.pos_conv.forward(h));
        let out = self.expansion_project.forward(h) + flat;
        let out = out
            .unsqueeze_dim::<4>(3)
            .repeat_dim(3, self.scale_time)
            .reshape([batch, width, strided * self.scale_time])
            .slice_dim(2, 0..frames as isize);
        let out = out + x.permute([0, 2, 1, 3]).reshape([batch, width, frames]);
        out.reshape([batch, f, c, frames]).permute([0, 2, 1, 3])
    }
}

/// Each stage of a [`Zonos3SpeakerEncoder`]'s pass, for tests.
#[derive(Clone, Debug)]
pub struct Zonos3SpeakerTrace {
    /// The head's output, `[B, C, M, T]`.
    pub head: Tensor<4>,

    /// Each block's output, stage by stage, `[B, c, f, T]`.
    pub blocks: Vec<Vec<Tensor<4>>>,

    /// The stages' mix, `[B, D, T]`.
    pub mix: Tensor<3>,

    /// The pooled statistics, `[B, 2 D]`.
    pub pooled: Tensor<2>,

    /// The embedding, `[B, S]`.
    pub embedding: Tensor<2>,
}

/// ZONOS3-Mini's speaker encoder (`ownmect_s2`): a clip to a raw,
/// unnormalized speaker embedding, the vector its prompt's speaker row
/// projects.
///
/// `AntSpeaker`'s MECT-B2 with ASTP pooling, behind a Kaldi filterbank: a
/// 3x3 head over the `[M, T]` features, stages of MECT blocks
/// ([`Zonos3MectBlock`]), a softmax-weighted mix of every stage's last
/// output, attentive statistics pooling, a batch norm and a projection to
/// `S`.
///
/// Built by [`Zonos3SpeakerEncoderConfig`]. Implements
/// [`Zonos3SpeakerEncoderMeta`].
#[derive(Module, Debug)]
pub struct Zonos3SpeakerEncoder {
    /// `1 -> C`, 3x3, with a bias.
    pub head_conv: Conv2d,

    /// Its norm.
    pub head_bn: BatchNorm,

    /// The stages' blocks.
    pub stages: Vec<Vec<Zonos3MectBlock>>,

    /// The stages' mix logits, `[n_stages, D]`: a softmax over the stages,
    /// per channel.
    pub stage_mix: Param<Tensor<2>>,

    /// The pooling.
    pub pooling: AttentiveStatsPooling,

    /// Over the pooled statistics.
    pub bn: BatchNorm,

    /// `2 D -> S`, no bias.
    pub linear: Linear,

    /// The front end's filterbank.
    pub fbank: KaldiFbank,

    /// The rate the encoder reads, in Hz.
    pub sample_rate: usize,

    /// The most samples read.
    pub max_samples: usize,

    /// The clip's scale before the filterbank.
    pub waveform_scale: f64,

    /// The resample to the encoder's rate.
    #[module(skip)]
    pub resampler: HannSincResampler,
}

impl ModuleInit<Zonos3SpeakerEncoder> for Zonos3SpeakerEncoderConfig {
    /// # Errors
    /// See [`validate`](Zonos3SpeakerEncoderConfig::validate) and
    /// [`KaldiFbankConfig::validate`](crate::ops::signal::KaldiFbankConfig::validate).
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Zonos3SpeakerEncoder> {
        self.validate()?;
        let width = self.width();
        let stages = self
            .geometry()
            .iter()
            .map(|stage| {
                stage
                    .iter()
                    .map(|block| Zonos3MectBlock::new(block, self, device))
                    .collect::<BunsenResult<Vec<_>>>()
            })
            .collect::<BunsenResult<Vec<_>>>()?;
        Ok(Zonos3SpeakerEncoder {
            head_conv: conv2d([1, self.in_channels], 3, [1, 1], 1, true, device),
            head_bn: batch_norm(self.in_channels, device),
            stages,
            stage_mix: Param::from_tensor(Tensor::zeros([self.stages.len(), width], device)),
            pooling: AttentiveStatsPoolingConfig::new(width, self.bottleneck_dim)
                .try_init(device)?,
            bn: batch_norm(2 * width, device),
            linear: LinearConfig::new(2 * width, self.embed_dim)
                .with_bias(false)
                .init(device),
            fbank: self.front_end.fbank().try_init(device)?,
            sample_rate: self.front_end.sample_rate,
            max_samples: self.front_end.max_samples(),
            waveform_scale: self.front_end.waveform_scale,
            resampler: self.front_end.resampler,
        })
    }
}

impl Zonos3SpeakerEncoderMeta for Zonos3SpeakerEncoder {
    fn mel_bins(&self) -> usize {
        self.fbank.num_mel_bins()
    }

    fn width(&self) -> usize {
        self.stage_mix.dims()[1]
    }

    fn embed_dim(&self) -> usize {
        self.linear.weight.dims()[1]
    }

    fn n_stages(&self) -> usize {
        self.stages.len()
    }
}

impl Zonos3SpeakerEncoder {
    /// The embeddings of `features`, `[B, M, T]` (a mean-normalized
    /// filterbank's, mel bins first): `[B, S]`.
    pub fn forward(
        &self,
        features: Tensor<3>,
    ) -> Tensor<2> {
        self.forward_trace(features).embedding
    }

    /// [`forward`](Self::forward), keeping each stage's output.
    pub fn forward_trace(
        &self,
        features: Tensor<3>,
    ) -> Zonos3SpeakerTrace {
        let [batch, _, frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "mels", "frames"],
            &features,
            &["batch", "mels", "frames"],
            &[("mels", self.mel_bins())],
        );
        let width = self.width();
        let head = self
            .head_bn
            .forward(self.head_conv.forward(features.unsqueeze_dim(1)));
        let mut x = head.clone();
        let mut blocks = Vec::with_capacity(self.stages.len());
        let mut lasts = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let mut outputs = Vec::with_capacity(stage.len());
            for block in stage {
                x = block.forward(x);
                outputs.push(x.clone());
            }
            lasts.push(
                x.clone()
                    .permute([0, 2, 1, 3])
                    .reshape([batch, 1, width, frames]),
            );
            blocks.push(outputs);
        }
        let n = self.stages.len();
        let weights = softmax(self.stage_mix.val(), 0).reshape([1, n, width, 1]);
        let mix = (Tensor::cat(lasts, 1) * weights)
            .sum_dim(1)
            .reshape([batch, width, frames]);
        let pooled = self.pooling.forward(mix.clone());
        let embedding = self.linear.forward(self.bn.forward(pooled.clone()));
        Zonos3SpeakerTrace {
            head,
            blocks,
            mix,
            pooled,
            embedding,
        }
    }

    /// The samples the network reads of `samples`, mono at `rate`:
    /// resampled to the encoder's rate, cut to its longest clip, scaled.
    pub fn prepare(
        &self,
        samples: &[f32],
        rate: usize,
    ) -> Vec<f32> {
        let mut x = self.resampler.resample(samples, rate, self.sample_rate);
        x.truncate(self.max_samples);
        let scale = self.waveform_scale as f32;
        x.iter_mut().for_each(|v| *v *= scale);
        x
    }

    /// The features of `prepared` samples ([`prepare`](Self::prepare)'s):
    /// the filterbank, mean-normalized, `[1, M, T]`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for fewer
    /// samples than one frame.
    pub fn features(
        &self,
        prepared: &[f32],
    ) -> BunsenResult<Tensor<3>> {
        let frame = self.fbank.frame_len();
        if prepared.len() < frame {
            return Err(BunsenError::illegal(format!(
                "a speaker clip of {} samples at {} Hz; the encoder reads at least one frame, \
                 {frame}",
                prepared.len(),
                self.sample_rate
            )));
        }
        let device = self.linear.weight.device();
        let signal = Tensor::<2>::from_data(
            TensorData::new(prepared.to_vec(), [1, prepared.len()]),
            &device,
        );
        Ok(self.fbank.forward(signal).swap_dims(1, 2))
    }

    /// The embedding of `samples`, mono at `rate`, as upstream's encoder
    /// takes a clip: [`prepare`](Self::prepare), then
    /// [`features`](Self::features), then [`forward`](Self::forward); `[1,
    /// S]`.
    ///
    /// # Errors
    /// As [`features`](Self::features).
    pub fn embed(
        &self,
        samples: &[f32],
        rate: usize,
    ) -> BunsenResult<Tensor<2>> {
        let features = self.features(&self.prepare(samples, rate))?;
        Ok(self.forward(features))
    }
}
