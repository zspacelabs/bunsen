//! [`Zonos3SpeakerEncoder`]: the MECT-B2 speaker encoder, its front end and its config.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        BatchNorm,
        Linear,
        LinearConfig,
        conv::Conv2d,
    },
    prelude::*,
    tensor::activation::softmax,
};

use super::{
    Zonos3MectBlock,
    Zonos3MectBlockSpec,
    mect::{
        batch_norm,
        conv2d,
    },
};
use crate::{
    blocks::pooling::{
        AttentiveStatsPooling,
        AttentiveStatsPoolingConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::tts::zonos3::blocks::Zonos3MectBlockGeometry,
    ops::signal::{
        KaldiFbank,
        KaldiFbankConfig,
        KaldiFbankMeta,
        kernels::window::StftWindowConfig,
        resample::HannSincResampler,
    },
};

/// Common meta for [`Zonos3SpeakerEncoder`](super::Zonos3SpeakerEncoder) and
/// [`Zonos3SpeakerEncoderConfig`].
pub trait Zonos3SpeakerEncoderMeta {
    /// The mel bins the network reads, `M`.
    fn mel_bins(&self) -> usize;

    /// The width between the stages, `D`: the head's channels times the mel
    /// bins, and every block's channels times its frequencies.
    fn width(&self) -> usize;

    /// The embedding's width, `S`.
    fn embed_dim(&self) -> usize;

    /// The stages.
    fn n_stages(&self) -> usize;
}

/// The front end before the network: the clip at the encoder's rate, cut,
/// scaled, then Kaldi filterbank features with the per-utterance mean
/// subtracted.
#[derive(Config, Debug)]
pub struct Zonos3SpeakerFrontEndConfig {
    /// The rate the encoder reads, in Hz.
    pub sample_rate: usize,

    /// The longest clip read, in seconds; the rest is cut.
    pub max_seconds: f64,

    /// The clip's scale before the filterbank (`32768`, `int16`'s).
    pub waveform_scale: f64,

    /// The mel bins.
    pub num_mel_bins: usize,

    /// A frame's length, in milliseconds.
    pub frame_length_ms: f64,

    /// The shift between frames, in milliseconds.
    pub frame_shift_ms: f64,

    /// The resample to the encoder's rate (upstream's: `torchaudio`'s
    /// `sinc_interp_hann`, at its defaults).
    #[config(default = "HannSincResampler::new()")]
    pub resampler: HannSincResampler,
}

impl Zonos3SpeakerFrontEndConfig {
    /// The filterbank: Kaldi's, at the encoder's rate, a Hamming window,
    /// dither 0, the mean over the frames subtracted.
    pub fn fbank(&self) -> KaldiFbankConfig {
        KaldiFbankConfig::new(StftWindowConfig::Hamming { periodic: false })
            .with_sample_rate(self.sample_rate as f64)
            .with_num_mel_bins(self.num_mel_bins)
            .with_frame_length_ms(self.frame_length_ms)
            .with_frame_shift_ms(self.frame_shift_ms)
            .with_subtract_mean(true)
    }

    /// The most samples read at the encoder's rate.
    pub fn max_samples(&self) -> usize {
        (self.max_seconds * self.sample_rate as f64) as usize
    }
}

/// Config for [`Zonos3SpeakerEncoder`](super::Zonos3SpeakerEncoder): the
/// MECT-B2 network (`AntSpeaker`'s, with ASTP pooling) and its front end.
///
/// A release's comes from `config.json`'s `speaker_encoder` section
/// ([`Zonos3SpeakerEncoderMirror::speaker_config`](crate::kits::speech::tts::zonos3::interop::zonos::Zonos3SpeakerEncoderMirror::speaker_config)).
///
/// Implements [`Zonos3SpeakerEncoderMeta`].
#[derive(Config, Debug)]
pub struct Zonos3SpeakerEncoderConfig {
    /// The head's channels, `C`.
    pub in_channels: usize,

    /// The mel bins the network reads, `M`.
    pub feat_dim: usize,

    /// The blocks' transformer width, `H`.
    pub hidden_dim: usize,

    /// The token soft-MoE's experts.
    pub num_experts: usize,

    /// The pooling attention's hidden width.
    pub bottleneck_dim: usize,

    /// The embedding's width, `S`.
    pub embed_dim: usize,

    /// The stages, each its blocks.
    pub stages: Vec<Vec<Zonos3MectBlockSpec>>,

    /// The front end.
    pub front_end: Zonos3SpeakerFrontEndConfig,

    /// The blocks' attention heads (upstream's constant).
    #[config(default = 4)]
    pub heads: usize,

    /// The position convolution's taps (upstream's constant).
    #[config(default = 113)]
    pub pos_conv_kernel: usize,
}

impl Zonos3SpeakerEncoderMeta for Zonos3SpeakerEncoderConfig {
    fn mel_bins(&self) -> usize {
        self.feat_dim
    }

    fn width(&self) -> usize {
        self.in_channels * self.feat_dim
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn n_stages(&self) -> usize {
        self.stages.len()
    }
}

impl Zonos3SpeakerEncoderConfig {
    /// Each stage's blocks' geometry, walked from the head: a block reads
    /// the channels and frequencies the one before wrote.
    pub fn geometry(&self) -> Vec<Vec<Zonos3MectBlockGeometry>> {
        let (mut channels, mut freqs) = (self.in_channels, self.feat_dim);
        self.stages
            .iter()
            .map(|stage| {
                stage
                    .iter()
                    .map(|&spec| {
                        let block = Zonos3MectBlockGeometry {
                            spec,
                            in_channels: channels,
                            in_freqs: freqs,
                        };
                        channels = spec.channels;
                        freqs /= spec.stride_freq.max(1);
                        block
                    })
                    .collect()
            })
            .collect()
    }

    /// Validates the geometry.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal): no stages or
    /// an empty one; a block of no residual blocks, channels, strides or
    /// expansion; frequencies a block's stride does not divide; a block
    /// whose output width (`c f`) is not the input's (`C M`), which its
    /// residual adds to; a block whose channels are not its input's times
    /// its frequency stride (a reduce convolution writes that many, and the
    /// next block reads `c`), or whose residual channels its input's do not
    /// divide (the grouped convolutions' groups); a hidden width the heads
    /// do not divide; a front end whose mel bins are not the network's.
    pub fn validate(&self) -> BunsenResult<()> {
        let illegal = |what: alloc::string::String| {
            Err(BunsenError::illegal(format!(
                "a ZONOS3 speaker encoder with {what}"
            )))
        };
        if self.stages.is_empty() || self.stages.iter().any(Vec::is_empty) {
            return illegal(format!("stages {:?}: none, or an empty one", self.stages));
        }
        if self.heads == 0 || !self.hidden_dim.is_multiple_of(self.heads) {
            return illegal(format!(
                "hidden width {} over {} heads",
                self.hidden_dim, self.heads
            ));
        }
        if self.front_end.num_mel_bins != self.feat_dim {
            return illegal(format!(
                "a front end of {} mel bins over a network of {}",
                self.front_end.num_mel_bins, self.feat_dim
            ));
        }
        let width = self.width();
        for (s, stage) in self.geometry().iter().enumerate() {
            for (j, block) in stage.iter().enumerate() {
                let spec = block.spec;
                let at = format!("block {s}.{j} ({spec:?})");
                if [
                    spec.res_blocks,
                    spec.channels,
                    spec.stride_freq,
                    spec.stride_time,
                    spec.expansion,
                ]
                .contains(&0)
                {
                    return illegal(format!("{at}: a zero"));
                }
                if !block.in_freqs.is_multiple_of(spec.stride_freq) {
                    return illegal(format!("{at}: {} frequencies", block.in_freqs));
                }
                if spec.channels * block.freqs() != width {
                    return illegal(format!(
                        "{at}: an output width of {}, not {width}",
                        spec.channels * block.freqs()
                    ));
                }
                if spec.channels != block.in_channels * spec.stride_freq {
                    return illegal(format!(
                        "{at}: {} input channels; its output is its input's times its frequency \
                         stride (the reduce convolution's, upstream's next block reads `c`)",
                        block.in_channels
                    ));
                }
                if !block.res_channels().is_multiple_of(block.in_channels) {
                    return illegal(format!(
                        "{at}: residual channels {} in groups of its {} input channels",
                        block.res_channels(),
                        block.in_channels
                    ));
                }
            }
        }
        Ok(())
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
    /// [`KaldiFbankConfig::try_validate`](crate::ops::signal::KaldiFbankConfig::try_validate).
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
