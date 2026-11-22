//! The speaker encoder's config.

use alloc::{
    format,
    vec::Vec,
};

use burn::config::Config;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    ops::signal::{
        KaldiFbankConfig,
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

/// One MECT block of a stage, as `config.json` writes it: `[res_blocks,
/// channels, stride_freq, stride_time, expansion]`.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub struct Zonos3MectBlockSpec {
    /// Its residual blocks.
    pub res_blocks: usize,

    /// Its output channels, `c`.
    pub channels: usize,

    /// The frequency stride of its first residual block: its frequencies
    /// are its input's over this.
    pub stride_freq: usize,

    /// The time stride of its first residual block; its transformer runs at
    /// this rate, and its output is repeated back to the input's.
    pub stride_time: usize,

    /// The residual blocks' channels over `c`; above 1, a reduce
    /// convolution brings them back.
    pub expansion: usize,
}

/// One block's geometry, from its stage's walk: what it reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zonos3MectBlockGeometry {
    /// The spec.
    pub spec: Zonos3MectBlockSpec,

    /// The channels it reads, `c_in`.
    pub in_channels: usize,

    /// The frequencies it reads, `f_in`.
    pub in_freqs: usize,
}

impl Zonos3MectBlockGeometry {
    /// The residual blocks' channels: `c` times the expansion.
    pub fn res_channels(&self) -> usize {
        self.spec.channels * self.spec.expansion
    }

    /// The frequencies it writes, `f`.
    pub fn freqs(&self) -> usize {
        self.in_freqs / self.spec.stride_freq
    }
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
