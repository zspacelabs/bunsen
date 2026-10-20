//! The Qwen3-ASR audio encoder.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNorm,
        LayerNormConfig,
        Linear,
        LinearConfig,
        PaddingConfig2d,
        activation::ActivationConfig,
        conv::Conv2dConfig,
    },
    prelude::{
        Bool,
        Int,
        TensorData,
    },
    tensor::{
        Device,
        activation::gelu,
    },
};

use crate::{
    blocks::{
        conv::{
            ConvBlock2dConfig,
            ConvSeq2d,
            ConvSeq2dConfig,
        },
        transformers::encoders::residual::{
            ResidualEncoderAttentionBlock,
            ResidualEncoderAttentionBlockConfig,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::embedding::sinusoid_table,
};

/// The encoder frames a chunk of `len` mel frames yields: three kernel-3,
/// stride-2 convolutions padded by one, `0` for an empty chunk.
pub fn post_conv_len(len: usize) -> usize {
    (0..3).fold(len, |l, _| if l == 0 { 0 } else { (l - 1) / 2 + 1 })
}

/// Common meta for [`Qwen3AsrAudioEncoder`] and
/// [`Qwen3AsrAudioEncoderConfig`].
pub trait Qwen3AsrAudioEncoderMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The mel frames of one convolution chunk: `2 * n_window`.
    fn chunk_frames(&self) -> usize;

    /// The chunks one attention window spans.
    fn window_chunks(&self) -> usize;

    /// The encoder frames of one full chunk.
    fn chunk_out(&self) -> usize {
        post_conv_len(self.chunk_frames())
    }

    /// The encoder frames of one attention window.
    fn window_frames(&self) -> usize {
        self.chunk_out() * self.window_chunks()
    }

    /// The encoder frames `frames` valid mel frames yield.
    fn output_len(
        &self,
        frames: usize,
    ) -> usize {
        let chunk = self.chunk_frames();
        (frames / chunk) * self.chunk_out() + post_conv_len(frames % chunk)
    }
}

/// Config for [`Qwen3AsrAudioEncoder`]: `transformers`'
/// `Qwen3ASREncoderConfig`.
///
/// Implements [`Qwen3AsrAudioEncoderMeta`].
#[derive(Config, Debug)]
pub struct Qwen3AsrAudioEncoderConfig {
    /// The mel channel count.
    pub n_mels: usize,

    /// The model width.
    pub d_model: usize,

    /// The attention head count.
    pub n_heads: usize,

    /// The feed-forward hidden width; a multiple of `d_model`.
    pub d_ff: usize,

    /// The layer count.
    pub n_layers: usize,

    /// The convolution stem's channel count (`downsample_hidden_size`).
    pub conv_channels: usize,

    /// Half the mel frames of a convolution chunk.
    #[config(default = "50")]
    pub n_window: usize,

    /// The mel frames an attention window spans at inference.
    #[config(default = "800")]
    pub n_window_infer: usize,
}

impl Qwen3AsrAudioEncoderConfig {
    /// `Qwen3-ASR-0.6B`'s: 128 mels, 18 layers of width 896 with 14 heads and
    /// a 3584-wide feed-forward, a 480-channel stem.
    pub fn qwen3_asr_0_6b() -> Self {
        Self::new(128, 896, 14, 3584, 18, 480)
    }

    /// `Qwen3-ASR-1.7B`'s audio tower.
    pub fn qwen3_asr_1_7b() -> Self {
        Self::new(128, 1024, 16, 4096, 24, 480)
    }

    /// The frequency bins left after the stem.
    pub fn freq_out(&self) -> usize {
        post_conv_len(self.n_mels)
    }
}

impl Qwen3AsrAudioEncoderMeta for Qwen3AsrAudioEncoderConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn chunk_frames(&self) -> usize {
        2 * self.n_window
    }

    fn window_chunks(&self) -> usize {
        self.n_window_infer / (2 * self.n_window)
    }
}

impl ModuleInit<Qwen3AsrAudioEncoder> for Qwen3AsrAudioEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3AsrAudioEncoder> {
        let conv = |c_in| {
            ConvBlock2dConfig::new(
                Conv2dConfig::new([c_in, self.conv_channels], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
            )
            .with_act(Some(ActivationConfig::Gelu))
        };
        let layers = (0..self.n_layers)
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.d_model)
                    .with_d_head(self.d_model / self.n_heads)
                    .with_expansion_factor(self.d_ff / self.d_model)
                    .with_key_bias(true)
                    .try_init(device)
            })
            .collect::<BunsenResult<_>>()?;
        Ok(Qwen3AsrAudioEncoder {
            stem: ConvSeq2dConfig::new(vec![
                conv(1),
                conv(self.conv_channels),
                conv(self.conv_channels),
            ])
            .try_init(device)?,
            conv_out: LinearConfig::new(self.conv_channels * self.freq_out(), self.d_model)
                .with_bias(false)
                .init(device),
            layers,
            ln_post: LayerNormConfig::new(self.d_model).init(device),
            n_window: self.n_window,
            n_window_infer: self.n_window_infer,
        })
    }
}

/// Qwen3-ASR's audio encoder: Whisper's encoder, run over chunks.
///
/// The log-mels are cut into chunks of `2 * n_window` frames; a stem of
/// three stride-2 convolutions with GELU, a projection (`conv_out`) and
/// sinusoidal positions restarting in every chunk turn each into 13 frames
/// (fewer for a short last chunk). The valid frames of every chunk are
/// joined into one sequence, and a stack of pre-norm encoder layers attends
/// over it in windows of `n_window_infer / (2 * n_window)` chunks, never
/// across a window's edge. A final normalization closes it.
///
/// The fields carry the names `transformers`' `Qwen3ASREncoder` gives them,
/// but for the convolution stem: its `conv2d1`, `conv2d2` and `conv2d3` are
/// the blocks of [`stem`](Self::stem).
///
/// Built by [`Qwen3AsrAudioEncoderConfig`].
/// Implements [`Qwen3AsrAudioEncoderMeta`].
#[derive(Module, Debug)]
pub struct Qwen3AsrAudioEncoder {
    /// Three convolutions, kernel 3, stride 2, each then GELU:
    /// `1 -> conv_channels -> conv_channels -> conv_channels`.
    pub stem: ConvSeq2d,

    /// `conv_channels * freq_out -> d_model`, no bias.
    pub conv_out: Linear,

    /// The encoder layers.
    pub layers: Vec<ResidualEncoderAttentionBlock>,

    /// The final normalization.
    pub ln_post: LayerNorm,

    /// Half the mel frames of a convolution chunk.
    pub n_window: usize,

    /// The mel frames an attention window spans at inference.
    pub n_window_infer: usize,
}

impl Qwen3AsrAudioEncoderMeta for Qwen3AsrAudioEncoder {
    fn d_model(&self) -> usize {
        self.ln_post.gamma.dims()[0]
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn chunk_frames(&self) -> usize {
        2 * self.n_window
    }

    fn window_chunks(&self) -> usize {
        self.n_window_infer / (2 * self.n_window)
    }
}

impl Qwen3AsrAudioEncoder {
    /// The chunked convolution stem, positions added, valid frames kept.
    ///
    /// # Arguments
    /// * `features` - `[n_mels, frames]` log-mels of one utterance; frames past
    ///   `valid` are the extractor's padding.
    /// * `valid` - the utterance's valid mel frames.
    ///
    /// # Returns
    /// `[output_len(valid), d_model]`.
    pub fn stem(
        &self,
        features: Tensor<2>,
        valid: usize,
    ) -> Tensor<2> {
        let [n_mels, frames] = crate::contracts::unpack_shape_contract!(
            ["n_mels", "frames"],
            &features.dims(),
            &["n_mels", "frames"]
        );
        let device = features.device();
        let chunk = self.chunk_frames();
        let chunks = valid.div_ceil(chunk).max(1);
        // Pad the frame axis to whole chunks; a chunk's padding is kept as the
        // extractor wrote it, as the reference's stem sees it.
        let features = if frames < chunks * chunk {
            Tensor::cat(
                vec![
                    features,
                    Tensor::zeros([n_mels, chunks * chunk - frames], &device),
                ],
                1,
            )
        } else {
            features.slice_dim(1, 0..chunks * chunk)
        };
        let x = features
            .reshape([n_mels, chunks, chunk])
            .swap_dims(0, 1)
            .unsqueeze_dim::<4>(1);
        let x = self.stem.forward(x);
        let [_, c, f, t] = x.dims();
        let x = x.permute([0, 3, 1, 2]).reshape([chunks, t, c * f]);
        let x = self.conv_out.forward(x)
            + sinusoid_table(t, self.d_model(), 10000.0, &device).unsqueeze_dim::<3>(0);

        let keep: Vec<i64> = (0..chunks)
            .flat_map(|k| {
                let len = post_conv_len(valid.saturating_sub(k * chunk).min(chunk));
                (0..len).map(move |j| (k * t + j) as i64)
            })
            .collect();
        let n = keep.len();
        let y = x.reshape([chunks * t, self.d_model()]).select(
            0,
            Tensor::<1, Int>::from_data(TensorData::new(keep, [n]), &device),
        );
        crate::contracts::assert_shape_contract_periodically!(
            ["n", "d_model"],
            &y.dims(),
            &[("n", self.output_len(valid)), ("d_model", self.d_model())]
        );
        y
    }

    /// `[1, n, n]`: true where frames `i` and `j` are in different attention
    /// windows; [`window_block_mask`] over
    /// [`window_frames`](Qwen3AsrAudioEncoderMeta::window_frames).
    ///
    /// [`window_block_mask`]: crate::ops::transformers::attention::window_block_mask
    pub fn window_mask(
        &self,
        n: usize,
        device: &Device,
    ) -> Tensor<3, Bool> {
        crate::ops::transformers::attention::window_block_mask(n, self.window_frames(), device)
            .unsqueeze_dim(0)
    }

    /// Encodes one utterance.
    ///
    /// # Arguments
    /// * `features` - `[n_mels, frames]` log-mels.
    /// * `valid` - the utterance's valid mel frames.
    ///
    /// # Returns
    /// `[output_len(valid), d_model]`.
    pub fn forward(
        &self,
        features: Tensor<2>,
        valid: usize,
    ) -> Tensor<2> {
        let x = self.stem(features, valid);
        let [n, d] = x.dims();
        let mask = self.window_mask(n, &x.device());
        let mut x = x.unsqueeze_dim::<3>(0);
        for layer in &self.layers {
            x = layer.forward_masked(x, Some(mask.clone()));
        }
        self.ln_post.forward(x).reshape([n, d])
    }
}

/// Common meta for [`Qwen3AsrProjector`] and [`Qwen3AsrProjectorConfig`].
pub trait Qwen3AsrProjectorMeta {
    /// The audio encoder's width.
    fn d_audio(&self) -> usize;

    /// The text model's width.
    fn d_text(&self) -> usize;
}

/// Config for [`Qwen3AsrProjector`].
///
/// Implements [`Qwen3AsrProjectorMeta`].
#[derive(Config, Debug)]
pub struct Qwen3AsrProjectorConfig {
    /// The audio encoder's width.
    pub d_audio: usize,

    /// The text model's width.
    pub d_text: usize,
}

impl Qwen3AsrProjectorMeta for Qwen3AsrProjectorConfig {
    fn d_audio(&self) -> usize {
        self.d_audio
    }

    fn d_text(&self) -> usize {
        self.d_text
    }
}

impl ModuleInit<Qwen3AsrProjector> for Qwen3AsrProjectorConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3AsrProjector> {
        Ok(Qwen3AsrProjector {
            linear_1: LinearConfig::new(self.d_audio, self.d_audio).init(device),
            linear_2: LinearConfig::new(self.d_audio, self.d_text).init(device),
        })
    }
}

/// Audio frames into the text model's embedding space: linear, GELU, linear
/// (`transformers`' `multi_modal_projector`).
///
/// Built by [`Qwen3AsrProjectorConfig`].
/// Implements [`Qwen3AsrProjectorMeta`].
#[derive(Module, Debug)]
pub struct Qwen3AsrProjector {
    /// `d_audio -> d_audio`.
    pub linear_1: Linear,

    /// `d_audio -> d_text`.
    pub linear_2: Linear,
}

impl Qwen3AsrProjectorMeta for Qwen3AsrProjector {
    fn d_audio(&self) -> usize {
        self.linear_1.weight.dims()[0]
    }

    fn d_text(&self) -> usize {
        self.linear_2.weight.dims()[1]
    }
}

impl Qwen3AsrProjector {
    /// Projects audio frames.
    ///
    /// # Arguments
    /// * `x` - `[n, d_audio]`.
    ///
    /// # Returns
    /// `[n, d_text]`.
    pub fn forward(
        &self,
        x: Tensor<2>,
    ) -> Tensor<2> {
        let [n] = crate::contracts::unpack_shape_contract!(
            ["n", "d_audio"],
            &x.dims(),
            &["n"],
            &[("d_audio", self.d_audio())]
        );
        let y = self.linear_2.forward(gelu(self.linear_1.forward(x)));
        crate::contracts::assert_shape_contract_periodically!(
            ["n", "d_text"],
            &y.dims(),
            &[("n", n), ("d_text", self.d_text())]
        );
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_geometry() {
        assert_eq!(post_conv_len(100), 13);
        assert_eq!(post_conv_len(50), 7);
        assert_eq!(post_conv_len(1), 1);
        assert_eq!(post_conv_len(0), 0);
        let config = Qwen3AsrAudioEncoderConfig::qwen3_asr_0_6b();
        assert_eq!(config.freq_out(), 16);
        assert_eq!(config.chunk_out(), 13);
        assert_eq!(config.window_frames(), 104);
        assert_eq!(config.output_len(3000), 390);
        assert_eq!(config.output_len(250), 33);
    }

    /// A config non-default in every field, and the encoder built from it,
    /// answer the meta alike.
    #[test]
    fn test_meta_and_shapes() {
        let config = Qwen3AsrAudioEncoderConfig::new(16, 32, 4, 64, 2, 8)
            .with_n_window(25)
            .with_n_window_infer(150);
        let encoder: Qwen3AsrAudioEncoder = config.init(&cpu_device());
        assert_eq!(encoder.d_model(), config.d_model());
        assert_eq!(encoder.n_layers(), config.n_layers());
        assert_eq!(encoder.chunk_frames(), config.chunk_frames());
        assert_eq!(encoder.window_chunks(), config.window_chunks());
        assert_eq!(encoder.window_frames(), config.window_frames());
        assert_eq!(config.chunk_frames(), 50);
        assert_eq!(config.window_chunks(), 3);
        let y = encoder.forward(Tensor::zeros([16, 300], &cpu_device()), 250);
        assert_eq!(y.dims(), [config.output_len(250), 32]);

        let config = Qwen3AsrAudioEncoderConfig::new(16, 32, 4, 64, 2, 8).with_n_window_infer(200);
        let encoder: Qwen3AsrAudioEncoder = config.init(&cpu_device());
        assert_eq!(encoder.layers[0].mlp.linear1.weight.dims(), [32, 64]);
        let y = encoder.forward(Tensor::zeros([16, 300], &cpu_device()), 250);
        assert_eq!(y.dims(), [33, 32]);

        let projector_config = Qwen3AsrProjectorConfig::new(32, 24);
        let projector: Qwen3AsrProjector = projector_config.init(&cpu_device());
        assert_eq!(projector.d_audio(), projector_config.d_audio());
        assert_eq!(projector.d_text(), projector_config.d_text());
        assert_eq!(projector.forward(y).dims(), [33, 24]);
    }
}
