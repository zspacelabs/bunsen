//! The Qwen3-ASR audio encoder.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNormConfig,
        Linear,
        LinearConfig,
        PaddingConfig2d,
        conv::Conv2dConfig,
    },
    prelude::{
        Bool,
        Int,
        TensorData,
    },
    tensor::Device,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        conv::{
            ConvBlock2dConfig,
            ConvSeq2d,
            ConvSeq2dConfig,
        },
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
        transformers::encoders::residual::{
            ResidualEncoderAttentionBlock,
            ResidualEncoderAttentionBlockConfig,
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    errors::BunsenResult,
    ops::embedding::sinusoid_table,
};

/// Config for the audio encoder's convolution stem: `convs` strided 2-D
/// convolutions, each then an activation; the defaults are the release's,
/// three of kernel 3, stride 2 and padding 1, each then GELU.
#[derive(Config, Debug)]
pub struct Qwen3AsrStemConfig {
    /// The convolutions.
    #[config(default = 3)]
    pub convs: usize,

    /// Each convolution's kernel, on both axes.
    #[config(default = 3)]
    pub kernel: usize,

    /// Each convolution's stride, on both axes.
    #[config(default = 2)]
    pub stride: usize,

    /// Each convolution's zero padding, on every side.
    #[config(default = 1)]
    pub padding: usize,

    /// The activation after each convolution.
    #[config(default = "BunsenActivationConfig::Gelu")]
    pub activation: BunsenActivationConfig,
}

impl Qwen3AsrStemConfig {
    /// The length `len` frames or bins come out at: each convolution's
    /// `(l + 2 p - k) / s + 1`, `0` for an empty input.
    pub fn out_len(
        &self,
        len: usize,
    ) -> usize {
        (0..self.convs).fold(len, |l, _| {
            if l == 0 {
                0
            } else {
                (l + 2 * self.padding).saturating_sub(self.kernel) / self.stride + 1
            }
        })
    }

    /// The mel frames between two encoder frames within a chunk:
    /// `stride^convs`.
    pub fn frames_per_output(&self) -> usize {
        self.stride.pow(self.convs as u32)
    }
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

    /// The convolution stem's shape.
    fn stem_shape(&self) -> &Qwen3AsrStemConfig;

    /// The encoder frames of one full chunk.
    fn chunk_out(&self) -> usize {
        self.stem_shape().out_len(self.chunk_frames())
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
        (frames / chunk) * self.chunk_out() + self.stem_shape().out_len(frames % chunk)
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

    /// The convolution stem's shape.
    #[config(default = "Qwen3AsrStemConfig::new()")]
    pub stem: Qwen3AsrStemConfig,

    /// Whether the attention's key projection carries a bias, as Whisper's
    /// does not.
    #[config(default = true)]
    pub key_bias: bool,

    /// Whether the stem's projection (`conv_out`) carries a bias.
    #[config(default = false)]
    pub conv_out_bias: bool,

    /// The final normalization's epsilon.
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,

    /// The sinusoidal positions' base.
    #[config(default = 10_000.0)]
    pub sinusoid_base: f64,
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
        self.stem.out_len(self.n_mels)
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

    fn stem_shape(&self) -> &Qwen3AsrStemConfig {
        &self.stem
    }
}

impl ModuleInit<Qwen3AsrAudioEncoder> for Qwen3AsrAudioEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3AsrAudioEncoder> {
        let s = &self.stem;
        let conv = |c_in| {
            ConvBlock2dConfig::new(
                Conv2dConfig::new([c_in, self.conv_channels], [s.kernel, s.kernel])
                    .with_stride([s.stride, s.stride])
                    .with_padding(PaddingConfig2d::Explicit(
                        s.padding, s.padding, s.padding, s.padding,
                    )),
            )
            .with_act(Some(s.activation.clone()))
        };
        let layers = (0..self.n_layers)
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.d_model)
                    .with_d_head(self.d_model / self.n_heads)
                    .with_d_hidden(Some(self.d_ff))
                    .with_key_bias(self.key_bias)
                    .try_init(device)
            })
            .collect::<BunsenResult<_>>()?;
        Ok(Qwen3AsrAudioEncoder {
            stem: ConvSeq2dConfig::new(
                (0..s.convs)
                    .map(|i| conv(if i == 0 { 1 } else { self.conv_channels }))
                    .collect(),
            )
            .try_init(device)?,
            conv_out: LinearConfig::new(self.conv_channels * self.freq_out(), self.d_model)
                .with_bias(self.conv_out_bias)
                .init(device),
            layers,
            ln_post: BunsenNormalizationConfig::from(
                LayerNormConfig::new(self.d_model).with_epsilon(self.layer_norm_eps),
            )
            .init(device),
            n_window: self.n_window,
            n_window_infer: self.n_window_infer,
            stem_shape: self.stem.clone(),
            sinusoid_base: self.sinusoid_base,
        })
    }
}

/// Qwen3-ASR's audio encoder: Whisper's encoder, run over chunks.
///
/// The log-mels are cut into chunks of `2 * n_window` frames; a stem of
/// strided convolutions (the release's three of stride 2, with GELU), a
/// projection (`conv_out`) and sinusoidal positions restarting in every chunk
/// turn each into fewer frames (the release's 13; fewer for a short last
/// chunk). The valid frames of every chunk are
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
    /// The convolutions, each then its activation (the release's three,
    /// kernel 3, stride 2, GELU): `1 -> conv_channels -> … -> conv_channels`.
    pub stem: ConvSeq2d,

    /// `conv_channels * freq_out -> d_model` (the release's without a bias).
    pub conv_out: Linear,

    /// The encoder layers.
    pub layers: Vec<ResidualEncoderAttentionBlock>,

    /// The final normalization.
    pub ln_post: BunsenNormalization,

    /// Half the mel frames of a convolution chunk.
    pub n_window: usize,

    /// The mel frames an attention window spans at inference.
    pub n_window_infer: usize,

    /// The stem's shape, which the frame counts follow.
    #[module(skip)]
    pub stem_shape: Qwen3AsrStemConfig,

    /// The sinusoidal positions' base.
    #[module(skip)]
    pub sinusoid_base: f64,
}

impl Qwen3AsrAudioEncoderMeta for Qwen3AsrAudioEncoder {
    fn d_model(&self) -> usize {
        self.ln_post.num_features()
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

    fn stem_shape(&self) -> &Qwen3AsrStemConfig {
        &self.stem_shape
    }
}

impl crate::burner::module::InputDTypes for Qwen3AsrAudioEncoder {
    /// Its stem's dtype: the dtype its parameters were loaded in.
    fn preferred_dtype(&self) -> burn::tensor::FloatDType {
        crate::burner::module::module_float_dtype(&self.stem)
            .unwrap_or(burn::tensor::FloatDType::F32)
    }
}

/// [`stem`](Qwen3AsrAudioEncoder::stem) casts its features, which a front end
/// makes in f32.
impl crate::burner::module::CoercesInput for Qwen3AsrAudioEncoder {}

impl crate::burner::module::ResultDType for Qwen3AsrAudioEncoder {
    /// Its own: the encoder's features stay in the model's dtype.
    fn result_dtype(
        &self,
        _input: burn::tensor::FloatDType,
    ) -> burn::tensor::FloatDType {
        crate::burner::module::InputDTypes::preferred_dtype(self)
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
        let features = crate::burner::module::InputDTypes::coerce_input(self, features);
        let device = features.device();
        let dtype = features.dtype();
        let chunk = self.chunk_frames();
        let chunks = valid.div_ceil(chunk).max(1);
        // Pad the frame axis to whole chunks; a chunk's padding is kept as the
        // extractor wrote it, as the reference's stem sees it.
        let features = if frames < chunks * chunk {
            Tensor::cat(
                vec![
                    features,
                    Tensor::zeros([n_mels, chunks * chunk - frames], (&device, dtype)),
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
            + sinusoid_table(t, self.d_model(), self.sinusoid_base as f32, &device)
                .cast(dtype)
                .unsqueeze_dim::<3>(0);

        let keep: Vec<i64> = (0..chunks)
            .flat_map(|k| {
                let len = self
                    .stem_shape
                    .out_len(valid.saturating_sub(k * chunk).min(chunk));
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

impl CanonicalKeys for Qwen3AsrAudioEncoder {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("layers.", &self.layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_geometry() {
        let stem = Qwen3AsrStemConfig::new();
        assert_eq!(stem.out_len(100), 13);
        assert_eq!(stem.out_len(50), 7);
        assert_eq!(stem.out_len(1), 1);
        assert_eq!(stem.out_len(0), 0);
        assert_eq!(stem.frames_per_output(), 8);
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
            .with_n_window_infer(150)
            .with_stem(
                Qwen3AsrStemConfig::new()
                    .with_convs(2)
                    .with_kernel(5)
                    .with_padding(2)
                    .with_activation(BunsenActivationConfig::Silu),
            )
            .with_key_bias(false)
            .with_conv_out_bias(true)
            .with_layer_norm_eps(1e-6)
            .with_sinusoid_base(500.0);
        let encoder: Qwen3AsrAudioEncoder = config.init(&cpu_device());
        assert_eq!(
            format!("{:?}", encoder.stem_shape()),
            format!("{:?}", config.stem_shape())
        );
        assert_eq!(encoder.stem.blocks.len(), 2);
        assert!(encoder.conv_out.bias.is_some());
        assert_eq!(encoder.sinusoid_base, 500.0);
        assert_eq!(config.freq_out(), 4, "16 bins through two stride-2 convs");
        assert_eq!(config.chunk_out(), 13, "50 frames through two");
        assert_eq!(config.stem.frames_per_output(), 4);
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
        assert_eq!(encoder.layers[0].mlp.branch.linear1.weight.dims(), [32, 64]);
        let y = encoder.forward(Tensor::zeros([16, 300], &cpu_device()), 250);
        assert_eq!(y.dims(), [33, 32]);
    }
}
