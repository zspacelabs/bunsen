//! The Voxtral Realtime audio encoder.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        PaddingConfig1d,
        activation::ActivationConfig,
        conv::Conv1dConfig,
        norm::{
            NormalizationConfig,
            RmsNorm,
            RmsNormConfig,
        },
    },
    tensor::Device,
};

use crate::{
    blocks::{
        conv::{
            ConvBlock1dConfig,
            ConvSeq1d,
            ConvSeq1dConfig,
        },
        transformers::{
            attention::csa::CausalSelfAttentionConfig,
            decoders::causal::{
                CausalDecoderBlock,
                CausalDecoderBlockConfig,
            },
            embedding::{
                RotaryDirection,
                RotaryEmbedding,
                RotaryEmbeddingConfig,
            },
            mlp::GatedMlpConfig,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
};

/// Mel frames per encoder frame: the stem's second convolution has stride 2.
pub const VOXTRAL_STEM_STRIDE: usize = 2;

/// Common meta for [`VoxtralAudioEncoder`] and [`VoxtralAudioEncoderConfig`].
pub trait VoxtralAudioEncoderMeta {
    /// The mel channel count.
    fn n_mels(&self) -> usize;

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The attention head count (as many key/value heads).
    fn n_heads(&self) -> usize;

    /// The head width.
    fn head_dim(&self) -> usize;

    /// The attention window, in encoder frames, if any.
    fn sliding_window(&self) -> Option<usize>;

    /// The encoder frames `frames` mel frames yield: `ceil(frames / 2)`.
    fn output_len(
        &self,
        frames: usize,
    ) -> usize {
        frames.div_ceil(VOXTRAL_STEM_STRIDE)
    }
}

/// The positions an encoder cache allocates up front by default: 30 s of
/// encoder frames.
pub const VOXTRAL_AUDIO_KV_CACHE_PRE_SIZE: usize = 1536;

fn kv_cache_config(meta: &impl VoxtralAudioEncoderMeta) -> KVCacheConfig {
    KVCacheConfig::new(
        1,
        meta.n_heads(),
        VOXTRAL_AUDIO_KV_CACHE_PRE_SIZE,
        meta.head_dim(),
        meta.n_layers(),
    )
}

impl KVCacheModel for VoxtralAudioEncoderConfig {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

impl KVCacheModel for VoxtralAudioEncoder {
    fn kv_cache_config(&self) -> KVCacheConfig {
        kv_cache_config(self)
    }
}

/// Config for [`VoxtralAudioEncoder`]: `transformers`'
/// `VoxtralRealtimeEncoderConfig`.
///
/// Implements [`VoxtralAudioEncoderMeta`].
#[derive(Config, Debug)]
pub struct VoxtralAudioEncoderConfig {
    /// The mel channel count.
    pub n_mels: usize,

    /// The model width.
    pub d_model: usize,

    /// The attention head count (as many key/value heads).
    pub n_heads: usize,

    /// The head width.
    pub head_dim: usize,

    /// The feed-forward hidden width.
    pub d_ff: usize,

    /// The layer count.
    pub n_layers: usize,

    /// The RMS normalization epsilon.
    #[config(default = "1e-5")]
    pub rms_eps: f64,

    /// The rotary embedding's base.
    #[config(default = "1_000_000")]
    pub rope_theta: usize,

    /// The attention window, in encoder frames; `None` attends to the whole
    /// past.
    #[config(default = "Some(750)")]
    pub sliding_window: Option<usize>,
}

impl VoxtralAudioEncoderConfig {
    /// `Voxtral-Mini-4B-Realtime-2602`'s: 128 mels, 32 layers of width 1280
    /// with 32 heads of 64, a 5120-wide feed-forward, a 750-frame (15 s)
    /// window.
    pub fn voxtral_mini_4b_realtime() -> Self {
        Self::new(128, 1280, 32, 64, 5120, 32)
    }

    /// The layer config.
    fn layer(&self) -> CausalDecoderBlockConfig {
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_heads, self.n_heads, self.d_model)
                .with_head_dim(Some(self.head_dim))
                .with_norm(None)
                .with_qkv_bias(true)
                .with_key_bias(Some(false))
                .with_output_bias(true)
                .with_sliding_window(self.sliding_window),
            GatedMlpConfig::new(self.d_model, self.d_ff)
                .with_down_bias(Some(true))
                .into(),
        )
        .with_norm(NormalizationConfig::Rms(
            RmsNormConfig::new(0).with_epsilon(self.rms_eps),
        ))
    }
}

impl VoxtralAudioEncoderMeta for VoxtralAudioEncoderConfig {
    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }
}

impl ModuleInit<VoxtralAudioEncoder> for VoxtralAudioEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<VoxtralAudioEncoder> {
        // Causal: each convolution's whole context is to the left, `kernel -
        // stride` frames of it.
        let conv = |c_in, stride: usize| {
            ConvBlock1dConfig::new(
                Conv1dConfig::new(c_in, self.d_model, 3)
                    .with_stride(stride)
                    .with_padding(PaddingConfig1d::Explicit(3 - stride, 0)),
            )
            .with_act(Some(ActivationConfig::Gelu))
        };
        let layer = self.layer();
        let layers = (0..self.n_layers)
            .map(|i| layer.try_init(i, device))
            .collect::<BunsenResult<_>>()?;
        Ok(VoxtralAudioEncoder {
            stem: ConvSeq1dConfig::new(vec![
                conv(self.n_mels, 1),
                conv(self.d_model, VOXTRAL_STEM_STRIDE),
            ])
            .try_init(device)?,
            layers,
            norm: RmsNormConfig::new(self.d_model)
                .with_epsilon(self.rms_eps)
                .init(device),
            rope_theta: self.rope_theta,
        })
    }
}

/// Voxtral Realtime's audio encoder: a causal convolution stem, then a stack
/// of causal rotary self-attention layers.
///
/// The stem's two convolutions (kernel 3, GELU, the second of stride 2) see
/// only the frames before them, so 100 Hz log-mels become 50 Hz encoder
/// frames that never look ahead. The layers are the generic
/// [`CausalDecoderBlock`]: pre-norm RMS, causal self-attention with rotary
/// embeddings, biased queries, values and output but unbiased keys, no
/// QK-norm, a sliding window; a `SwiGLU` whose down projection alone is
/// biased. A final RMS norm closes it. Because every part is causal, the
/// encoder streams: a caller that runs it chunk by chunk through a
/// [`KVCache`] gets the frames a whole-utterance pass gets.
///
/// `transformers`' names map onto the fields by
/// [`weights::hf_rekey`](crate::kits::speech::voxtral::weights::hf_rekey):
/// its `embedder.conv1` and `conv2` are the blocks of [`stem`](Self::stem).
///
/// Built by [`VoxtralAudioEncoderConfig`].
/// Implements [`VoxtralAudioEncoderMeta`].
#[derive(Module, Debug)]
pub struct VoxtralAudioEncoder {
    /// The causal convolution stem: `n_mels -> d_model -> d_model`.
    pub stem: ConvSeq1d,

    /// The layers.
    pub layers: Vec<CausalDecoderBlock>,

    /// The final normalization.
    pub norm: RmsNorm,

    /// The rotary embedding's base.
    pub rope_theta: usize,
}

impl VoxtralAudioEncoderMeta for VoxtralAudioEncoder {
    fn n_mels(&self) -> usize {
        self.stem.blocks[0].conv.weight.dims()[1]
    }

    fn d_model(&self) -> usize {
        self.norm.gamma.dims()[0]
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_heads(&self) -> usize {
        use crate::blocks::transformers::attention::csa::CausalSelfAttentionMeta;
        self.layers[0].attn.n_head()
    }

    fn head_dim(&self) -> usize {
        self.layers[0].attn.head_dim
    }

    fn sliding_window(&self) -> Option<usize> {
        self.layers[0].attn.sliding_window
    }
}

impl VoxtralAudioEncoder {
    /// The rotary embedding for encoder frames `0..seq_len`.
    pub fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotaryEmbedding {
        RotaryEmbeddingConfig::new(seq_len, self.head_dim())
            .with_base(self.rope_theta)
            .with_direction(RotaryDirection::Forward)
            .init(device)
    }

    /// The convolution stem.
    ///
    /// # Arguments
    /// * `features` - `[batch, n_mels, frames]` log-mels, from the stream's
    ///   first frame.
    ///
    /// # Returns
    /// `[batch, ceil(frames/2), d_model]`.
    pub fn stem(
        &self,
        features: Tensor<3>,
    ) -> Tensor<3> {
        let [batch, frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "n_mels", "frames"],
            &features.dims(),
            &["batch", "frames"],
            &[("n_mels", self.n_mels())]
        );
        let y = self.stem.forward(features).swap_dims(1, 2);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[
                ("batch", batch),
                ("seq", self.output_len(frames)),
                ("d_model", self.d_model())
            ]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (batch, frames);
        y
    }

    /// The layers and the final norm, over stem frames at positions
    /// `start..start + seq`, through the cache when there is one.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq, d_model]` stem frames.
    /// * `r_emb` - a rotary embedding covering `start + seq` positions.
    /// * `kv_cache` - the cache, at position `start`; `None` means `start` is 0
    ///   and nothing is kept.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, normalized.
    pub fn forward_frames(
        &self,
        x: Tensor<3>,
        r_emb: &RotaryEmbedding,
        mut kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &x.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())]
        );
        let start = kv_cache.as_ref().map(|c| c.pos()).unwrap_or(0);
        let r_emb = r_emb.clip_range(start..start + seq);
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(x, &r_emb, &mut kv_cache);
        }
        let y = self.norm.forward(x);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("d_model", self.d_model())]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;
        y
    }

    /// Encodes a whole utterance: the stem, then the layers.
    ///
    /// # Arguments
    /// * `features` - `[batch, n_mels, frames]` log-mels.
    ///
    /// # Returns
    /// `[batch, ceil(frames/2), d_model]`.
    pub fn forward(
        &self,
        features: Tensor<3>,
    ) -> Tensor<3> {
        let x = self.stem(features);
        let r_emb = self.rotary(x.dims()[1], &x.device());
        self.forward_frames(x, &r_emb, None)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    fn toy() -> VoxtralAudioEncoderConfig {
        VoxtralAudioEncoderConfig::new(6, 16, 2, 6, 24, 2)
            .with_rms_eps(1e-6)
            .with_rope_theta(10_000)
            .with_sliding_window(Some(4))
    }

    /// A config non-default in every field, and the encoder built from it,
    /// answer the meta alike.
    #[test]
    fn test_meta() {
        let config = toy();
        let encoder: VoxtralAudioEncoder = config.init(&cpu_device());
        assert_eq!(encoder.n_mels(), config.n_mels());
        assert_eq!(encoder.d_model(), config.d_model());
        assert_eq!(encoder.n_layers(), config.n_layers());
        assert_eq!(encoder.n_heads(), config.n_heads());
        assert_eq!(encoder.head_dim(), config.head_dim());
        assert_eq!(encoder.sliding_window(), config.sliding_window());
        assert_eq!(encoder.rope_theta, config.rope_theta);
        assert_eq!(encoder.norm.epsilon, 1e-6);
        assert_eq!(config.output_len(7), 4);
        assert_eq!(encoder.kv_cache_config().num_layers, 2);

        let preset = VoxtralAudioEncoderConfig::voxtral_mini_4b_realtime();
        assert_eq!(preset.sliding_window(), Some(750));
        assert_eq!(preset.rope_theta, 1_000_000);
    }

    /// The stem halves the frame rate and is causal: a frame's output does
    /// not change when later input does.
    #[test]
    fn test_stem_is_causal() {
        let device = cpu_device();
        let encoder: VoxtralAudioEncoder = toy().init(&device);
        let a = seeded_tensor::<3>(1, [1, 6, 12], Distribution::Default, &device);
        let b = Tensor::cat(
            vec![
                a.clone().slice_dim(2, 0..8),
                seeded_tensor::<3>(2, [1, 6, 4], Distribution::Default, &device),
            ],
            2,
        );
        let ya = encoder.stem(a);
        let yb = encoder.stem(b);
        assert_eq!(ya.dims(), [1, 6, 16]);
        assert_tensors_close(
            &ya.slice_dim(1, 0..4),
            &yb.slice_dim(1, 0..4),
            Tolerance::default(),
        );
    }

    /// Stem frames run through the cache in chunks give the whole pass's
    /// output: the encoder streams.
    #[test]
    fn test_chunked_through_the_cache() {
        let device = cpu_device();
        let encoder: VoxtralAudioEncoder = toy().init(&device);
        let features = seeded_tensor::<3>(3, [1, 6, 20], Distribution::Default, &device);
        let whole = encoder.forward(features.clone());

        let x = encoder.stem(features);
        let r_emb = encoder.rotary(10, &device);
        let mut cache = encoder.kv_cache_config().with_seq_len(10).init();
        let parts: Vec<Tensor<3>> = [0..3, 3..4, 4..10]
            .into_iter()
            .map(|r| encoder.forward_frames(x.clone().slice_dim(1, r), &r_emb, Some(&mut cache)))
            .collect();
        assert_tensors_close(&Tensor::cat(parts, 1), &whole, Tolerance::default());
    }
}
