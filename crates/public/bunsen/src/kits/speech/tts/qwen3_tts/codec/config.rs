//! The codec decoder's config.

use alloc::{
    format,
    string::String,
    vec,
    vec::Vec,
};

use burn::{
    config::Config,
    nn::RmsNormConfig,
};
use serde::Deserialize;

use crate::{
    blocks::{
        norm::BunsenNormalizationConfig,
        transformers::{
            attention::csa::CausalSelfAttentionConfig,
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::UniformStackConfig,
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
            },
            mlp::TransformerGatedMlpConfig,
        },
    },
    burner::tensor::Widening,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The `decoder_config` section of `Qwen/Qwen3-TTS-Tokenizer-12Hz`'s
/// `config.json`, under upstream's names. Fields the decoder does not read
/// are not kept.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3TtsCodecDecoderJson {
    /// Codes per codebook.
    pub codebook_size: usize,
    /// The quantizer's latent width; each codebook's vectors are half of it.
    pub codebook_dim: usize,
    /// The transformer's width.
    pub hidden_size: usize,
    /// The width the pre-conv writes and the upsamplers keep.
    pub latent_dim: usize,
    /// The transformer's rotary base.
    pub rope_theta: f64,
    /// Its attention heads.
    pub num_attention_heads: usize,
    /// Its key/value heads.
    pub num_key_value_heads: usize,
    /// Its head width; `hidden_size / num_attention_heads` when absent.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Whether its attention projections carry biases.
    #[serde(default)]
    pub attention_bias: bool,
    /// Its causal window, in frames, the frame's own included.
    pub sliding_window: usize,
    /// Its feed-forward's hidden width.
    pub intermediate_size: usize,
    /// Its feed-forward's gate activation.
    pub hidden_act: String,
    /// Its RMS norms' epsilon.
    pub rms_norm_eps: f64,
    /// Its layers.
    pub num_hidden_layers: usize,
    /// The codebooks.
    pub num_quantizers: usize,
    /// The vocoder's upsampling factors, its blocks'.
    pub upsample_rates: Vec<usize>,
    /// The upsamplers' factors, before the vocoder.
    pub upsampling_ratios: Vec<usize>,
    /// The vocoder's first width, halved by each block.
    pub decoder_dim: usize,
}

/// Config for [`Qwen3TtsCodecDecoder`](super::Qwen3TtsCodecDecoder): the
/// 12 Hz codec's decoder, codes in, audio out.
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecDecoderConfig {
    /// The codebooks, `K`; the first apart from the rest.
    pub codebooks: usize,

    /// Codes per codebook.
    pub codebook_size: usize,

    /// The quantizer's latent width, the pre-conv's input; each codebook's
    /// vectors are half of it.
    pub codebook_dim: usize,

    /// The width the pre-conv writes and the upsamplers keep.
    pub latent_width: usize,

    /// The transformer's width.
    pub width: usize,

    /// Its layers.
    pub n_layers: usize,

    /// Its attention heads.
    pub n_heads: usize,

    /// Its key/value heads.
    pub n_kv_heads: usize,

    /// Its head width.
    pub head_dim: usize,

    /// Its feed-forward's hidden width.
    pub d_ff: usize,

    /// Its causal window, in frames, the frame's own included.
    pub window: usize,

    /// Its rotary base.
    pub rope_theta: usize,

    /// Its RMS norms' epsilon.
    pub rms_eps: f64,

    /// Whether its attention projections carry biases.
    #[config(default = false)]
    pub attention_bias: bool,

    /// The upsamplers' factors, each a transposed convolution and a
    /// `ConvNeXt` block.
    pub upsampling_ratios: Vec<usize>,

    /// The vocoder's blocks' factors.
    pub upsample_rates: Vec<usize>,

    /// The vocoder's first width, halved by each block.
    pub vocoder_width: usize,
}

impl Qwen3TtsCodecDecoderConfig {
    /// The config of a `decoder_config` section.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// feed-forward gate other than `silu`, a rotary base that is not a whole
    /// number, an odd `codebook_dim`, fewer than two codebooks, heads that do
    /// not group the key/value heads, or a vocoder width that its blocks do
    /// not halve evenly.
    pub fn from_json(json: &Qwen3TtsCodecDecoderJson) -> BunsenResult<Self> {
        let unsupported = |what: String| Err(BunsenError::unsupported(what));
        if json.hidden_act != "silu" {
            return unsupported(format!(
                "a codec feed-forward gate of {:?}; only \"silu\"",
                json.hidden_act
            ));
        }
        if json.rope_theta.fract() != 0.0 || json.rope_theta < 1.0 {
            return unsupported(format!("a codec rotary base of {}", json.rope_theta));
        }
        if !json.codebook_dim.is_multiple_of(2) {
            return unsupported(format!("an odd codebook_dim, {}", json.codebook_dim));
        }
        if json.num_quantizers < 2 {
            return unsupported(format!(
                "{} codebooks; the split quantizer needs at least 2",
                json.num_quantizers
            ));
        }
        if json.num_key_value_heads == 0
            || !json
                .num_attention_heads
                .is_multiple_of(json.num_key_value_heads)
        {
            return unsupported(format!(
                "{} attention heads over {} key/value heads",
                json.num_attention_heads, json.num_key_value_heads
            ));
        }
        let halvings = json.upsample_rates.len() as u32;
        if !json.decoder_dim.is_multiple_of(1 << halvings) {
            return unsupported(format!(
                "a vocoder width of {} over {halvings} halving blocks",
                json.decoder_dim
            ));
        }
        Ok(Self::new(
            json.num_quantizers,
            json.codebook_size,
            json.codebook_dim,
            json.latent_dim,
            json.hidden_size,
            json.num_hidden_layers,
            json.num_attention_heads,
            json.num_key_value_heads,
            json.head_dim
                .unwrap_or(json.hidden_size / json.num_attention_heads),
            json.intermediate_size,
            json.sliding_window,
            json.rope_theta as usize,
            json.rms_norm_eps,
            json.upsampling_ratios.clone(),
            json.upsample_rates.clone(),
            json.decoder_dim,
        )
        .with_attention_bias(json.attention_bias))
    }

    /// Audio samples per frame of codes: every upsampler's and vocoder
    /// block's factor, multiplied.
    pub fn samples_per_frame(&self) -> usize {
        self.upsampling_ratios
            .iter()
            .chain(&self.upsample_rates)
            .product()
    }

    /// Each vocoder block's input and output widths.
    pub fn vocoder_widths(&self) -> Vec<[usize; 2]> {
        (0..self.upsample_rates.len())
            .map(|i| [self.vocoder_width >> i, self.vocoder_width >> (i + 1)])
            .collect()
    }

    /// The vocoder's last width, its output convolution's input.
    pub fn vocoder_out_width(&self) -> usize {
        self.vocoder_width >> self.upsample_rates.len()
    }

    /// The transformer's layers: pre-RMS-norm, attention over a causal sliding
    /// window with half-split rotary and no QK-norm (its softmax in f32, as
    /// `transformers`' eager attention), a `SwiGLU` feed-forward without
    /// biases, a layer scale on each sublayer, and a final RMS norm.
    pub fn stack(&self) -> UniformStackConfig {
        let norm = BunsenNormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(self.rms_eps));
        let layer = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, self.width)
                .with_head_dim(Some(self.head_dim))
                .with_norm(None)
                .with_qkv_bias(self.attention_bias)
                .with_output_bias(self.attention_bias)
                .with_sliding_window(Some(self.window))
                .with_softmax_widening(Widening::F32)
                .into(),
            TransformerGatedMlpConfig::new(self.width, self.d_ff).into(),
        )
        .with_norm(norm.clone())
        .with_layer_scale(true);
        UniformStackConfig::new(
            self.n_layers,
            layer,
            norm,
            RotaryEmbeddingConfig::new(0, self.head_dim)
                .with_base(self.rope_theta)
                .with_direction(RotaryDirection::Forward),
        )
    }
}

/// `Qwen/Qwen3-TTS-Tokenizer-12Hz`'s `decoder_config`, as its `config.json`
/// writes it: the release's.
pub fn qwen3_tts_codec_12hz_decoder_json() -> Qwen3TtsCodecDecoderJson {
    Qwen3TtsCodecDecoderJson {
        codebook_size: 2048,
        codebook_dim: 512,
        hidden_size: 512,
        latent_dim: 1024,
        rope_theta: 10000.0,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        head_dim: Some(64),
        attention_bias: false,
        sliding_window: 72,
        intermediate_size: 1024,
        hidden_act: "silu".into(),
        rms_norm_eps: 1e-5,
        num_hidden_layers: 8,
        num_quantizers: 16,
        upsample_rates: vec![8, 5, 4, 3],
        upsampling_ratios: vec![2, 2],
        decoder_dim: 1536,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The release's decoder: 1920 samples a frame (24 kHz at 12.5 Hz), the
    /// vocoder 1536 wide down to 96.
    #[test]
    fn test_release_config() {
        let config = Qwen3TtsCodecDecoderConfig::from_json(&qwen3_tts_codec_12hz_decoder_json())
            .expect("the release's config");
        assert_eq!(config.samples_per_frame(), 1920);
        assert_eq!(
            config.vocoder_widths(),
            vec![[1536, 768], [768, 384], [384, 192], [192, 96]]
        );
        assert_eq!(config.vocoder_out_width(), 96);
        assert_eq!(
            (config.head_dim, config.window, config.rope_theta),
            (64, 72, 10000)
        );
    }

    #[test]
    fn test_refusals() {
        let release = qwen3_tts_codec_12hz_decoder_json();
        for json in [
            Qwen3TtsCodecDecoderJson {
                hidden_act: "gelu".into(),
                ..release.clone()
            },
            Qwen3TtsCodecDecoderJson {
                codebook_dim: 511,
                ..release.clone()
            },
            Qwen3TtsCodecDecoderJson {
                num_quantizers: 1,
                ..release.clone()
            },
            Qwen3TtsCodecDecoderJson {
                num_key_value_heads: 3,
                ..release.clone()
            },
            Qwen3TtsCodecDecoderJson {
                decoder_dim: 1000,
                ..release.clone()
            },
            Qwen3TtsCodecDecoderJson {
                rope_theta: 1e4 + 0.5,
                ..release
            },
        ] {
            let error = Qwen3TtsCodecDecoderConfig::from_json(&json).unwrap_err();
            assert_eq!(error.kind(), crate::errors::BunsenErrorKind::Unsupported);
        }
    }
}
