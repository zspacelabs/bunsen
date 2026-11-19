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

use super::{
    Qwen3TtsCodecDecoderMeta,
    Qwen3TtsCodecDecoderStructureConfig,
    Qwen3TtsCodecTransformerConfig,
    Qwen3TtsCodecUpsampleConfig,
    Qwen3TtsVocoderConfig,
};
use crate::{
    blocks::{
        conv::CausalConv1dConfig,
        norm::BunsenNormalizationConfig,
        quantization::SplitResidualVectorQuantizerConfig,
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
    burner::{
        module::ToStructureConfig,
        tensor::Widening,
    },
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

/// The policy of [`Qwen3TtsCodecDecoder`](super::Qwen3TtsCodecDecoder), the
/// 12 Hz codec's decoder (codes in, audio out): its geometry, as a
/// `decoder_config` gives it ([`from_json`](Self::from_json)).
///
/// Lowers to the [`Qwen3TtsCodecDecoderStructureConfig`] that builds the
/// decoder ([`ToStructureConfig`]): the vocoder's widths halving block by
/// block, the transformer's layers from the geometry.
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecDecoderContractConfig {
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

impl Qwen3TtsCodecDecoderContractConfig {
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

impl Qwen3TtsCodecDecoderMeta for Qwen3TtsCodecDecoderContractConfig {
    fn codebooks(&self) -> usize {
        self.codebooks
    }

    fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    fn samples_per_frame(&self) -> usize {
        self.upsampling_ratios
            .iter()
            .chain(&self.upsample_rates)
            .product()
    }
}

impl ToStructureConfig for Qwen3TtsCodecDecoderContractConfig {
    type Structure = Qwen3TtsCodecDecoderStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<Qwen3TtsCodecDecoderStructureConfig> {
        let latent = self.latent_width;
        Ok(Qwen3TtsCodecDecoderStructureConfig::new(
            SplitResidualVectorQuantizerConfig::new(
                self.codebooks,
                self.codebook_size,
                self.codebook_dim / 2,
                self.codebook_dim,
            ),
            CausalConv1dConfig::new([self.codebook_dim, latent], 3),
            Qwen3TtsCodecTransformerConfig::new(latent, self.stack()),
            self.upsampling_ratios
                .iter()
                .map(|&r| Qwen3TtsCodecUpsampleConfig::new(latent, r))
                .collect(),
            Qwen3TtsVocoderConfig::new(latent, self.vocoder_width, self.upsample_rates.clone()),
        ))
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
    use burn::module::Module;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::tts::qwen3_tts::codec::{
            QWEN3_TTS_VOCODER_DILATIONS,
            Qwen3TtsCodecDecoder,
            Qwen3TtsCodecTransformerMeta,
            Qwen3TtsCodecUpsampleMeta,
            Qwen3TtsVocoderBlockConfig,
            Qwen3TtsVocoderBlockMeta,
            Qwen3TtsVocoderMeta,
            Qwen3TtsVocoderResidualUnitConfig,
            Qwen3TtsVocoderResidualUnitMeta,
        },
        support::testing::cpu_device,
    };

    /// A small decoder, every width distinct.
    fn small() -> Qwen3TtsCodecDecoderContractConfig {
        Qwen3TtsCodecDecoderContractConfig::from_json(&Qwen3TtsCodecDecoderJson {
            codebook_size: 9,
            codebook_dim: 4,
            hidden_size: 8,
            latent_dim: 6,
            rope_theta: 10000.0,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(4),
            attention_bias: false,
            sliding_window: 4,
            intermediate_size: 16,
            hidden_act: "silu".into(),
            rms_norm_eps: 1e-5,
            num_hidden_layers: 1,
            num_quantizers: 3,
            upsample_rates: vec![2, 3],
            upsampling_ratios: vec![2],
            decoder_dim: 12,
        })
        .expect("a small config")
    }

    /// The policy, its lowering and the decoders built from either answer
    /// the meta alike, and the decoders have the same parameters.
    #[test]
    fn test_lowering_agrees() {
        let device = cpu_device();
        let policy = small();
        let structure = policy.to_structure();
        let direct: Qwen3TtsCodecDecoder = policy.init(&device);
        let lowered: Qwen3TtsCodecDecoder = structure.init(&device);
        for meta in [
            &policy as &dyn Qwen3TtsCodecDecoderMeta,
            &structure,
            &direct,
            &lowered,
        ] {
            assert_eq!(
                [
                    meta.codebooks(),
                    meta.codebook_size(),
                    meta.samples_per_frame()
                ],
                [3, 9, 12]
            );
        }
        assert_eq!(direct.num_params(), lowered.num_params());
    }

    /// Each part's config and the part built from it answer its meta alike.
    #[test]
    fn test_part_meta() {
        let device = cpu_device();
        let structure = small().to_structure();
        let decoder: Qwen3TtsCodecDecoder = structure.init(&device);

        let t = &structure.pre_transformer;
        for meta in [
            t as &dyn Qwen3TtsCodecTransformerMeta,
            &decoder.pre_transformer,
        ] {
            assert_eq!([meta.latent_width(), meta.width()], [6, 8]);
        }
        let u = &structure.upsample[0];
        for meta in [u as &dyn Qwen3TtsCodecUpsampleMeta, &decoder.upsample[0]] {
            assert_eq!([meta.width(), meta.factor()], [6, 2]);
        }
        let v = &structure.decoder;
        for meta in [v as &dyn Qwen3TtsVocoderMeta, &decoder.decoder] {
            assert_eq!(
                (meta.latent_width(), meta.width(), meta.factors()),
                (6, 12, vec![2, 3])
            );
            assert_eq!([meta.out_width(), meta.factor()], [3, 6]);
        }
        let b = Qwen3TtsVocoderBlockConfig::new([12, 6], 2);
        for meta in [
            &b as &dyn Qwen3TtsVocoderBlockMeta,
            &decoder.decoder.blocks[0],
        ] {
            assert_eq!((meta.channels(), meta.factor()), ([12, 6], 2));
        }
        let r = Qwen3TtsVocoderResidualUnitConfig::new(3, QWEN3_TTS_VOCODER_DILATIONS[2]);
        for meta in [
            &r as &dyn Qwen3TtsVocoderResidualUnitMeta,
            &decoder.decoder.blocks[1].residuals[2],
        ] {
            assert_eq!([meta.channels(), meta.dilation()], [3, 9]);
        }
    }

    /// The release's decoder: 1920 samples a frame (24 kHz at 12.5 Hz), the
    /// vocoder 1536 wide down to 96.
    #[test]
    fn test_release_config() {
        let config =
            Qwen3TtsCodecDecoderContractConfig::from_json(&qwen3_tts_codec_12hz_decoder_json())
                .expect("the release's config");
        assert_eq!(config.samples_per_frame(), 1920);
        let structure = config.to_structure();
        assert_eq!(structure.samples_per_frame(), 1920);
        assert_eq!(
            [structure.decoder.width(), structure.decoder.out_width()],
            [1536, 96]
        );
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
            let error = Qwen3TtsCodecDecoderContractConfig::from_json(&json).unwrap_err();
            assert_eq!(error.kind(), crate::errors::BunsenErrorKind::Unsupported);
        }
    }
}
