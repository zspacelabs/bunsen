//! The 12 Hz codec's `config.json` and its decoder's state-dict names.

use alloc::{
    format,
    string::String,
    vec,
    vec::Vec,
};

use serde::Deserialize;

use crate::{
    blocks::transformers::decoders::causal::CausalDecoderBlock,
    burner::module::Rekey,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::tts::qwen3_tts::blocks::{
            Qwen3TtsCodecDecoderContractConfig,
            Qwen3TtsCodecDecoderMeta,
        },
        support::transformers::interop::llama_layer_keys,
    },
};

/// The codec's `config.json`, as far as the decoder reads it.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3TtsCodecMirror {
    /// The output's sample rate, in Hz.
    pub output_sample_rate: u32,
    /// Audio samples per frame of codes, as the file states it; the decoder's
    /// rates determine it, and the two are checked to agree.
    #[serde(default)]
    pub decode_upsample_rate: Option<usize>,
    /// The decoder's section.
    pub decoder_config: Qwen3TtsCodecDecoderMirror,
}

impl Qwen3TtsCodecMirror {
    /// The [`Qwen3TtsCodecDecoderContractConfig`] the `decoder_config` section
    /// describes ([`Qwen3TtsCodecDecoderMirror::codec_decoder_config`]),
    /// checked against the stated `decode_upsample_rate`.
    ///
    /// # Errors
    /// As [`Qwen3TtsCodecDecoderMirror::codec_decoder_config`];
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when `decode_upsample_rate` is not the product of the decoder's
    /// `upsampling_ratios` and `upsample_rates`.
    pub fn codec_decoder_config(&self) -> BunsenResult<Qwen3TtsCodecDecoderContractConfig> {
        let config = self.decoder_config.codec_decoder_config()?;
        if let Some(rate) = self.decode_upsample_rate
            && rate != config.samples_per_frame()
        {
            return Err(BunsenError::invalid_resource(format!(
                "a codec config whose decode_upsample_rate, {rate}, is not its decoder's \
                 rates' product, {}",
                config.samples_per_frame()
            )));
        }
        Ok(config)
    }
}

/// The `decoder_config` section of `Qwen/Qwen3-TTS-Tokenizer-12Hz`'s
/// `config.json`, under upstream's names. Fields the decoder does not read
/// are not kept.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen3TtsCodecDecoderMirror {
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
    /// The semantic codebooks, the split quantizer's first part. Upstream's
    /// decoder ignores it and splits after one (`n_q_semantic=1`), so any
    /// other value is refused.
    #[serde(default = "one_semantic_quantizer")]
    pub num_semantic_quantizers: usize,
    /// The vocoder's upsampling factors, its blocks'.
    pub upsample_rates: Vec<usize>,
    /// The upsamplers' factors, before the vocoder.
    pub upsampling_ratios: Vec<usize>,
    /// The vocoder's first width, halved by each block.
    pub decoder_dim: usize,
}

/// `num_semantic_quantizers`' default: upstream's `n_q_semantic`.
fn one_semantic_quantizer() -> usize {
    1
}

impl Qwen3TtsCodecDecoderMirror {
    /// The [`Qwen3TtsCodecDecoderContractConfig`] this `decoder_config` section
    /// describes.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// feed-forward gate other than `silu`, an odd `codebook_dim`, other than
    /// one semantic codebook, fewer than two codebooks, heads that do not group
    /// the key/value heads, or a vocoder width that its blocks do not halve
    /// evenly.
    pub fn codec_decoder_config(&self) -> BunsenResult<Qwen3TtsCodecDecoderContractConfig> {
        let unsupported = |what: String| Err(BunsenError::unsupported(what));
        if self.hidden_act != "silu" {
            return unsupported(format!(
                "a codec feed-forward gate of {:?}; only \"silu\"",
                self.hidden_act
            ));
        }
        if !self.codebook_dim.is_multiple_of(2) {
            return unsupported(format!("an odd codebook_dim, {}", self.codebook_dim));
        }
        if self.num_semantic_quantizers != 1 {
            return unsupported(format!(
                "num_semantic_quantizers = {}; upstream's decoder splits after one \
                 (`n_q_semantic`)",
                self.num_semantic_quantizers
            ));
        }
        if self.num_quantizers < 2 {
            return unsupported(format!(
                "{} codebooks; the split quantizer needs at least 2",
                self.num_quantizers
            ));
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return unsupported(format!(
                "{} attention heads over {} key/value heads",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        let halvings = self.upsample_rates.len() as u32;
        if !self.decoder_dim.is_multiple_of(1 << halvings) {
            return unsupported(format!(
                "a vocoder width of {} over {halvings} halving blocks",
                self.decoder_dim
            ));
        }
        Ok(Qwen3TtsCodecDecoderContractConfig::new(
            self.num_quantizers,
            self.codebook_size,
            self.codebook_dim,
            self.latent_dim,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim
                .unwrap_or(self.hidden_size / self.num_attention_heads),
            self.intermediate_size,
            self.sliding_window,
            self.rope_theta,
            self.rms_norm_eps,
            self.upsampling_ratios.clone(),
            self.upsample_rates.clone(),
            self.decoder_dim,
        )
        .with_semantic_codebooks(self.num_semantic_quantizers)
        .with_attention_bias(self.attention_bias))
    }
}

/// `Qwen/Qwen3-TTS-Tokenizer-12Hz`'s `decoder_config`, as its `config.json`
/// writes it: the release's.
pub fn qwen3_tts_codec_12hz_decoder_mirror() -> Qwen3TtsCodecDecoderMirror {
    Qwen3TtsCodecDecoderMirror {
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
        num_semantic_quantizers: 1,
        upsample_rates: vec![8, 5, 4, 3],
        upsampling_ratios: vec![2, 2],
        decoder_dim: 1536,
    }
}

/// The prefix the decoder's tensors sit under in the codec's file.
pub const QWEN3_TTS_CODEC_DECODER_PREFIX: &str = "decoder.";

/// A transformer layer's names onto a decoder block's canonical ones:
/// Llama's (`input_layernorm`, `post_attention_layernorm`, `self_attn`'s
/// `{q,k,v,o}_proj`, the `SwiGLU`'s projections as they are), and the layer
/// scales, `self_attn_layer_scale.scale` and `mlp_layer_scale.scale`.
pub fn codec_decoder_layer_rekey() -> Rekey {
    CausalDecoderBlock::layer_rekey(&llama_layer_keys(
        "input_layernorm",
        "post_attention_layernorm",
    ))
    .exact("self_attn_layer_scale.scale", "attn_layer_scale")
    .exact("mlp_layer_scale.scale", "mlp_layer_scale")
}

/// The decoder's names, without [`QWEN3_TTS_CODEC_DECODER_PREFIX`], onto a
/// [`Qwen3TtsCodecDecoder`](crate::kits::speech::tts::qwen3_tts::blocks::Qwen3TtsCodecDecoder)'s
/// canonical ones.
pub fn codec_decoder_rekey(config: &Qwen3TtsCodecDecoderContractConfig) -> Rekey {
    let codebooks = Rekey::keep_rest().mount_indexed(
        "vq.layers.",
        "codebooks.",
        Rekey::keep_rest().rename("_codebook.", ""),
    );
    let rekey = Rekey::drop_rest()
        .mount(
            "quantizer.rvq_first.",
            "quantizer.rvq_first.",
            codebooks.clone(),
        )
        .mount("quantizer.rvq_rest.", "quantizer.rvq_rest.", codebooks)
        .rename("pre_conv.", "pre_conv.")
        .rename("pre_transformer.input_proj.", "pre_transformer.input_proj.")
        .rename(
            "pre_transformer.output_proj.",
            "pre_transformer.output_proj.",
        )
        .rename("pre_transformer.norm.", "pre_transformer.stack.norm.")
        .mount_indexed(
            "pre_transformer.layers.",
            "pre_transformer.stack.layers.",
            codec_decoder_layer_rekey(),
        )
        .mount_indexed(
            "upsample.",
            "upsample.",
            Rekey::drop_rest()
                .rename("0.", "up.")
                .rename("1.dwconv.conv.", "block.dwconv.")
                .rename("1.", "block."),
        );
    let blocks = config.upsample_rates.len();
    let rekey = (0..blocks).fold(rekey.rename("decoder.0.", "decoder.input."), |rekey, i| {
        let units = (0..config.vocoder_dilations.len()).fold(
            Rekey::drop_rest()
                .rename("0.", "act.")
                .rename("1.", "upsample."),
            |units, u| units.rename(format!("{}.", u + 2), format!("residuals.{u}.")),
        );
        rekey.mount(
            format!("decoder.{}.block.", i + 1),
            format!("decoder.blocks.{i}."),
            units,
        )
    });
    rekey
        .rename(format!("decoder.{}.", blocks + 1), "decoder.output_act.")
        .rename(format!("decoder.{}.", blocks + 2), "decoder.output.")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::signals::DacDecoderMeta,
        burner::module::ToStructureConfig,
        errors::testing::ErrorMatcher,
        kits::speech::tts::qwen3_tts::blocks::Qwen3TtsCodecDecoderMeta,
    };

    /// The release's decoder: 1920 samples a frame (24 kHz at 12.5 Hz), the
    /// vocoder 1536 wide down to 96.
    #[test]
    fn test_release_config() {
        let config = Qwen3TtsCodecDecoderMirror::codec_decoder_config(
            &qwen3_tts_codec_12hz_decoder_mirror(),
        )
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
            (64, 72, 10000.0)
        );
    }

    #[test]
    fn test_refusals() {
        let release = qwen3_tts_codec_12hz_decoder_mirror();
        for json in [
            Qwen3TtsCodecDecoderMirror {
                hidden_act: "gelu".into(),
                ..release.clone()
            },
            Qwen3TtsCodecDecoderMirror {
                codebook_dim: 511,
                ..release.clone()
            },
            Qwen3TtsCodecDecoderMirror {
                num_quantizers: 1,
                ..release.clone()
            },
            Qwen3TtsCodecDecoderMirror {
                num_semantic_quantizers: 2,
                ..release.clone()
            },
            Qwen3TtsCodecDecoderMirror {
                num_key_value_heads: 3,
                ..release.clone()
            },
            Qwen3TtsCodecDecoderMirror {
                decoder_dim: 1000,
                ..release
            },
        ] {
            let error = Qwen3TtsCodecDecoderMirror::codec_decoder_config(&json).unwrap_err();
            ErrorMatcher::unsupported().assert(&error);
        }
    }

    /// A stated `decode_upsample_rate` must be the decoder's rates' product;
    /// an absent one is not checked.
    #[test]
    fn test_decode_upsample_rate_checked() {
        let codec = |rate| Qwen3TtsCodecMirror {
            output_sample_rate: 24_000,
            decode_upsample_rate: rate,
            decoder_config: qwen3_tts_codec_12hz_decoder_mirror(),
        };
        for rate in [Some(1920), None] {
            let config = codec(rate)
                .codec_decoder_config()
                .expect("a consistent config");
            assert_eq!(config.samples_per_frame(), 1920);
        }
        let error = codec(Some(1921)).codec_decoder_config().unwrap_err();
        ErrorMatcher::invalid_resource().assert(&error);
    }
}
