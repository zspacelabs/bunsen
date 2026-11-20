//! The language model's config.

use alloc::vec::Vec;

use burn::{
    config::Config,
    nn::norm::RmsNormConfig,
};

use crate::{
    blocks::{
        mlp::BunsenGatedMlpConfig,
        norm::{
            BunsenNormalizationConfig,
            GatedNormConfig,
        },
        transformers::{
            attention::csa::{
                AttentionOutputGate,
                CausalSelfAttentionConfig,
                QkNormPlacement,
            },
            decoders::{
                attn_res::AttnResStackConfig,
                causal::CausalDecoderBlockConfig,
                stack::DecoderStackStructureConfig,
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
                RotaryPairLayout,
            },
            mixers::{
                GatedDeltaNetConfig,
                TokenMixerConfig,
            },
            mlp::TransformerFeedForwardConfig,
            moe::{
                experts::ExpertStackConfig,
                router::{
                    ChoiceBias,
                    ChoiceBiasMode,
                    ChoiceBiasSite,
                    TopKRouterConfig,
                },
                sparse::SparseMoeConfig,
            },
        },
    },
    burner::tensor::Widening,
};

/// Config for [`Zonos3Lm`](super::Zonos3Lm): ZONOS3-Mini's language model,
/// rows of `n_codebooks + 1` ids in, each row's logits over every codebook's
/// codes out.
#[derive(Config, Debug)]
pub struct Zonos3LmConfig {
    /// The model width, `D`.
    pub d_model: usize,

    /// The layers.
    pub n_layers: usize,

    /// Every `global_period`-th layer (`(l + 1) % global_period == 0`)
    /// attends; the others are gated `DeltaNet`s.
    pub global_period: usize,

    /// The audio codebooks.
    pub n_codebooks: usize,

    /// Codes per codebook; each codebook's vocabulary is two more: the end
    /// of audio, then the pad.
    pub codebook_size: usize,

    /// The text table's rows.
    pub text_rows: usize,

    /// The attention's query heads.
    pub n_heads: usize,

    /// The attention's key and value heads.
    pub n_kv_heads: usize,

    /// Each attention head's width.
    pub head_dim: usize,

    /// The rotary base.
    pub rope_base: f64,

    /// The gated `DeltaNet` heads.
    pub gdn_heads: usize,

    /// Each gated `DeltaNet` head's width.
    pub gdn_head_dim: usize,

    /// The gated `DeltaNet`'s convolution kernel.
    pub gdn_kernel: usize,

    /// The first layer's dense `SwiGLU` hidden width.
    pub ffn_hidden: usize,

    /// The experts of every later layer's mixture.
    pub n_experts: usize,

    /// The experts each row is routed to.
    pub top_k: usize,

    /// Each expert's hidden width.
    pub moe_hidden: usize,

    /// The latent the experts run in.
    pub moe_latent: usize,

    /// The speaker embedding's width.
    pub speaker_dim: usize,

    /// Each condition table's buckets, without its unknown bucket (the table
    /// has one more row).
    pub condition_buckets: Vec<usize>,

    /// The longest sequence of rows.
    pub max_rows: usize,

    /// The RMS norms' epsilon: the embedding's, the sublayers', the
    /// router's, the routes' keys and the final norm.
    #[config(default = 1e-5)]
    pub norm_eps: f64,

    /// The attention's weightless QK-norm epsilon.
    #[config(default = 1e-6)]
    pub qk_norm_eps: f64,

    /// The gated `DeltaNet`'s output norm epsilon.
    #[config(default = 1e-5)]
    pub gdn_norm_eps: f64,

    /// The logits' tanh soft-cap.
    #[config(default = 15.0)]
    pub softcap: f64,
}

impl Zonos3LmConfig {
    /// Whether layer `l` attends (else it is a gated `DeltaNet`).
    pub fn layer_attends(
        &self,
        l: usize,
    ) -> bool {
        (l + 1).is_multiple_of(self.global_period)
    }

    /// An RMS norm at the model's epsilon.
    fn rms(&self) -> BunsenNormalizationConfig {
        BunsenNormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(self.norm_eps))
    }

    /// Layer `l`'s block: a global attention or a gated `DeltaNet`, then the
    /// first layer's dense `SwiGLU` or a later layer's latent mixture of
    /// experts; each sublayer behind an RMS norm.
    pub fn layer(
        &self,
        l: usize,
    ) -> CausalDecoderBlockConfig {
        let d = self.d_model;
        let mixer: TokenMixerConfig = if self.layer_attends(l) {
            CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, d)
                .with_head_dim(Some(self.head_dim))
                .with_norm(Some(BunsenNormalizationConfig::Rms(
                    RmsNormConfig::new(0).with_epsilon(self.qk_norm_eps),
                )))
                .with_qk_norm(QkNormPlacement::BeforeRope)
                .with_query_gain(true)
                .with_output_gate(Some(AttentionOutputGate::Headwise))
                .into()
        } else {
            GatedDeltaNetConfig::new(
                d,
                self.gdn_heads,
                self.gdn_heads,
                self.gdn_head_dim,
                self.gdn_head_dim,
            )
            .with_conv_kernel(self.gdn_kernel)
            .with_norm(GatedNormConfig::rms(0, self.gdn_norm_eps))
            .into()
        };
        let ffn: TransformerFeedForwardConfig = if l == 0 {
            BunsenGatedMlpConfig::new(d, self.ffn_hidden).into()
        } else {
            SparseMoeConfig::new(
                TopKRouterConfig::new(d, self.n_experts, self.top_k)
                    .with_choice_bias(Some(ChoiceBias::new(
                        ChoiceBiasSite::Logits,
                        ChoiceBiasMode::Subtract,
                    )))
                    .with_input_norm(Some(self.rms())),
                ExpertStackConfig::new(self.n_experts, self.moe_latent, self.moe_hidden),
            )
            .with_latent(Some(self.moe_latent))
            .into()
        };
        CausalDecoderBlockConfig::new(mixer, ffn).with_norm(self.rms())
    }

    /// The stack: the layers without a residual stream (Full Attention
    /// Residuals), a weightless input norm (the embedding's), the final
    /// norm, and the interleaved rotary table the attention layers share,
    /// which rotates half inputs in f32, as the reference does.
    pub fn stack(&self) -> AttnResStackConfig {
        let layers = (0..self.n_layers).map(|l| self.layer(l)).collect();
        let rotary = RotaryEmbeddingConfig::new(0, self.head_dim)
            .with_base(self.rope_base)
            .with_direction(RotaryDirection::Forward)
            .with_pair_layout(RotaryPairLayout::Interleaved)
            .with_widening(Widening::F32);
        AttnResStackConfig::new(
            DecoderStackStructureConfig::new(layers, self.rms(), rotary)
                .with_input_norm(Some(self.rms()))
                .into(),
        )
        .with_key_eps(self.norm_eps)
    }
}
