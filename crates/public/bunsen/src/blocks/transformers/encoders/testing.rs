//! Encoder stacks to test with.

use crate::blocks::transformers::decoders::stack::DecoderStackStructureConfig;

/// A pre-norm rotary encoder stack at toy widths: `n_layers` bidirectional
/// layers of `heads` heads over `d_model`, a `LayerNorm` before each
/// sublayer, over the input and after the last layer. The stack an encoder
/// transformer's tests run on.
pub fn rope_stack(
    d_model: usize,
    heads: usize,
    n_layers: usize,
) -> DecoderStackStructureConfig {
    use crate::blocks::{
        activation::BunsenActivationConfig,
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalizationConfig,
        },
        transformers::{
            attention::csa::{
                AttentionMask,
                CausalSelfAttentionConfig,
            },
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::UniformStackConfig,
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
            },
        },
    };
    let norm = BunsenNormalizationConfig::Layer(BunsenLayerNormConfig::new(0));
    crate::burner::module::ToStructureConfig::to_structure(
        &UniformStackConfig::new(
            n_layers,
            CausalDecoderBlockConfig::new(
                CausalSelfAttentionConfig::new(heads, heads, d_model)
                    .with_norm(None)
                    .with_output_bias(true)
                    .with_mask(AttentionMask::Bidirectional)
                    .into(),
                crate::blocks::mlp::BunsenMlpConfig::expansion(d_model, 4)
                    .with_act(BunsenActivationConfig::Gelu)
                    .with_bias(true)
                    .into(),
            )
            .with_norm(norm.clone()),
            norm.clone(),
            RotaryEmbeddingConfig::new(0, d_model / heads).with_direction(RotaryDirection::Forward),
        )
        .with_input_norm(Some(norm)),
    )
}
