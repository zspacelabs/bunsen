//! `EncoderTransformer`: any of the encoder stacks, chosen by config.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        decoders::stack::{
            DecoderStack,
            DecoderStackMeta,
            DecoderStackStructureConfig,
        },
        encoders::{
            conformer::{
                ConformerTransformer,
                ConformerTransformerConfig,
                ConformerTransformerMeta,
            },
            residual::{
                ResidualEncoderTransformer,
                ResidualEncoderTransformerConfig,
                ResidualEncoderTransformerMeta,
            },
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    errors::BunsenResult,
};

/// Common meta for [`EncoderTransformer`] and [`EncoderTransformerConfig`].
pub trait EncoderTransformerMeta {
    /// The width it reads and writes.
    fn d_model(&self) -> usize;

    /// The number of layers.
    fn n_layers(&self) -> usize;
}

/// Config for [`EncoderTransformer`]: which stack, and its config.
///
/// Implements [`EncoderTransformerMeta`].
#[derive(Config, Debug)]
pub enum EncoderTransformerConfig {
    /// [`ResidualEncoderTransformer`]: residual blocks, pre- or post-norm,
    /// without position embeddings.
    Residual(ResidualEncoderTransformerConfig),

    /// [`DecoderStack`]: its blocks' attention bidirectional, its rotary
    /// table over the whole input (`Nemotron-3-Diarization`'s pre-norm rotary
    /// encoder, an input norm before the layers).
    Stack(DecoderStackStructureConfig),

    /// [`ConformerTransformer`]: Conformer blocks over one relative-position
    /// table.
    Conformer(ConformerTransformerConfig),
}

impl From<ResidualEncoderTransformerConfig> for EncoderTransformerConfig {
    fn from(config: ResidualEncoderTransformerConfig) -> Self {
        Self::Residual(config)
    }
}

impl From<DecoderStackStructureConfig> for EncoderTransformerConfig {
    fn from(config: DecoderStackStructureConfig) -> Self {
        Self::Stack(config)
    }
}

impl From<ConformerTransformerConfig> for EncoderTransformerConfig {
    fn from(config: ConformerTransformerConfig) -> Self {
        Self::Conformer(config)
    }
}

impl EncoderTransformerMeta for EncoderTransformerConfig {
    fn d_model(&self) -> usize {
        match self {
            Self::Residual(c) => c.d_model(),
            Self::Stack(c) => c.d_model(),
            Self::Conformer(c) => c.d_model(),
        }
    }

    fn n_layers(&self) -> usize {
        match self {
            Self::Residual(c) => c.n_layers(),
            Self::Stack(c) => c.n_layers(),
            Self::Conformer(c) => c.n_layers(),
        }
    }
}

impl ModuleInit<EncoderTransformer> for EncoderTransformerConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<EncoderTransformer> {
        Ok(match self {
            Self::Residual(c) => EncoderTransformer::Residual(c.try_init(device)?),
            Self::Stack(c) => EncoderTransformer::Stack(c.try_init(device)?),
            Self::Conformer(c) => EncoderTransformer::Conformer(c.try_init(device)?),
        })
    }
}

/// A bidirectional encoder stack, whichever its config chose: the generic
/// encoder a model holds when the stack is a choice, as
/// [`BunsenNormalization`](crate::blocks::norm::BunsenNormalization) is for a
/// norm.
///
/// Every variant keeps the frame rate and shares one contract,
/// [`forward`](Self::forward): `[batch, frames, d_model]` in and out, each
/// item's padding masked by its length. Each owns its position scheme:
/// none, rotary, or relative. A front end that changes the frame rate runs
/// before.
///
/// The variant is not part of its tensors' paths for the loaders bunsen
/// uses (they skip enum variants, as `transformers` and `NeMo` checkpoints
/// do), so a model's keys are the chosen stack's.
///
/// Built by [`EncoderTransformerConfig`].
/// Implements [`EncoderTransformerMeta`].
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum EncoderTransformer {
    /// [`ResidualEncoderTransformer`].
    Residual(ResidualEncoderTransformer),

    /// [`DecoderStack`], bidirectional.
    Stack(DecoderStack),

    /// [`ConformerTransformer`].
    Conformer(ConformerTransformer),
}

impl From<ResidualEncoderTransformer> for EncoderTransformer {
    fn from(module: ResidualEncoderTransformer) -> Self {
        Self::Residual(module)
    }
}

impl From<DecoderStack> for EncoderTransformer {
    fn from(module: DecoderStack) -> Self {
        Self::Stack(module)
    }
}

impl From<ConformerTransformer> for EncoderTransformer {
    fn from(module: ConformerTransformer) -> Self {
        Self::Conformer(module)
    }
}

impl EncoderTransformerMeta for EncoderTransformer {
    fn d_model(&self) -> usize {
        match self {
            Self::Residual(m) => m.d_model(),
            Self::Stack(m) => m.d_model(),
            Self::Conformer(m) => m.d_model(),
        }
    }

    fn n_layers(&self) -> usize {
        match self {
            Self::Residual(m) => m.n_layers(),
            Self::Stack(m) => m.n_layers(),
            Self::Conformer(m) => m.n_layers(),
        }
    }
}

impl EncoderTransformer {
    /// Runs the stack.
    ///
    /// # Arguments
    /// * `x` - `[batch, frames, d_model]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `frames`.
    ///
    /// # Returns
    /// `[batch, frames, d_model]`. Frames past an item's length hold values
    /// that mean nothing.
    pub fn forward(
        &self,
        x: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        match self {
            Self::Residual(m) => m.forward(x, lengths),
            Self::Stack(m) => m.encoder_forward(x, lengths),
            Self::Conformer(m) => m.forward(x, lengths),
        }
    }
}

impl CanonicalKeys for EncoderTransformer {
    fn canonical_rekey(&self) -> Rekey {
        // The variant is not in the paths.
        match self {
            Self::Residual(t) => t.canonical_rekey(),
            Self::Stack(t) => t.canonical_rekey(),
            Self::Conformer(t) => t.canonical_rekey(),
        }
    }
}

/// A pre-norm rotary encoder stack, for tests: `n_layers` bidirectional
/// layers of `heads` heads, a `LayerNorm` before each sublayer, over the
/// input and after the last layer.
#[cfg(test)]
pub(crate) fn test_rope_stack(
    d_model: usize,
    heads: usize,
    n_layers: usize,
) -> DecoderStackStructureConfig {
    use burn::nn::LayerNormConfig;

    use crate::blocks::{
        activation::BunsenActivationConfig,
        norm::BunsenNormalizationConfig,
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
            mlp::TransformerMlpConfig,
        },
    };
    let norm = BunsenNormalizationConfig::Layer(LayerNormConfig::new(0));
    crate::burner::module::ToStructureConfig::to_structure(
        &UniformStackConfig::new(
            n_layers,
            CausalDecoderBlockConfig::new(
                CausalSelfAttentionConfig::new(heads, heads, d_model)
                    .with_norm(None)
                    .with_output_bias(true)
                    .with_mask(AttentionMask::Bidirectional)
                    .into(),
                TransformerMlpConfig::new(d_model)
                    .with_activation(BunsenActivationConfig::Gelu)
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

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::{
        blocks::transformers::encoders::{
            conformer::ConformerBlockConfig,
            residual::ResidualEncoderAttentionBlockConfig,
        },
        support::testing::cpu_device,
    };

    fn configs() -> Vec<EncoderTransformerConfig> {
        vec![
            ResidualEncoderTransformerConfig::new(
                ResidualEncoderAttentionBlockConfig::new(8).with_d_head(4),
                2,
            )
            .into(),
            test_rope_stack(8, 2, 2).into(),
            ConformerTransformerConfig::new(ConformerBlockConfig::new(8, 2, 16, 3), 2).into(),
        ]
    }

    /// Each variant is its stack: the same meta, and the same output as the
    /// stack run directly, padding masked alike.
    #[test]
    #[serial_test::serial]
    fn test_each_variant_is_its_stack() {
        let device = cpu_device();
        for config in configs() {
            device.seed(3);
            let encoder: EncoderTransformer = config.init(&device);
            assert_eq!(encoder.d_model(), config.d_model());
            assert_eq!(encoder.n_layers(), 2);

            let x = Tensor::<3>::random([2, 6, 8], Distribution::Default, &device);
            let direct = match &encoder {
                EncoderTransformer::Residual(m) => m.forward(x.clone(), Some(&[6, 4])),
                EncoderTransformer::Stack(m) => m.encoder_forward(x.clone(), Some(&[6, 4])),
                EncoderTransformer::Conformer(m) => m.forward(x.clone(), Some(&[6, 4])),
            };
            encoder
                .forward(x, Some(&[6, 4]))
                .to_data()
                .assert_approx_eq::<f32>(&direct.to_data(), Tolerance::absolute(1e-6));
        }
    }

    /// With variants skipped, as bunsen's loaders skip them, a variant's
    /// tensor paths are its stack's; a native record would name the variant.
    #[test]
    #[serial_test::serial]
    #[cfg(feature = "store")]
    fn test_the_variant_is_not_in_the_loaded_paths() {
        use burn::store::ModuleSnapshot;

        let device = cpu_device();
        for config in configs() {
            let encoder: EncoderTransformer = config.init(&device);
            let inner: Vec<String> = match &encoder {
                EncoderTransformer::Residual(m) => m.collect(None, None, true),
                EncoderTransformer::Stack(m) => m.collect(None, None, true),
                EncoderTransformer::Conformer(m) => m.collect(None, None, true),
            }
            .into_iter()
            .map(|t| t.name)
            .collect();
            let skipped: Vec<String> = encoder
                .collect(None, None, true)
                .into_iter()
                .map(|t| t.name)
                .collect();
            assert_eq!(skipped, inner);
            assert!(!inner.is_empty());

            let native: Vec<String> = encoder
                .collect(None, None, false)
                .into_iter()
                .map(|t| t.name)
                .collect();
            assert_ne!(native, inner, "a native record names the variant");
        }
    }
}
