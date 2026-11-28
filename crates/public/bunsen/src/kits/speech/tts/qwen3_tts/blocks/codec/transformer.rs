//! [`Qwen3TtsCodecTransformerConfig`]: the codec decoder's transformer's config.

use burn::config::Config;

use crate::{
    blocks::transformers::decoders::{
        causal::CausalDecoderBlockMeta,
        projected::{
            ProjectedDecoderStackConfig,
            ProjectedDecoderStackMeta,
        },
        stack::UniformStackConfig,
    },
    burner::module::ToStructureConfig,
    errors::BunsenResult,
};

/// Config for the codec decoder's transformer: a
/// [`ProjectedDecoderStack`](crate::blocks::transformers::decoders::projected::ProjectedDecoderStack)
/// from the latent width to the stack's and back, the projections with a
/// bias ([`projected`](Self::projected)).
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecTransformerConfig {
    /// The latent width, `L`: its input's and output's.
    pub latent_width: usize,

    /// The stack; its layers' width is the transformer's
    /// [`d_model`](ProjectedDecoderStackMeta::d_model).
    pub stack: UniformStackConfig,
}

impl Qwen3TtsCodecTransformerConfig {
    /// The transformer as a [`ProjectedDecoderStackConfig`].
    ///
    /// # Errors
    /// As the stack's lowering.
    pub fn projected(&self) -> BunsenResult<ProjectedDecoderStackConfig> {
        Ok(
            ProjectedDecoderStackConfig::new(self.latent_width, self.stack.try_to_structure()?)
                .with_output_width(Some(self.latent_width))
                .with_bias(true),
        )
    }
}

impl ProjectedDecoderStackMeta for Qwen3TtsCodecTransformerConfig {
    fn input_width(&self) -> usize {
        self.latent_width
    }

    fn d_model(&self) -> usize {
        self.stack.layer.d_model()
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers
    }

    fn output_width(&self) -> usize {
        self.latent_width
    }
}
