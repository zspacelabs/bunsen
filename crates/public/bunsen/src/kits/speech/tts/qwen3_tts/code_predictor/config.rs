//! The code predictor's config.

use burn::config::Config;

use crate::{
    blocks::transformers::decoders::depth::{
        DepthDecoderConfig,
        DepthDecoderMeta,
    },
    burner::module::ToStructureConfig,
    errors::BunsenResult,
    kits::language::qwen3::Qwen3Config,
};

/// Config for Qwen3-TTS's code predictor: a Qwen3 stack (QK-norm before the
/// rotary, a `SwiGLU`, RMS norms) predicting a frame's codebooks after the
/// first, each of [`lm`](Self::lm)'s `n_vocab` codes; its tables at the
/// talker's width, and a projection into its own when the widths differ
/// (1.7B's `small_to_mtp_projection`).
///
/// Lowers to the [`DepthDecoderConfig`] that builds the predictor
/// ([`ToStructureConfig`]).
#[derive(Config, Debug)]
pub struct Qwen3TtsCodePredictorConfig {
    /// The stack's geometry; its `n_vocab` is the codes per codebook. Only
    /// the stack is built from it: the tables and heads are the depth
    /// decoder's own.
    pub lm: Qwen3Config,

    /// The talker's width: the rows the predictor is given, and its tables'.
    pub talker_width: usize,

    /// The codebooks it predicts: a frame's after the first.
    pub codebooks: usize,
}

impl DepthDecoderMeta for Qwen3TtsCodePredictorConfig {
    fn input_width(&self) -> usize {
        self.talker_width
    }

    fn width(&self) -> usize {
        self.lm.d_model
    }

    fn codebooks(&self) -> usize {
        self.codebooks
    }

    fn vocab(&self) -> usize {
        self.lm.n_vocab
    }
}

impl ToStructureConfig for Qwen3TtsCodePredictorConfig {
    type Structure = DepthDecoderConfig;

    fn try_to_structure(&self) -> BunsenResult<DepthDecoderConfig> {
        Ok(DepthDecoderConfig::new(
            self.talker_width,
            self.codebooks,
            self.lm.n_vocab,
            self.lm.stack(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::decoders::depth::DepthDecoder,
        burner::module::ModuleInit,
        support::testing::cpu_device,
    };

    /// The policy, its lowering and the predictor built from either answer
    /// the meta alike, with a projection when the widths differ.
    #[test]
    fn test_lowering_agrees() {
        let device = cpu_device();
        let config =
            Qwen3TtsCodePredictorConfig::new(Qwen3Config::new(11, 8, 2, 2, 1, 4, 16), 12, 3);
        let structure = config.to_structure();
        let direct: DepthDecoder = config.init(&device);
        let lowered: DepthDecoder = structure.init(&device);
        for meta in [
            &config as &dyn DepthDecoderMeta,
            &structure,
            &direct,
            &lowered,
        ] {
            assert_eq!(
                [
                    meta.input_width(),
                    meta.width(),
                    meta.codebooks(),
                    meta.vocab()
                ],
                [12, 8, 3, 11]
            );
        }
        assert_eq!(structure.stack.n_layers, 2);
        assert!(direct.input_proj.is_some() && lowered.input_proj.is_some());
    }
}
