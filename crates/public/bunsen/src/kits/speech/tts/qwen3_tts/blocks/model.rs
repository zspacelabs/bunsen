//! [`Qwen3TtsModel`]: the talker and its code predictor.

use burn::{
    config::Config,
    prelude::*,
};

use super::{
    Qwen3TtsCodePredictorConfig,
    Qwen3TtsTalker,
    Qwen3TtsTalkerConfig,
    Qwen3TtsTalkerMeta,
};
use crate::{
    blocks::transformers::decoders::depth::{
        DepthDecoder,
        DepthDecoderMeta,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Config for [`Qwen3TtsModel`].
#[derive(Config, Debug)]
pub struct Qwen3TtsModelConfig {
    /// The talker.
    pub talker: Qwen3TtsTalkerConfig,

    /// The code predictor.
    pub code_predictor: Qwen3TtsCodePredictorConfig,
}

/// A [`Qwen3TtsModel`]'s geometry, which its config and the model answer
/// alike.
pub trait Qwen3TtsModelMeta {
    /// The talker's width, `W`.
    fn d_model(&self) -> usize;

    /// The codebooks a frame holds, `K + 1`: the talker's first and the code
    /// predictor's rest.
    fn code_groups(&self) -> usize;

    /// The codec table's control rows, at its end, which a take never samples
    /// but EOS: the table past the code predictor's codes (the release's
    /// 1024, which upstream's `generate` fixes).
    fn control_rows(&self) -> usize;
}

impl Qwen3TtsModelMeta for Qwen3TtsModelConfig {
    fn d_model(&self) -> usize {
        self.talker.d_model()
    }

    fn code_groups(&self) -> usize {
        self.code_predictor.codebooks() + 1
    }

    fn control_rows(&self) -> usize {
        self.talker
            .codec_vocab()
            .saturating_sub(self.code_predictor.vocab())
    }
}

impl ModuleInit<Qwen3TtsModel> for Qwen3TtsModelConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsModel> {
        Ok(Qwen3TtsModel {
            talker: self.talker.try_init(device)?,
            code_predictor: self.code_predictor.try_init(device)?,
        })
    }
}

/// Qwen3-TTS's model (`Qwen3TTSForConditionalGeneration` without its
/// speaker encoder): the talker and its code predictor.
///
/// A take ([`generate_codes`](Self::generate_codes)) prefills the talker
/// with the prompt, then per frame: the talker's logits choose codebook 0 (EOS
/// ends the take); the code predictor, from the talker's hidden state and
/// codebook 0's embedding, chooses the rest; the frame's embeddings, summed
/// (codebook 0's through the codec table, the rest's through the
/// predictor's tables), plus the frame's trailing text, are the talker's
/// next row.
///
/// Built by [`Qwen3TtsModelConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsModel {
    /// The talker.
    pub talker: Qwen3TtsTalker,

    /// The code predictor.
    pub code_predictor: DepthDecoder,
}

impl Qwen3TtsModelMeta for Qwen3TtsModel {
    fn d_model(&self) -> usize {
        self.talker.d_model()
    }

    fn code_groups(&self) -> usize {
        self.code_predictor.codebooks() + 1
    }

    fn control_rows(&self) -> usize {
        self.talker
            .codec_vocab()
            .saturating_sub(self.code_predictor.vocab())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kits::{
            language::generation::qwen3::Qwen3Config,
            speech::tts::qwen3_tts::blocks::release_text_projection,
        },
        support::testing::cpu_device,
    };

    /// The config and the model built from it answer the meta alike: the
    /// control rows are the codec table's 11 rows past the predictor's 7
    /// codes.
    #[test]
    fn test_model_meta() {
        let config = Qwen3TtsModelConfig::new(
            Qwen3TtsTalkerConfig::new(
                Qwen3Config::new(11, 8, 1, 2, 1, 4, 12),
                13,
                6,
                release_text_projection(6, 8),
                4,
            ),
            Qwen3TtsCodePredictorConfig::new(Qwen3Config::new(7, 4, 1, 2, 1, 2, 6), 8, 3),
        );
        let model: Qwen3TtsModel = config.init(&cpu_device());
        for meta in [&config as &dyn Qwen3TtsModelMeta, &model] {
            assert_eq!(
                [meta.d_model(), meta.code_groups(), meta.control_rows()],
                [8, 4, 4]
            );
        }
    }
}
