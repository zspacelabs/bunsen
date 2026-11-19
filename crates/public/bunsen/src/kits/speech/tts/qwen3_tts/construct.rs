//! A loaded Qwen3-TTS model.

use std::path::PathBuf;

use super::{
    Qwen3TtsModel,
    Qwen3TtsModelConfig,
    Qwen3TtsTakeConfig,
    codec::{
        Qwen3TtsCodecDecoder,
        Qwen3TtsCodecDecoderContractConfig,
    },
    prompt::Qwen3TtsPromptConfig,
    sampler::Qwen3TtsGenerationJson,
};

/// A Qwen3-TTS tokenizer's slow files, unread: `transformers` builds the
/// fast tokenizer from them, and so does `load` (feature `tokenizer`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen3TtsTokenizerFiles {
    /// `vocab.json`.
    pub vocab: PathBuf,

    /// `merges.txt`.
    pub merges: PathBuf,

    /// `tokenizer_config.json`, its added tokens.
    pub tokenizer_config: PathBuf,
}

impl Qwen3TtsTokenizerFiles {
    /// The tokenizer
    /// ([`BunsenTokenizer::from_qwen2_files`](crate::kits::support::tokens::BunsenTokenizer::from_qwen2_files)).
    ///
    /// # Errors
    /// As `from_qwen2_files`.
    #[cfg(feature = "tokenizer")]
    pub fn load(
        &self
    ) -> crate::errors::BunsenResult<crate::kits::support::tokens::BunsenTokenizer> {
        crate::kits::support::tokens::BunsenTokenizer::from_qwen2_files(
            &self.vocab,
            &self.merges,
            &self.tokenizer_config,
        )
    }
}

/// What a Qwen3-TTS pretrained loads to: the model (talker and code
/// predictor) and the codec decoder with their configs, the prompt's ids,
/// languages and speakers, the release's take settings, the output's sample
/// rate, and the tokenizer's files.
#[derive(Debug)]
pub struct Qwen3TtsConstruct {
    /// The talker and its code predictor.
    pub model: Qwen3TtsModel,

    /// Its config.
    pub model_config: Qwen3TtsModelConfig,

    /// The 12 Hz codec's decoder.
    pub codec: Qwen3TtsCodecDecoder,

    /// Its config.
    pub codec_config: Qwen3TtsCodecDecoderContractConfig,

    /// The prompt's ids, languages and speakers.
    pub prompt: Qwen3TtsPromptConfig,

    /// The release's take: `generation_config.json`'s settings.
    pub take: Qwen3TtsTakeConfig,

    /// `generation_config.json`, as the release writes it.
    pub generation: Qwen3TtsGenerationJson,

    /// The output's sample rate, in Hz.
    pub sample_rate: u32,

    /// The tokenizer's files.
    pub tokenizer: Qwen3TtsTokenizerFiles,
}
