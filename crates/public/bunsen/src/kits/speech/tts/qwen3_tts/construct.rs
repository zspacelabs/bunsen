//! A loaded Qwen3-TTS model.

use alloc::collections::BTreeMap;
use std::path::PathBuf;

use super::{
    blocks::{
        Qwen3TtsCodecDecoder,
        Qwen3TtsCodecDecoderContractConfig,
        Qwen3TtsDecodeConfig,
        Qwen3TtsModel,
        Qwen3TtsModelConfig,
    },
    prompt::Qwen3TtsPromptConfig,
};
use crate::kits::{
    speech::tts::qwen3_tts::driver::Qwen3TtsTakeConfig,
    support::text::BunsenLanguageTag,
};

/// Which of the release's model kinds a checkpoint is (`config.json`'s
/// `tts_model_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3TtsModelKind {
    /// `custom_voice`: stock speakers, and on the larger sizes a style
    /// instruction.
    CustomVoice,

    /// `voice_design`: a voice described in words, no stock speakers.
    VoiceDesign,

    /// `base`: a voice cloned from a reference (no stock speakers).
    Base,
}

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
/// predictor) and the codec decoder with their configs and its chunking, the
/// prompt's ids, languages and speakers, the release's take settings, the
/// output's sample rate, and the tokenizer's files.
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

    /// How the codec decodes a take: upstream's chunking.
    pub decode: Qwen3TtsDecodeConfig,

    /// The prompt's ids, languages and speakers.
    pub prompt: Qwen3TtsPromptConfig,

    /// The release's take: `generation_config.json`'s settings.
    pub take: Qwen3TtsTakeConfig,

    /// The output's sample rate, in Hz.
    pub sample_rate: u32,

    /// The tokenizer's files.
    pub tokenizer: Qwen3TtsTokenizerFiles,

    /// The model's kind.
    pub kind: Qwen3TtsModelKind,

    /// Whether it takes a style instruction: its catalog row's
    /// `instruction`, else upstream's rule (a `CustomVoice` model larger than
    /// 0.6B).
    pub instruction: bool,

    /// Each language key's tag: its catalog row's `language_keys`; empty
    /// without a row.
    pub language_tags: BTreeMap<String, BunsenLanguageTag>,
}
