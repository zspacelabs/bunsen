//! Qwen3-ASR's chat prompt, and parsing its reply.
use burn::config::Config;

use crate::{
    errors::BunsenResult,
    kits::tokens::ByteLevelVocabulary,
};

/// `Qwen3-ASR-0.6B`'s prompt before the audio:
/// `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>`.
pub const QWEN3_ASR_PROMPT_PREFIX: [usize; 9] =
    [151_644, 8948, 198, 151_645, 198, 151_644, 872, 198, 151_669];

/// `Qwen3-ASR-0.6B`'s prompt after the audio:
/// `<|audio_end|><|im_end|>\n<|im_start|>assistant\n`.
pub const QWEN3_ASR_PROMPT_SUFFIX: [usize; 6] = [151_670, 151_645, 198, 151_644, 77_091, 198];

/// `Qwen3-ASR-0.6B`'s `<asr_text>`: the reply's header ends, and its
/// transcript begins, here.
pub const QWEN3_ASR_TEXT_TOKEN: usize = 151_704;

/// The chat template a Qwen3-ASR model is prompted with, as token ids: the
/// prompt around an utterance's audio placeholders, and the token splitting
/// the reply's header from its transcript.
///
/// The defaults are `Qwen3-ASR-0.6B`'s (`transformers`' chat template with an
/// empty system turn and the generation prompt), which `FermionResearch`'s
/// Phonon-1 builds share.
#[derive(Config, Debug)]
pub struct Qwen3AsrPromptConfig {
    /// The tokens before the audio placeholders.
    #[config(default = "QWEN3_ASR_PROMPT_PREFIX.to_vec()")]
    pub prefix: Vec<usize>,

    /// The tokens after the audio placeholders, through the assistant turn's
    /// opening.
    #[config(default = "QWEN3_ASR_PROMPT_SUFFIX.to_vec()")]
    pub suffix: Vec<usize>,

    /// The token ending the reply's header (`language English`, say) and
    /// beginning its transcript.
    #[config(default = "QWEN3_ASR_TEXT_TOKEN")]
    pub text_token: usize,
}

/// A parsed reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen3AsrTranscript {
    /// The generated tokens, the end token included when one was generated.
    pub tokens: Vec<usize>,

    /// The language the reply's header names; `None` when it names none
    /// (`language None`, for audio without speech) or has no header.
    pub language: Option<String>,

    /// The transcript.
    pub text: String,
}

impl Qwen3AsrPromptConfig {
    /// The prompt of an utterance whose audio is `n_audio` encoder frames:
    /// the prefix, `n_audio` copies of `audio_token`, the suffix.
    pub fn prompt(
        &self,
        n_audio: usize,
        audio_token: usize,
    ) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.prefix.len() + n_audio + self.suffix.len());
        ids.extend_from_slice(&self.prefix);
        ids.extend(std::iter::repeat_n(audio_token, n_audio));
        ids.extend_from_slice(&self.suffix);
        ids
    }

    /// Parses a reply: `language <name>`, the text token, the transcript.
    ///
    /// The tokens from the first of `eos` on are not text. A reply without
    /// the text token is all transcript, with no language.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if a token is outside `vocab`.
    pub fn parse(
        &self,
        tokens: Vec<usize>,
        eos: &[usize],
        vocab: &ByteLevelVocabulary,
    ) -> BunsenResult<Qwen3AsrTranscript> {
        let end = tokens
            .iter()
            .position(|t| eos.contains(t))
            .unwrap_or(tokens.len());
        let reply = &tokens[..end];
        let (language, text) = match reply.iter().position(|&t| t == self.text_token) {
            Some(at) => {
                let header = vocab.text(&reply[..at])?;
                let language = header
                    .trim()
                    .strip_prefix("language")
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case("none"))
                    .map(String::from);
                (language, vocab.text(&reply[at + 1..])?)
            }
            None => (None, vocab.text(reply)?),
        };
        Ok(Qwen3AsrTranscript {
            tokens,
            language,
            text: text.trim().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> ByteLevelVocabulary {
        let vocab = r#"{"language": 0, "ĠEnglish": 1, "ĠNone": 2, "ĠAnd": 3, "Ġso": 4}"#;
        let config = r#"{"added_tokens_decoder": {
            "5": {"content": "<asr_text>"}, "6": {"content": "<|im_end|>"}}}"#;
        ByteLevelVocabulary::from_vocab_json(vocab, Some(config)).unwrap()
    }

    #[test]
    fn test_default_prompt() {
        let config = Qwen3AsrPromptConfig::new();
        let ids = config.prompt(3, 151_676);
        assert_eq!(ids.len(), 9 + 3 + 6);
        assert_eq!(&ids[9..12], [151_676; 3]);
        assert_eq!(ids[..9], QWEN3_ASR_PROMPT_PREFIX);
        assert_eq!(ids[12..], QWEN3_ASR_PROMPT_SUFFIX);
    }

    #[test]
    fn test_parse() {
        let config = Qwen3AsrPromptConfig::new()
            .with_prefix(vec![9])
            .with_suffix(vec![8])
            .with_text_token(5);
        assert_eq!(config.prompt(2, 7), [9, 7, 7, 8]);
        let vocab = vocab();

        let t = config.parse(vec![0, 1, 5, 3, 4, 6], &[6], &vocab).unwrap();
        assert_eq!(t.language.as_deref(), Some("English"));
        assert_eq!(t.text, "And so");
        assert_eq!(t.tokens, [0, 1, 5, 3, 4, 6]);

        let t = config.parse(vec![0, 2, 5, 6], &[6], &vocab).unwrap();
        assert_eq!(t.language, None);
        assert_eq!(t.text, "");

        let t = config.parse(vec![3, 4], &[6], &vocab).unwrap();
        assert_eq!(t.language, None);
        assert_eq!(t.text, "And so");

        assert!(config.parse(vec![5, 40], &[6], &vocab).is_err());
    }
}
