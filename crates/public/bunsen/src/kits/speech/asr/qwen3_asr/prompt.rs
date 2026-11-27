//! Qwen3-ASR's chat prompt, and parsing its reply.
use alloc::{
    format,
    string::String,
    vec::Vec,
};

use burn::config::Config;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::support::tokens::ByteLevelVocabulary,
};

/// The chat template's pieces before the audio, `transformers`' with an
/// empty system turn: `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n`
/// `<|audio_start|>`.
const PREFIX: [&str; 9] = [
    "<|im_start|>",
    "system",
    "\n",
    "<|im_end|>",
    "\n",
    "<|im_start|>",
    "user",
    "\n",
    "<|audio_start|>",
];

/// The pieces after the audio, through the generation prompt:
/// `<|audio_end|><|im_end|>\n<|im_start|>assistant\n`.
const SUFFIX: [&str; 6] = [
    "<|audio_end|>",
    "<|im_end|>",
    "\n",
    "<|im_start|>",
    "assistant",
    "\n",
];

/// The piece ending a reply's header and beginning its transcript.
const TEXT_TOKEN: &str = "<asr_text>";

/// The id of `piece` in `vocab`: an added token by its content, else the
/// entry that spells it.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
/// naming the piece, for a vocabulary that has neither.
pub(crate) fn piece_id(
    vocab: &ByteLevelVocabulary,
    piece: &str,
) -> BunsenResult<usize> {
    vocab
        .added_id(piece)
        .or_else(|| vocab.piece_id(piece.as_bytes()))
        .ok_or_else(|| {
            BunsenError::invalid_resource(format!(
                "a Qwen3-ASR vocabulary without {piece:?}, as one token"
            ))
        })
}

/// The chat template a Qwen3-ASR model is prompted with, as token ids: the
/// prompt around an utterance's audio placeholders, and the token splitting
/// the reply's header from its transcript.
///
/// A model's are its vocabulary's ids for `transformers`' chat template, with
/// an empty system turn and the generation prompt
/// ([`from_vocabulary`](Self::from_vocabulary)).
#[derive(Config, Debug)]
pub struct Qwen3AsrPromptConfig {
    /// The tokens before the audio placeholders.
    pub prefix: Vec<usize>,

    /// The tokens after the audio placeholders, through the assistant turn's
    /// opening.
    pub suffix: Vec<usize>,

    /// The token ending the reply's header (`language English`, say) and
    /// beginning its transcript (`<asr_text>`).
    pub text_token: usize,
}

/// How a reply was laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3AsrReplyForm {
    /// A header (`language English`), the text token, the transcript: the
    /// form the model is trained to reply in.
    Headed,

    /// No text token: the reply is read as all transcript. A model replies so
    /// when it skips the header, or runs out of tokens before the text token.
    Unheaded,
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

    /// How the reply was laid out: with a header, or without one and read as
    /// all transcript.
    pub form: Qwen3AsrReplyForm,
}

impl Qwen3AsrPromptConfig {
    /// The template's ids in `vocab`: each piece an added token by its
    /// content, or the vocabulary entry that spells it.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
    /// naming the piece, for one the vocabulary lacks.
    pub fn from_vocabulary(vocab: &ByteLevelVocabulary) -> BunsenResult<Self> {
        let ids = |pieces: &[&str]| {
            pieces
                .iter()
                .map(|piece| piece_id(vocab, piece))
                .collect::<BunsenResult<Vec<_>>>()
        };
        Ok(Self::new(
            ids(&PREFIX)?,
            ids(&SUFFIX)?,
            piece_id(vocab, TEXT_TOKEN)?,
        ))
    }

    /// The prompt of an utterance whose audio is `n_audio` encoder frames:
    /// the prefix, `n_audio` copies of `audio_token`, the suffix.
    pub fn prompt(
        &self,
        n_audio: usize,
        audio_token: usize,
    ) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.prefix.len() + n_audio + self.suffix.len());
        ids.extend_from_slice(&self.prefix);
        ids.extend(core::iter::repeat_n(audio_token, n_audio));
        ids.extend_from_slice(&self.suffix);
        ids
    }

    /// Splits a reply into its header, `language <name>`, and its
    /// transcript, the ids on either side of the text token.
    ///
    /// The tokens from the first of `eos` on are neither. A reply without
    /// the text token is all transcript, with no header
    /// ([`Qwen3AsrReplyForm::Unheaded`]).
    pub fn split_reply<'a>(
        &self,
        tokens: &'a [usize],
        eos: &[usize],
    ) -> (Option<&'a [usize]>, &'a [usize]) {
        let end = tokens
            .iter()
            .position(|t| eos.contains(t))
            .unwrap_or(tokens.len());
        let reply = &tokens[..end];
        match reply.iter().position(|&t| t == self.text_token) {
            Some(at) => (Some(&reply[..at]), &reply[at + 1..]),
            None => (None, reply),
        }
    }

    /// Parses a reply: `language <name>`, the text token, the transcript
    /// ([`split_reply`](Self::split_reply)); a reply without the text token
    /// is all transcript, and says so in its
    /// [`form`](Qwen3AsrTranscript::form).
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
        let (header, text) = self.split_reply(&tokens, eos);
        let form = match header {
            Some(_) => Qwen3AsrReplyForm::Headed,
            None => Qwen3AsrReplyForm::Unheaded,
        };
        let language = match header {
            Some(header) => vocab
                .text(header)?
                .trim()
                .strip_prefix("language")
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.eq_ignore_ascii_case("none"))
                .map(String::from),
            None => None,
        };
        let text = vocab.text(text)?.trim().to_string();
        Ok(Qwen3AsrTranscript {
            tokens,
            language,
            text,
            form,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::testing::ErrorMatcher;

    fn vocab() -> ByteLevelVocabulary {
        let vocab = r#"{"language": 0, "ĠEnglish": 1, "ĠNone": 2, "ĠAnd": 3, "Ġso": 4}"#;
        let config = r#"{"added_tokens_decoder": {
            "5": {"content": "<asr_text>"}, "6": {"content": "<|im_end|>"}}}"#;
        ByteLevelVocabulary::from_vocab_json(vocab, Some(config)).unwrap()
    }

    /// The release's specials and pieces, at the release's ids.
    fn release_vocab() -> ByteLevelVocabulary {
        let vocab = r#"{"system": 8948, "user": 872, "assistant": 77091, "Ċ": 198}"#;
        let config = r#"{"added_tokens_decoder": {
            "151643": {"content": "<|endoftext|>"}, "151644": {"content": "<|im_start|>"},
            "151645": {"content": "<|im_end|>"}, "151669": {"content": "<|audio_start|>"},
            "151670": {"content": "<|audio_end|>"}, "151676": {"content": "<|audio_pad|>"},
            "151704": {"content": "<asr_text>"}}}"#;
        ByteLevelVocabulary::from_vocab_json(vocab, Some(config)).unwrap()
    }

    /// The template's ids are the vocabulary's: at the release's ids, the
    /// release's prompt; a vocabulary without a piece is refused, naming it.
    #[test]
    fn test_from_vocabulary() {
        let config = Qwen3AsrPromptConfig::from_vocabulary(&release_vocab()).unwrap();
        let ids = config.prompt(3, 151_676);
        assert_eq!(ids.len(), 9 + 3 + 6);
        assert_eq!(&ids[9..12], [151_676; 3]);
        assert_eq!(
            ids[..9],
            [151_644, 8948, 198, 151_645, 198, 151_644, 872, 198, 151_669]
        );
        assert_eq!(ids[12..], [151_670, 151_645, 198, 151_644, 77_091, 198]);
        assert_eq!(config.text_token, 151_704);
        let decoder =
            crate::kits::speech::asr::qwen3_asr::Qwen3AsrGreedyDecoderConfig::from_vocabulary(
                &release_vocab(),
            )
            .unwrap();
        assert_eq!(decoder.eos, [151_643, 151_645]);

        crate::errors::testing::ErrorMatcher::invalid_resource()
            .message_contains("<|im_start|>")
            .assert_err(&Qwen3AsrPromptConfig::from_vocabulary(&vocab()));
    }

    #[test]
    fn test_parse() {
        let config = Qwen3AsrPromptConfig::new(vec![9], vec![8], 5);
        assert_eq!(config.prompt(2, 7), [9, 7, 7, 8]);
        let vocab = vocab();

        let t = config.parse(vec![0, 1, 5, 3, 4, 6], &[6], &vocab).unwrap();
        assert_eq!(t.language.as_deref(), Some("English"));
        assert_eq!(t.text, "And so");
        assert_eq!(t.tokens, [0, 1, 5, 3, 4, 6]);
        assert_eq!(t.form, Qwen3AsrReplyForm::Headed);

        let t = config.parse(vec![0, 2, 5, 6], &[6], &vocab).unwrap();
        assert_eq!(t.language, None);
        assert_eq!(t.text, "");

        let t = config.parse(vec![3, 4], &[6], &vocab).unwrap();
        assert_eq!(t.language, None);
        assert_eq!(t.text, "And so");
        assert_eq!(t.form, Qwen3AsrReplyForm::Unheaded, "no text token");

        ErrorMatcher::invalid_resource().assert_err(&config.parse(vec![5, 40], &[6], &vocab));
    }

    #[test]
    fn test_split_reply() {
        let config = Qwen3AsrPromptConfig::new(vec![], vec![], 5);
        let tokens = [0, 1, 5, 3, 4, 6, 9];
        assert_eq!(
            config.split_reply(&tokens, &[6]),
            (Some(&tokens[..2]), &tokens[3..5])
        );
        assert_eq!(
            config.split_reply(&[0, 5, 6], &[6]),
            (Some(&[0][..]), &[][..])
        );
        assert_eq!(config.split_reply(&[3, 4], &[6]), (None, &[3, 4][..]));
        assert_eq!(config.split_reply(&[6, 5, 3], &[6]), (None, &[][..]));
    }
}
