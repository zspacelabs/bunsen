//! # The token layout a checkpoint's vocabulary follows.
//!
//! `whisper/tokenizer.py` appends its specials after the base vocabulary in
//! a fixed order: `<|endoftext|>` and `<|startoftranscript|>`, one token per
//! language, six control tokens, then the timestamps. What a checkpoint
//! decides is the base vocabulary and the language count, and those are
//! read off it. What it takes on convention is everything else: the
//! language codes and their order, the two base sizes, the spellings, and
//! the timestamp grid. [`WhisperTokenLayoutConfig`] declares those on the
//! model, defaulting to upstream's. The ids derived from it, and the policy
//! over them, live with the driver.

use burn::config::Config;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Language codes, in the order Whisper assigns their tokens.
///
/// `whisper/tokenizer.py::LANGUAGES`, keys only. The first 99 are the
/// original set; `yue` (Cantonese) was added for `large-v3`, which is why
/// that checkpoint's vocabulary is one token larger and every special after
/// the language block sits one id higher.
pub const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// The specials before the language block, in id order: upstream's
/// spellings.
pub const LEADING_SPECIALS: [&str; 2] = ["<|endoftext|>", "<|startoftranscript|>"];

/// The control tokens between the language block and the timestamps, in id
/// order: upstream's spellings.
pub const CONTROL_TOKENS: [&str; 6] = [
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
];

/// The token layout a checkpoint's vocabulary follows.
///
/// The roles are positional and fixed &mdash; two leading specials, the
/// language block, six control tokens, the timestamps &mdash; and this
/// names them, sizes the two base vocabularies, and times the timestamps.
#[derive(Config, Debug, PartialEq)]
pub struct WhisperTokenLayoutConfig {
    /// Language codes in token order. A layout takes a prefix of them: the
    /// first 99 for the original checkpoints, all 100 for `large-v3`.
    #[config(default = "LANGUAGES.iter().map(|&code| code.to_string()).collect()")]
    pub languages: Vec<String>,

    /// Base ranks in the English-only vocabulary, `gpt2.tiktoken`.
    #[config(default = "50256")]
    pub english_base_ranks: usize,

    /// Base ranks in the multilingual vocabulary, `multilingual.tiktoken`:
    /// one more, and that one &mdash; rank 50256 &mdash; is a genuinely
    /// empty token.
    #[config(default = "50257")]
    pub multilingual_base_ranks: usize,

    /// The two specials before the language block, in id order:
    /// `<|endoftext|>`, then `<|startoftranscript|>`.
    #[config(default = "LEADING_SPECIALS.iter().map(|&name| name.to_string()).collect()")]
    pub leading_specials: Vec<String>,

    /// The six control tokens between the language block and the
    /// timestamps, in id order: translate, transcribe, start of LM, start of
    /// previous, no speech, no timestamps.
    #[config(default = "CONTROL_TOKENS.iter().map(|&name| name.to_string()).collect()")]
    pub control_tokens: Vec<String>,

    /// The timestamp tokens: `<|0.00|>` through `<|30.00|>`.
    #[config(default = "1501")]
    pub timestamp_tokens: usize,

    /// Seconds between adjacent timestamp tokens.
    #[config(default = "0.02")]
    pub timestamp_step_seconds: f64,
}

impl WhisperTokenLayoutConfig {
    /// Checks that the layout has the roles ids are derived by position
    /// from: two leading specials, six control tokens, at least one language
    /// and one timestamp, and a positive step.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] naming what is off.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.leading_specials.len() != LEADING_SPECIALS.len() {
            return Err(BunsenError::Invalid(format!(
                "a layout has {} leading specials, not {}",
                LEADING_SPECIALS.len(),
                self.leading_specials.len(),
            )));
        }
        if self.control_tokens.len() != CONTROL_TOKENS.len() {
            return Err(BunsenError::Invalid(format!(
                "a layout has {} control tokens, not {}",
                CONTROL_TOKENS.len(),
                self.control_tokens.len(),
            )));
        }
        if self.languages.is_empty() {
            return Err(BunsenError::Invalid(
                "a layout needs at least one language".to_string(),
            ));
        }
        if self.timestamp_tokens == 0 {
            return Err(BunsenError::Invalid(
                "a layout needs at least one timestamp token".to_string(),
            ));
        }
        if self.timestamp_step_seconds.is_nan() || self.timestamp_step_seconds <= 0.0 {
            return Err(BunsenError::Invalid(format!(
                "the timestamp step must be positive, got {}",
                self.timestamp_step_seconds,
            )));
        }
        Ok(())
    }

    /// Every id but the language block, for a base vocabulary of `n_base`.
    pub fn size_without_languages(
        &self,
        n_base: usize,
    ) -> usize {
        n_base + self.leading_specials.len() + self.control_tokens.len() + self.timestamp_tokens
    }

    /// The seconds timestamp `index` denotes, relative to its window.
    pub fn timestamp_seconds(
        &self,
        index: usize,
    ) -> f64 {
        index as f64 * self.timestamp_step_seconds
    }

    /// `<|s.ss|>` for timestamp `index`, spelled as Python's
    /// `f"<|{i * 0.02:.2f}|>"` spells it &mdash; in integer hundredths, so
    /// exactly.
    pub fn timestamp_name(
        &self,
        index: usize,
    ) -> String {
        let hundredths = (self.timestamp_seconds(index) * 100.0).round() as usize;
        format!("<|{}.{:02}|>", hundredths / 100, hundredths % 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_languages_table() {
        assert_eq!(LANGUAGES.len(), 100);
        assert_eq!(LANGUAGES[0], "en");
        assert_eq!(LANGUAGES[98], "su");
        assert_eq!(LANGUAGES[99], "yue");
    }

    #[test]
    fn test_defaults_are_upstreams() {
        let layout = WhisperTokenLayoutConfig::new();
        assert!(layout.validate().is_ok());
        assert_eq!(layout.languages.len(), 100);
        assert_eq!(layout.leading_specials[0], "<|endoftext|>");
        assert_eq!(layout.control_tokens[5], "<|notimestamps|>");
        assert_eq!(layout.size_without_languages(50257), 51_766);

        assert_eq!(layout.timestamp_name(0), "<|0.00|>");
        assert_eq!(layout.timestamp_name(1), "<|0.02|>");
        assert_eq!(layout.timestamp_name(50), "<|1.00|>");
        assert_eq!(layout.timestamp_name(103), "<|2.06|>");
        assert_eq!(layout.timestamp_name(1500), "<|30.00|>");
        assert_eq!(layout.timestamp_seconds(100), 2.0);
    }

    #[test]
    fn test_validate_rejects_missing_roles() {
        let layout = WhisperTokenLayoutConfig::new();
        assert!(
            layout
                .clone()
                .with_leading_specials(vec!["<|eot|>".to_string()])
                .validate()
                .is_err()
        );
        assert!(
            layout
                .clone()
                .with_control_tokens(Vec::new())
                .validate()
                .is_err()
        );
        assert!(
            layout
                .clone()
                .with_languages(Vec::new())
                .validate()
                .is_err()
        );
        assert!(layout.clone().with_timestamp_tokens(0).validate().is_err());
        assert!(layout.with_timestamp_step_seconds(0.0).validate().is_err());
    }
}
