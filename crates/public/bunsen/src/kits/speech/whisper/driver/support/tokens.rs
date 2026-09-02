//! # Whisper's token layout.
//!
//! The decode loop runs on ids, and a handful of them are structural: the
//! prompt that selects language and task, the stop token, the no-speech
//! marker, and the 1501 timestamp tokens. None of them needs a tokenizer.
//! Whisper appends its special tokens after the base vocabulary in a fixed
//! order (`whisper/tokenizer.py::get_encoding`), so every one is arithmetic
//! over two numbers: how many base ranks the vocabulary has, and how many
//! languages the checkpoint knows.
//!
//! Both are recoverable from the checkpoint alone, through
//! [`WhisperSpecialIds::from_vocab_size`]. That is what keeps a multilingual
//! model from being driven with English-only ids, or the reverse — a mistake
//! that produces plausible text rather than an error.
//!
//! [`WhisperSpecialIds`] is the layout; [`TokenPolicy`] is the view of it the
//! decode loop holds. Text is a separate concern, and an optional one: see
//! [`text`](super::text).

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

/// Base ranks in `gpt2.tiktoken`, the English-only vocabulary.
pub const ENGLISH_BASE_RANKS: usize = 50256;

/// Base ranks in `multilingual.tiktoken`.
///
/// One more than [`ENGLISH_BASE_RANKS`], and that one — rank 50256 — is a
/// genuinely empty token. See [`TiktokenRanks`](super::TiktokenRanks).
pub const MULTILINGUAL_BASE_RANKS: usize = 50257;

/// The timestamp tokens: `<|0.00|>` through `<|30.00|>`.
pub const TIMESTAMP_TOKENS: usize = 1501;

/// Seconds between adjacent timestamp tokens.
pub const TIMESTAMP_STEP_SECONDS: f64 = 0.02;

/// Samples between adjacent timestamp tokens at Whisper's 16 kHz: two mel
/// hops, which is one encoder frame.
pub const TIMESTAMP_STEP_SAMPLES: usize = 320;

/// `<|endoftext|>` and `<|startoftranscript|>`, which precede the language
/// block.
const LEADING_SPECIALS: [&str; 2] = ["<|endoftext|>", "<|startoftranscript|>"];

/// The control tokens between the language block and the timestamps, in id
/// order.
const CONTROL_TOKENS: [&str; 6] = [
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
];

/// What the model is asked to do with the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Task {
    /// Emit the speech in its own language.
    Transcribe,
    /// Emit an English translation of it.
    Translate,
}

/// Where Whisper's special tokens sit, for one vocabulary and language count.
///
/// Every id is derived in [`new`](Self::new) from `n_base` and
/// `num_languages`: this is the layout `whisper/tokenizer.py` builds, as
/// numbers. The two real vocabularies are [`ENGLISH_BASE_RANKS`] and
/// [`MULTILINGUAL_BASE_RANKS`]; other sizes are accepted so a test can build
/// a small one.
///
/// Ids are `i64`, matching the model's token tensors; counts are `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WhisperSpecialIds {
    /// Ranks in the base vocabulary, which is also the first special id.
    pub n_base: usize,

    /// Language tokens, taken from the front of [`LANGUAGES`]: 99, or 100
    /// for `large-v3`.
    pub num_languages: usize,

    /// `<|endoftext|>`: ends a decode, and is the first non-text id.
    pub eot: i64,

    /// `<|startoftranscript|>`: the first prompt token.
    pub sot: i64,

    /// `<|en|>`. Language `i` of [`LANGUAGES`] is `language_begin + i`.
    pub language_begin: i64,

    /// `<|translate|>`.
    pub translate: i64,

    /// `<|transcribe|>`.
    pub transcribe: i64,

    /// `<|startoflm|>`.
    pub sot_lm: i64,

    /// `<|startofprev|>`: precedes the previous window's text when it is
    /// carried forward as a prompt.
    pub sot_prev: i64,

    /// `<|nospeech|>`: its probability at the first step is the no-speech
    /// score.
    pub no_speech: i64,

    /// `<|notimestamps|>`: the last prompt token when timestamps are off.
    pub no_timestamps: i64,

    /// `<|0.00|>`. Timestamp `i` is `timestamp_begin + i`, and means
    /// `i * 0.02` seconds.
    pub timestamp_begin: i64,
}

impl WhisperSpecialIds {
    /// Lays out the specials after `n_base` ranks, for `num_languages`
    /// languages.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if `num_languages` is zero or exceeds
    /// [`LANGUAGES`].
    pub fn new(
        n_base: usize,
        num_languages: usize,
    ) -> BunsenResult<Self> {
        if num_languages == 0 || num_languages > LANGUAGES.len() {
            return Err(BunsenError::Invalid(format!(
                "num_languages must be in 1..={}, got {num_languages}",
                LANGUAGES.len(),
            )));
        }

        let eot = n_base as i64;
        let sot = eot + 1;
        let language_begin = sot + 1;
        let translate = language_begin + num_languages as i64;

        Ok(Self {
            n_base,
            num_languages,
            eot,
            sot,
            language_begin,
            translate,
            transcribe: translate + 1,
            sot_lm: translate + 2,
            sot_prev: translate + 3,
            no_speech: translate + 4,
            no_timestamps: translate + 5,
            timestamp_begin: translate + 6,
        })
    }

    /// Recovers the layout from a checkpoint's vocabulary size.
    ///
    /// The rule is `whisper/model.py`'s: a vocabulary of 51865 or more is
    /// multilingual, and the language count is whatever is left after the
    /// fixed tokens. So 51864 is English-only, 51865 is multilingual with 99
    /// languages, and 51866 is `large-v3` with 100. (An English-only layout
    /// with 100 languages would also be 51865; no such checkpoint exists, and
    /// upstream makes the same call.)
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if `n_vocab` is not one of those sizes.
    pub fn from_vocab_size(n_vocab: usize) -> BunsenResult<Self> {
        let multilingual_threshold = Self::size_without_languages(MULTILINGUAL_BASE_RANKS) + 99;
        let n_base = if n_vocab >= multilingual_threshold {
            MULTILINGUAL_BASE_RANKS
        } else {
            ENGLISH_BASE_RANKS
        };

        let num_languages = n_vocab
            .checked_sub(Self::size_without_languages(n_base))
            .filter(|n| (1..=LANGUAGES.len()).contains(n))
            .ok_or_else(|| {
                BunsenError::Invalid(format!(
                    "{n_vocab} is not a Whisper vocabulary size: expected {} (English-only) or \
                     {} (multilingual) plus a language count in 1..={}",
                    Self::size_without_languages(ENGLISH_BASE_RANKS),
                    Self::size_without_languages(MULTILINGUAL_BASE_RANKS),
                    LANGUAGES.len(),
                ))
            })?;

        Self::new(n_base, num_languages)
    }

    /// Every id but the language block.
    fn size_without_languages(n_base: usize) -> usize {
        n_base + LEADING_SPECIALS.len() + CONTROL_TOKENS.len() + TIMESTAMP_TOKENS
    }

    /// Total ids: the base ranks and every special.
    pub fn n_vocab(&self) -> usize {
        Self::size_without_languages(self.n_base) + self.num_languages
    }

    /// Whether the prompt takes language and task tokens.
    ///
    /// English-only is exactly the `gpt2.tiktoken` layout. Its language and
    /// task tokens exist — the specials are appended to it in the same order
    /// — but its checkpoints were never trained on them, and upstream never
    /// emits them.
    pub fn is_multilingual(&self) -> bool {
        self.n_base != ENGLISH_BASE_RANKS
    }

    /// `<|30.00|>`: the last timestamp token, and the last id.
    pub fn timestamp_end(&self) -> i64 {
        self.timestamp_begin + (TIMESTAMP_TOKENS - 1) as i64
    }

    /// The token for a [`LANGUAGES`] code, if this layout has it.
    pub fn language_token(
        &self,
        code: &str,
    ) -> Option<i64> {
        LANGUAGES[..self.num_languages]
            .iter()
            .position(|&c| c == code)
            .map(|i| self.language_begin + i as i64)
    }

    /// The [`LANGUAGES`] code of a language token, if `id` is one.
    pub fn language_code(
        &self,
        id: i64,
    ) -> Option<&'static str> {
        let i = usize::try_from(id.checked_sub(self.language_begin)?).ok()?;
        LANGUAGES[..self.num_languages].get(i).copied()
    }

    /// The token for a [`Task`].
    pub fn task_token(
        &self,
        task: Task,
    ) -> i64 {
        match task {
            Task::Transcribe => self.transcribe,
            Task::Translate => self.translate,
        }
    }

    /// The special tokens' spellings, in id order from [`eot`](Self::eot).
    ///
    /// This is `whisper/tokenizer.py`'s `specials` list, generated rather
    /// than stored: item `i` is the spelling of id `eot + i`, and there are
    /// `n_vocab - n_base` of them — 1608 for the multilingual layout, of
    /// which 1501 are timestamps.
    pub fn special_names(&self) -> impl Iterator<Item = String> {
        LEADING_SPECIALS
            .into_iter()
            .map(String::from)
            .chain(
                LANGUAGES[..self.num_languages]
                    .iter()
                    .map(|code| format!("<|{code}|>")),
            )
            .chain(CONTROL_TOKENS.into_iter().map(String::from))
            .chain((0..TIMESTAMP_TOKENS).map(timestamp_name))
    }
}

/// `<|s.ss|>` for timestamp `index`, spelled as Python's
/// `f"<|{i * 0.02:.2f}|>"` spells it — in integer arithmetic, so exactly.
fn timestamp_name(index: usize) -> String {
    format!("<|{}.{:02}|>", index / 50, (index % 50) * 2)
}

/// The ids a decode loop consults, and the questions it asks of them.
///
/// A plain value with no dependencies, built from the checkpoint's own
/// vocabulary size — [`from_vocab_size`](Self::from_vocab_size) — so it
/// cannot disagree with the model it drives. The layout underneath is
/// [`ids`](Self::ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenPolicy {
    ids: WhisperSpecialIds,
}

impl TokenPolicy {
    /// A policy over an explicit layout.
    pub fn new(ids: WhisperSpecialIds) -> Self {
        Self { ids }
    }

    /// A policy for a checkpoint, from its vocabulary size.
    ///
    /// See [`WhisperSpecialIds::from_vocab_size`].
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if `n_vocab` is not a Whisper vocabulary
    /// size.
    pub fn from_vocab_size(n_vocab: usize) -> BunsenResult<Self> {
        WhisperSpecialIds::from_vocab_size(n_vocab).map(Self::new)
    }

    /// The layout: every special id, by name.
    pub fn ids(&self) -> &WhisperSpecialIds {
        &self.ids
    }

    /// Whether `id` is a base-vocabulary token: text, not a special.
    pub fn is_text(
        &self,
        id: i64,
    ) -> bool {
        (0..self.ids.eot).contains(&id)
    }

    /// Whether `id` is a special token: `<|endoftext|>` or above, within the
    /// vocabulary.
    pub fn is_special(
        &self,
        id: i64,
    ) -> bool {
        (self.ids.eot..self.ids.n_vocab() as i64).contains(&id)
    }

    /// Whether `id` is a timestamp token.
    pub fn is_timestamp(
        &self,
        id: i64,
    ) -> bool {
        self.timestamp_index(id).is_some()
    }

    /// The step of a timestamp token, `0..1501`, if `id` is one.
    pub fn timestamp_index(
        &self,
        id: i64,
    ) -> Option<usize> {
        let i = usize::try_from(id.checked_sub(self.ids.timestamp_begin)?).ok()?;
        (i < TIMESTAMP_TOKENS).then_some(i)
    }

    /// The timestamp token for a step, if `index` is within `0..1501`.
    pub fn timestamp_token(
        &self,
        index: usize,
    ) -> Option<i64> {
        (index < TIMESTAMP_TOKENS).then(|| self.ids.timestamp_begin + index as i64)
    }

    /// The seconds a timestamp token denotes, relative to its window, if
    /// `id` is one.
    pub fn timestamp_seconds(
        &self,
        id: i64,
    ) -> Option<f64> {
        self.timestamp_index(id)
            .map(|i| i as f64 * TIMESTAMP_STEP_SECONDS)
    }

    /// Keeps only the text tokens: what a transcript is made of.
    ///
    /// This is the filter `whisper/transcribe.py` applies before decoding
    /// text. (`Tokenizer.decode` drops only the timestamps, keeping the
    /// prompt's language and task tokens; use [`is_timestamp`] for that.)
    ///
    /// [`is_timestamp`]: Self::is_timestamp
    pub fn text_ids(
        &self,
        ids: &[i64],
    ) -> Vec<i64> {
        ids.iter().copied().filter(|&id| self.is_text(id)).collect()
    }

    /// The prompt that opens every window's decode.
    ///
    /// `<|startoftranscript|>`, then the language and task tokens when
    /// given, then `<|notimestamps|>` when `timestamps` is off — which is
    /// `Tokenizer.sot_sequence_including_notimestamps`.
    ///
    /// # Arguments
    /// * `language` - a [`LANGUAGES`] code. `None` leaves the model to choose,
    ///   which upstream resolves with a language-detection pass.
    /// * `task` - transcribe or translate. `None` leaves the model to choose.
    /// * `timestamps` - whether the model may emit timestamp tokens.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the language is not one this layout has,
    /// or if either is given for an English-only layout, which takes neither.
    pub fn sot_sequence(
        &self,
        language: Option<&str>,
        task: Option<Task>,
        timestamps: bool,
    ) -> BunsenResult<Vec<i64>> {
        let ids = &self.ids;
        let mut seq = vec![ids.sot];

        if ids.is_multilingual() {
            if let Some(code) = language {
                seq.push(ids.language_token(code).ok_or_else(|| {
                    BunsenError::Invalid(format!(
                        "unknown language `{code}`: this vocabulary has the first {} of {}",
                        ids.num_languages,
                        LANGUAGES.len(),
                    ))
                })?);
            }
            if let Some(task) = task {
                seq.push(ids.task_token(task));
            }
        } else if language.is_some() || task.is_some() {
            return Err(BunsenError::Invalid(
                "an English-only vocabulary takes no language or task token".to_string(),
            ));
        }

        if !timestamps {
            seq.push(ids.no_timestamps);
        }

        Ok(seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout `whisper.tokenizer` produces, read off the real thing for
    /// both vocabularies and both language counts.
    struct Expected {
        n_base: usize,
        num_languages: usize,
        last_language: &'static str,
        eot: i64,
        sot: i64,
        language_begin: i64,
        language_end: i64,
        translate: i64,
        transcribe: i64,
        sot_lm: i64,
        sot_prev: i64,
        no_speech: i64,
        no_timestamps: i64,
        timestamp_begin: i64,
        timestamp_end: i64,
        n_vocab: usize,
    }

    const EXPECTED: [Expected; 4] = [
        Expected {
            n_base: 50257,
            num_languages: 99,
            last_language: "su",
            eot: 50257,
            sot: 50258,
            language_begin: 50259,
            language_end: 50357,
            translate: 50358,
            transcribe: 50359,
            sot_lm: 50360,
            sot_prev: 50361,
            no_speech: 50362,
            no_timestamps: 50363,
            timestamp_begin: 50364,
            timestamp_end: 51864,
            n_vocab: 51865,
        },
        Expected {
            n_base: 50257,
            num_languages: 100,
            last_language: "yue",
            eot: 50257,
            sot: 50258,
            language_begin: 50259,
            language_end: 50358,
            translate: 50359,
            transcribe: 50360,
            sot_lm: 50361,
            sot_prev: 50362,
            no_speech: 50363,
            no_timestamps: 50364,
            timestamp_begin: 50365,
            timestamp_end: 51865,
            n_vocab: 51866,
        },
        Expected {
            n_base: 50256,
            num_languages: 99,
            last_language: "su",
            eot: 50256,
            sot: 50257,
            language_begin: 50258,
            language_end: 50356,
            translate: 50357,
            transcribe: 50358,
            sot_lm: 50359,
            sot_prev: 50360,
            no_speech: 50361,
            no_timestamps: 50362,
            timestamp_begin: 50363,
            timestamp_end: 51863,
            n_vocab: 51864,
        },
        Expected {
            n_base: 50256,
            num_languages: 100,
            last_language: "yue",
            eot: 50256,
            sot: 50257,
            language_begin: 50258,
            language_end: 50357,
            translate: 50358,
            transcribe: 50359,
            sot_lm: 50360,
            sot_prev: 50361,
            no_speech: 50362,
            no_timestamps: 50363,
            timestamp_begin: 50364,
            timestamp_end: 51864,
            n_vocab: 51865,
        },
    ];

    #[test]
    fn test_languages_table() {
        assert_eq!(LANGUAGES.len(), 100);
        assert_eq!(LANGUAGES[0], "en");
        assert_eq!(LANGUAGES[98], "su");
        assert_eq!(LANGUAGES[99], "yue");
    }

    /// Derived ids match `whisper.tokenizer` for both vocabularies and both
    /// language counts.
    #[test]
    fn test_layout_matches_upstream() {
        for e in &EXPECTED {
            let ids = WhisperSpecialIds::new(e.n_base, e.num_languages).unwrap();
            let label = format!("n_base={} num_languages={}", e.n_base, e.num_languages);

            assert_eq!(ids.eot, e.eot, "{label} eot");
            assert_eq!(ids.sot, e.sot, "{label} sot");
            assert_eq!(
                ids.language_begin, e.language_begin,
                "{label} language_begin"
            );
            assert_eq!(
                ids.language_token(e.last_language),
                Some(e.language_end),
                "{label} last language"
            );
            assert_eq!(ids.translate, e.translate, "{label} translate");
            assert_eq!(ids.transcribe, e.transcribe, "{label} transcribe");
            assert_eq!(ids.sot_lm, e.sot_lm, "{label} sot_lm");
            assert_eq!(ids.sot_prev, e.sot_prev, "{label} sot_prev");
            assert_eq!(ids.no_speech, e.no_speech, "{label} no_speech");
            assert_eq!(ids.no_timestamps, e.no_timestamps, "{label} no_timestamps");
            assert_eq!(
                ids.timestamp_begin, e.timestamp_begin,
                "{label} timestamp_begin"
            );
            assert_eq!(
                ids.timestamp_end(),
                e.timestamp_end,
                "{label} timestamp_end"
            );
            assert_eq!(ids.n_vocab(), e.n_vocab, "{label} n_vocab");
        }
    }

    #[test]
    fn test_new_rejects_bad_language_counts() {
        assert!(WhisperSpecialIds::new(50257, 0).is_err());
        assert!(WhisperSpecialIds::new(50257, 101).is_err());
    }

    /// The checkpoint's vocabulary size alone picks the layout.
    #[test]
    fn test_from_vocab_size() {
        for e in &EXPECTED {
            // English-only with 100 languages is the same size as multilingual
            // with 99; upstream reads that size as multilingual, and so do we.
            if e.n_base == ENGLISH_BASE_RANKS && e.num_languages == 100 {
                continue;
            }
            let ids = WhisperSpecialIds::from_vocab_size(e.n_vocab).unwrap();
            assert_eq!(
                ids,
                WhisperSpecialIds::new(e.n_base, e.num_languages).unwrap()
            );
        }

        let ids = WhisperSpecialIds::from_vocab_size(51865).unwrap();
        assert!(ids.is_multilingual());
        assert_eq!(ids.num_languages, 99);

        let ids = WhisperSpecialIds::from_vocab_size(51864).unwrap();
        assert!(!ids.is_multilingual());
        assert_eq!(ids.num_languages, 99);

        // Too small, zero languages, too many languages.
        assert!(WhisperSpecialIds::from_vocab_size(32).is_err());
        assert!(WhisperSpecialIds::from_vocab_size(51765).is_err());
        assert!(WhisperSpecialIds::from_vocab_size(51867).is_err());
    }

    #[test]
    fn test_language_lookup() {
        let ids = WhisperSpecialIds::new(MULTILINGUAL_BASE_RANKS, 99).unwrap();

        assert_eq!(ids.language_token("en"), Some(50259));
        assert_eq!(ids.language_token("zh"), Some(50260));
        assert_eq!(ids.language_token("su"), Some(50357));
        assert_eq!(ids.language_token("yue"), None, "beyond num_languages");
        assert_eq!(ids.language_token("xx"), None);

        assert_eq!(ids.language_code(50259), Some("en"));
        assert_eq!(ids.language_code(50357), Some("su"));
        assert_eq!(ids.language_code(50358), None, "that is <|translate|>");
        assert_eq!(
            ids.language_code(50258),
            None,
            "that is <|startoftranscript|>"
        );
        assert_eq!(ids.language_code(-1), None);

        let large_v3 = WhisperSpecialIds::new(MULTILINGUAL_BASE_RANKS, 100).unwrap();
        assert_eq!(large_v3.language_token("yue"), Some(50358));
        assert_eq!(large_v3.language_code(50358), Some("yue"));

        assert_eq!(ids.task_token(Task::Transcribe), ids.transcribe);
        assert_eq!(ids.task_token(Task::Translate), ids.translate);
    }

    /// The generated spellings are `whisper/tokenizer.py`'s `specials` list.
    #[test]
    fn test_special_names() {
        for e in &EXPECTED {
            let ids = WhisperSpecialIds::new(e.n_base, e.num_languages).unwrap();
            let names: Vec<String> = ids.special_names().collect();
            let at = |id: i64| &names[(id - ids.eot) as usize];

            assert_eq!(names.len(), e.n_vocab - e.n_base);
            assert_eq!(at(ids.eot), "<|endoftext|>");
            assert_eq!(at(ids.sot), "<|startoftranscript|>");
            assert_eq!(at(ids.language_begin), "<|en|>");
            assert_eq!(at(e.language_end), &format!("<|{}|>", e.last_language));
            assert_eq!(at(ids.translate), "<|translate|>");
            assert_eq!(at(ids.transcribe), "<|transcribe|>");
            assert_eq!(at(ids.sot_lm), "<|startoflm|>");
            assert_eq!(at(ids.sot_prev), "<|startofprev|>");
            assert_eq!(at(ids.no_speech), "<|nospeech|>");
            assert_eq!(at(ids.no_timestamps), "<|notimestamps|>");
            assert_eq!(at(ids.timestamp_begin), "<|0.00|>");
            assert_eq!(at(ids.timestamp_begin + 1), "<|0.02|>");
            assert_eq!(at(ids.timestamp_begin + 50), "<|1.00|>");
            assert_eq!(at(ids.timestamp_begin + 103), "<|2.06|>");
            assert_eq!(at(ids.timestamp_end()), "<|30.00|>");
        }
    }

    #[test]
    fn test_policy_predicates() {
        let policy = TokenPolicy::from_vocab_size(51865).unwrap();
        let ids = *policy.ids();

        assert!(policy.is_text(0));
        assert!(policy.is_text(50256), "the empty token is still a text id");
        assert!(!policy.is_text(ids.eot));
        assert!(!policy.is_text(-1));

        assert!(policy.is_special(ids.eot));
        assert!(policy.is_special(ids.timestamp_end()));
        assert!(
            !policy.is_special(ids.timestamp_end() + 1),
            "past the vocabulary"
        );
        assert!(!policy.is_special(0));

        assert!(!policy.is_timestamp(ids.no_timestamps));
        assert!(policy.is_timestamp(ids.timestamp_begin));
        assert!(policy.is_timestamp(ids.timestamp_end()));
        assert!(!policy.is_timestamp(ids.timestamp_end() + 1));

        assert_eq!(policy.timestamp_index(ids.timestamp_begin), Some(0));
        assert_eq!(policy.timestamp_index(50464), Some(100));
        assert_eq!(policy.timestamp_index(51864), Some(1500));
        assert_eq!(policy.timestamp_index(51865), None);
        assert_eq!(policy.timestamp_index(i64::MIN), None);

        assert_eq!(policy.timestamp_token(0), Some(ids.timestamp_begin));
        assert_eq!(policy.timestamp_token(1500), Some(51864));
        assert_eq!(policy.timestamp_token(1501), None);

        assert_eq!(policy.timestamp_seconds(50364), Some(0.0));
        assert_eq!(policy.timestamp_seconds(50464), Some(2.0));
        assert_eq!(policy.timestamp_seconds(51864), Some(30.0));
        assert_eq!(policy.timestamp_seconds(50363), None);

        // Prompt, timestamps, text, timestamp, eot: only the text survives.
        let window = [50258, 50259, 50359, 50364, 15947, 1002, 13, 50464, 50257];
        assert_eq!(policy.text_ids(&window), [15947, 1002, 13]);
    }

    /// `Tokenizer.sot_sequence` and `sot_sequence_including_notimestamps`,
    /// for the multilingual and English-only tokenizers.
    #[test]
    fn test_sot_sequence() {
        let multilingual = TokenPolicy::from_vocab_size(51865).unwrap();

        assert_eq!(
            multilingual
                .sot_sequence(Some("en"), Some(Task::Transcribe), true)
                .unwrap(),
            [50258, 50259, 50359],
        );
        assert_eq!(
            multilingual
                .sot_sequence(Some("en"), Some(Task::Transcribe), false)
                .unwrap(),
            [50258, 50259, 50359, 50363],
        );
        assert_eq!(
            multilingual
                .sot_sequence(Some("ja"), Some(Task::Translate), false)
                .unwrap(),
            [50258, 50266, 50358, 50363],
        );
        assert_eq!(
            multilingual.sot_sequence(None, None, true).unwrap(),
            [50258]
        );
        assert_eq!(
            multilingual
                .sot_sequence(None, Some(Task::Transcribe), true)
                .unwrap(),
            [50258, 50359],
        );
        assert!(multilingual.sot_sequence(Some("xx"), None, true).is_err());
        assert!(
            multilingual.sot_sequence(Some("yue"), None, true).is_err(),
            "yue needs the 100-language layout",
        );

        let english = TokenPolicy::from_vocab_size(51864).unwrap();
        assert_eq!(english.sot_sequence(None, None, true).unwrap(), [50257]);
        assert_eq!(
            english.sot_sequence(None, None, false).unwrap(),
            [50257, 50362]
        );
        assert!(english.sot_sequence(Some("en"), None, true).is_err());
        assert!(
            english
                .sot_sequence(None, Some(Task::Transcribe), true)
                .is_err()
        );
    }
}
