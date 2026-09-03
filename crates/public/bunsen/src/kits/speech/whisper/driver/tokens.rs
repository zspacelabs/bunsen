//! # Whisper's token layout, as ids.
//!
//! The decode loop runs on ids, and a handful of them are structural: the
//! prompt that selects language and task, the stop token, the no-speech
//! marker, and the timestamp tokens. None of them needs a tokenizer.
//! Whisper appends its special tokens after the base vocabulary in a fixed
//! order (`whisper/tokenizer.py::get_encoding`), so every one is arithmetic
//! over two numbers a checkpoint decides &mdash; how many base ranks the
//! vocabulary has, and how many languages it knows &mdash; and a
//! [`WhisperTokenLayoutConfig`] for everything it takes on convention.
//!
//! Both numbers are recoverable from the checkpoint alone, through
//! [`WhisperTokenLayoutConfig::special_ids_for_vocab`]. That is what keeps a
//! multilingual model from being driven with English-only ids, or the
//! reverse &mdash; a mistake that produces plausible text rather than an
//! error.
//!
//! [`WhisperSpecialIds`] is the layout as numbers, a plain `Copy` value;
//! [`WhisperTokenLayout`] pairs it with the layout's names and timestamp grid,
//! and is the view of it the decode loop holds. Text is a separate concern, and
//! an optional one: see [`text`](super::text).

use std::sync::Arc;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::{
        blocks::WhisperTokenLayoutConfig,
        driver::TiktokenRanks,
    },
};

/// What the model is asked to do with the audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum WhisperTask {
    /// Emit the speech in its own language.
    Transcribe,
    /// Emit an English translation of it.
    Translate,
}

/// Where Whisper's special tokens sit, for one vocabulary and language count.
///
/// Every id is derived by [`WhisperTokenLayoutConfig::special_ids`] from
/// `n_base` and `num_languages`: this is the layout `whisper/tokenizer.py`
/// builds, as numbers, with nothing named. The two real base vocabularies
/// are the layout's `english_base_ranks` and `multilingual_base_ranks`;
/// other sizes are accepted so a test can build a small one.
///
/// Ids are `i64`, matching the model's token tensors; counts are `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WhisperSpecialIds {
    /// Ranks in the base vocabulary, which is also the first special id.
    pub n_base: usize,

    /// Language tokens, taken from the front of the layout's languages: 99,
    /// or 100 for `large-v3`.
    pub num_languages: usize,

    /// Timestamp tokens, from [`timestamp_begin`](Self::timestamp_begin) up.
    pub timestamp_tokens: usize,

    /// Whether the prompt takes language and task tokens: the base
    /// vocabulary is not the English-only one.
    ///
    /// English-only is exactly the `gpt2.tiktoken` layout. Its language and
    /// task tokens exist &mdash; the specials are appended to it in the same
    /// order &mdash; but its checkpoints were never trained on them, and
    /// upstream never emits them.
    pub multilingual: bool,

    /// `<|endoftext|>`: ends a decode, and is the first non-text id.
    pub eot: i64,

    /// `<|startoftranscript|>`: the first prompt token.
    pub sot: i64,

    /// `<|en|>`. Language `i` of the layout is `language_begin + i`.
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
    /// `i` steps of the layout's grid.
    pub timestamp_begin: i64,
}

impl WhisperSpecialIds {
    /// Lays out the specials after `n_base` ranks, for `num_languages`
    /// languages, under upstream's layout.
    ///
    /// [`WhisperTokenLayoutConfig::special_ids`] on the default config.
    ///
    /// # Errors
    /// As it.
    pub fn new(
        n_base: usize,
        num_languages: usize,
    ) -> BunsenResult<Self> {
        WhisperTokenLayoutConfig::new().special_ids(n_base, num_languages)
    }

    /// Recovers upstream's layout from a checkpoint's vocabulary size.
    ///
    /// [`WhisperTokenLayoutConfig::special_ids_for_vocab`] on the default
    /// config.
    ///
    /// # Errors
    /// As it.
    pub fn from_vocab_size(n_vocab: usize) -> BunsenResult<Self> {
        WhisperTokenLayoutConfig::new().special_ids_for_vocab(n_vocab)
    }

    /// Total ids: the base ranks and every special.
    pub fn n_vocab(&self) -> usize {
        self.timestamp_begin as usize + self.timestamp_tokens
    }

    /// Whether the prompt takes language and task tokens.
    pub fn is_multilingual(&self) -> bool {
        self.multilingual
    }

    /// The last timestamp token, and the last id: `<|30.00|>` upstream.
    pub fn timestamp_end(&self) -> i64 {
        self.timestamp_begin + (self.timestamp_tokens - 1) as i64
    }

    /// The token for a [`WhisperTask`].
    pub fn task_token(
        &self,
        task: WhisperTask,
    ) -> i64 {
        match task {
            WhisperTask::Transcribe => self.transcribe,
            WhisperTask::Translate => self.translate,
        }
    }
}

impl WhisperTokenLayoutConfig {
    /// Lays out the specials after `n_base` ranks, for `num_languages` of
    /// this layout's languages.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the layout does not
    /// [`validate`](Self::validate), or `num_languages` is zero or exceeds
    /// the languages it has.
    pub fn special_ids(
        &self,
        n_base: usize,
        num_languages: usize,
    ) -> BunsenResult<WhisperSpecialIds> {
        self.validate()?;
        if num_languages == 0 || num_languages > self.languages.len() {
            return Err(BunsenError::Invalid(format!(
                "num_languages must be in 1..={}, got {num_languages}",
                self.languages.len(),
            )));
        }

        // The leading specials, the language block, the control tokens, the
        // timestamps; `validate` has fixed the first and third at two and six.
        let eot = n_base as i64;
        let sot = eot + 1;
        let language_begin = sot + 1;
        let translate = language_begin + num_languages as i64;

        Ok(WhisperSpecialIds {
            n_base,
            num_languages,
            timestamp_tokens: self.timestamp_tokens,
            multilingual: n_base != self.english_base_ranks,
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
    pub fn special_ids_for_vocab(
        &self,
        n_vocab: usize,
    ) -> BunsenResult<WhisperSpecialIds> {
        // Upstream's threshold is the multilingual layout with the original
        // 99 languages; `large-v3`'s hundredth came later.
        let multilingual_threshold = self.size_without_languages(self.multilingual_base_ranks) + 99;
        let n_base = if n_vocab >= multilingual_threshold {
            self.multilingual_base_ranks
        } else {
            self.english_base_ranks
        };

        let num_languages = n_vocab
            .checked_sub(self.size_without_languages(n_base))
            .filter(|n| (1..=self.languages.len()).contains(n))
            .ok_or_else(|| {
                BunsenError::Invalid(format!(
                    "{n_vocab} is not a Whisper vocabulary size: expected {} (English-only) or \
                     {} (multilingual) plus a language count in 1..={}",
                    self.size_without_languages(self.english_base_ranks),
                    self.size_without_languages(self.multilingual_base_ranks),
                    self.languages.len(),
                ))
            })?;

        self.special_ids(n_base, num_languages)
    }

    /// A [`WhisperTokenLayout`] over this layout, for `n_base` ranks and
    /// `num_languages` languages.
    ///
    /// # Errors
    /// As [`special_ids`](Self::special_ids).
    pub fn policy(
        &self,
        n_base: usize,
        num_languages: usize,
    ) -> BunsenResult<WhisperTokenLayout> {
        Ok(WhisperTokenLayout::with_layout(
            self.clone(),
            self.special_ids(n_base, num_languages)?,
        ))
    }

    /// A [`WhisperTokenLayout`] over this layout, for a checkpoint's vocabulary
    /// size.
    ///
    /// # Errors
    /// As [`special_ids_for_vocab`](Self::special_ids_for_vocab).
    pub fn policy_for_vocab(
        &self,
        n_vocab: usize,
    ) -> BunsenResult<WhisperTokenLayout> {
        Ok(WhisperTokenLayout::with_layout(
            self.clone(),
            self.special_ids_for_vocab(n_vocab)?,
        ))
    }
}

/// The ids a decode loop consults, and the questions it asks of them.
///
/// The [`ids`](Self::ids) as numbers, with the [`layout`](Self::layout)
/// that names them and times the timestamps. Built from the checkpoint's own
/// vocabulary size &mdash; [`WhisperTokenLayoutConfig::policy_for_vocab`]
/// &mdash; so it cannot disagree with the model it drives. Cheap to clone:
/// the layout is shared.
#[derive(Debug, Clone, PartialEq)]
pub struct WhisperTokenLayout {
    layout: Arc<WhisperTokenLayoutConfig>,
    ids: WhisperSpecialIds,
}

impl WhisperTokenLayout {
    /// A policy over an explicit layout of ids, with upstream's names and
    /// timestamp grid.
    pub fn new(ids: WhisperSpecialIds) -> Self {
        Self::with_layout(WhisperTokenLayoutConfig::new(), ids)
    }

    /// A policy over `ids`, named and timed by `layout`.
    pub fn with_layout(
        layout: WhisperTokenLayoutConfig,
        ids: WhisperSpecialIds,
    ) -> Self {
        Self {
            layout: Arc::new(layout),
            ids,
        }
    }

    /// A policy for a checkpoint, from its vocabulary size, under upstream's
    /// layout.
    ///
    /// See [`WhisperTokenLayoutConfig::special_ids_for_vocab`].
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if `n_vocab` is not a Whisper vocabulary
    /// size.
    pub fn from_vocab_size(n_vocab: usize) -> BunsenResult<Self> {
        WhisperTokenLayoutConfig::new().policy_for_vocab(n_vocab)
    }

    /// The layout: every special id, by name.
    pub fn ids(&self) -> &WhisperSpecialIds {
        &self.ids
    }

    /// The layout's names and timestamp grid.
    pub fn layout(&self) -> &WhisperTokenLayoutConfig {
        &self.layout
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

    /// The step of a timestamp token, `0..timestamp_tokens`, if `id` is one.
    pub fn timestamp_index(
        &self,
        id: i64,
    ) -> Option<usize> {
        let i = usize::try_from(id.checked_sub(self.ids.timestamp_begin)?).ok()?;
        (i < self.ids.timestamp_tokens).then_some(i)
    }

    /// The timestamp token for a step, if `index` is within
    /// `0..timestamp_tokens`.
    pub fn timestamp_token(
        &self,
        index: usize,
    ) -> Option<i64> {
        (index < self.ids.timestamp_tokens).then(|| self.ids.timestamp_begin + index as i64)
    }

    /// The seconds a timestamp token denotes, relative to its window, if
    /// `id` is one.
    pub fn timestamp_seconds(
        &self,
        id: i64,
    ) -> Option<f64> {
        self.timestamp_index(id)
            .map(|i| self.layout.timestamp_seconds(i))
    }

    /// The language codes this layout has, in token order.
    pub fn languages(&self) -> &[String] {
        &self.layout.languages[..self.ids.num_languages]
    }

    /// The token for a language code, if this layout has it.
    pub fn language_token(
        &self,
        code: &str,
    ) -> Option<i64> {
        self.languages()
            .iter()
            .position(|c| c == code)
            .map(|i| self.ids.language_begin + i as i64)
    }

    /// The language code of a language token, if `id` is one.
    pub fn language_code(
        &self,
        id: i64,
    ) -> Option<&str> {
        let i = usize::try_from(id.checked_sub(self.ids.language_begin)?).ok()?;
        self.languages().get(i).map(String::as_str)
    }

    /// The special tokens' spellings, in id order from
    /// [`eot`](WhisperSpecialIds::eot).
    ///
    /// This is `whisper/tokenizer.py`'s `specials` list, generated rather
    /// than stored: item `i` is the spelling of id `eot + i`, and there are
    /// `n_vocab - n_base` of them &mdash; 1608 for the multilingual layout,
    /// of which 1501 are timestamps.
    pub fn special_names(&self) -> impl Iterator<Item = String> + '_ {
        self.layout
            .leading_specials
            .iter()
            .cloned()
            .chain(self.languages().iter().map(|code| format!("<|{code}|>")))
            .chain(self.layout.control_tokens.iter().cloned())
            .chain((0..self.ids.timestamp_tokens).map(|i| self.layout.timestamp_name(i)))
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
    /// given, then `<|notimestamps|>` when `timestamps` is off &mdash; which
    /// is `Tokenizer.sot_sequence_including_notimestamps`.
    ///
    /// # Arguments
    /// * `language` - a language code of the layout. `None` leaves the model to
    ///   choose, which upstream resolves with a language-detection pass.
    /// * `task` - transcribe or translate. `None` leaves the model to choose.
    /// * `timestamps` - whether the model may emit timestamp tokens.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the language is not one this layout has,
    /// or if either is given for an English-only layout, which takes neither.
    pub fn sot_sequence(
        &self,
        language: Option<&str>,
        task: Option<WhisperTask>,
        timestamps: bool,
    ) -> BunsenResult<Vec<i64>> {
        let ids = &self.ids;
        let mut seq = vec![ids.sot];

        if ids.is_multilingual() {
            if let Some(code) = language {
                seq.push(self.language_token(code).ok_or_else(|| {
                    BunsenError::Invalid(format!(
                        "unknown language `{code}`: this vocabulary has the first {} of {}",
                        ids.num_languages,
                        self.layout.languages.len(),
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

    /// The full `{ id -> bytes }` table of a Whisper vocabulary, indexed by id.
    ///
    /// The base ranks first, then every special in
    /// [`WhisperSpecialIds`](super::WhisperSpecialIds)'s order, spelled as
    /// `<|name|>` — `n_vocab` entries in all.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the ranks are not the base of
    /// `WhisperTokenlayout`'s layout: the vocabulary file and the
    /// checkpoint disagree.
    pub fn token_spans(
        &self,
        ranks: &TiktokenRanks,
    ) -> BunsenResult<Vec<Vec<u8>>> {
        let ids = self.ids();
        if ranks.len() != ids.n_base {
            return Err(BunsenError::Invalid(format!(
                "the vocabulary has {} base ranks but the layout expects {}",
                ranks.len(),
                ids.n_base,
            )));
        }

        let mut spans = Vec::with_capacity(ids.n_vocab());
        spans.extend(ranks.iter().map(<[u8]>::to_vec));
        spans.extend(self.special_names().map(String::into_bytes));
        Ok(spans)
    }

    /// A [`Detokenizer`](crate::kits::tokens::Detokenizer) over a Whisper
    /// vocabulary.
    ///
    /// [`Self::token_spans`] handed to `wordchipper`'s decode-only path.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the ranks are not the base of
    /// `WhisperTokenlayout`'s layout: the vocabulary file and the
    /// checkpoint disagree.
    #[cfg(feature = "tokenizer")]
    pub fn detokenizer(
        &self,
        ranks: &TiktokenRanks,
    ) -> BunsenResult<crate::kits::tokens::WordchipperDetokenizer<u16>> {
        crate::kits::tokens::WordchipperDetokenizer::from_spans(
            self.token_spans(ranks)?.into_iter().enumerate(),
        )
    }

    /// [`Self::detokenizer`] over a `.tiktoken` file.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the ranks are not the base of
    /// `WhisperTokenlayout`'s layout: the vocabulary file and the
    /// checkpoint disagree.
    #[cfg(feature = "tokenizer")]
    pub fn load_detokenizer(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> BunsenResult<crate::kits::tokens::WordchipperDetokenizer<u16>> {
        self.detokenizer(&TiktokenRanks::load(path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kits::speech::whisper::{
        blocks::CONTROL_TOKENS,
        driver::tokens::WhisperSpecialIds,
    };

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

    /// Derived ids match `whisper.tokenizer` for both vocabularies and both
    /// language counts.
    #[test]
    fn test_layout_matches_upstream() {
        for e in &EXPECTED {
            let ids = WhisperSpecialIds::new(e.n_base, e.num_languages).unwrap();
            let policy = WhisperTokenLayout::new(ids);
            let label = format!("n_base={} num_languages={}", e.n_base, e.num_languages);

            assert_eq!(ids.eot, e.eot, "{label} eot");
            assert_eq!(ids.sot, e.sot, "{label} sot");
            assert_eq!(
                ids.language_begin, e.language_begin,
                "{label} language_begin"
            );
            assert_eq!(
                policy.language_token(e.last_language),
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
            assert_eq!(ids.is_multilingual(), e.n_base == 50257, "{label}");
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
        let layout = WhisperTokenLayoutConfig::new();
        for e in &EXPECTED {
            // English-only with 100 languages is the same size as multilingual
            // with 99; upstream reads that size as multilingual, and so do we.
            if e.n_base == layout.english_base_ranks && e.num_languages == 100 {
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
        let policy = WhisperTokenLayout::from_vocab_size(51865).unwrap();
        let ids = *policy.ids();

        assert_eq!(policy.language_token("en"), Some(50259));
        assert_eq!(policy.language_token("zh"), Some(50260));
        assert_eq!(policy.language_token("su"), Some(50357));
        assert_eq!(policy.language_token("yue"), None, "beyond num_languages");
        assert_eq!(policy.language_token("xx"), None);

        assert_eq!(policy.language_code(50259), Some("en"));
        assert_eq!(policy.language_code(50357), Some("su"));
        assert_eq!(policy.language_code(50358), None, "that is <|translate|>");
        assert_eq!(
            policy.language_code(50258),
            None,
            "that is <|startoftranscript|>"
        );
        assert_eq!(policy.language_code(-1), None);
        assert_eq!(policy.languages().len(), 99);

        let large_v3 = WhisperTokenLayout::from_vocab_size(51866).unwrap();
        assert_eq!(large_v3.language_token("yue"), Some(50358));
        assert_eq!(large_v3.language_code(50358), Some("yue"));

        assert_eq!(ids.task_token(WhisperTask::Transcribe), ids.transcribe);
        assert_eq!(ids.task_token(WhisperTask::Translate), ids.translate);
    }

    /// The generated spellings are `whisper/tokenizer.py`'s `specials` list.
    #[test]
    fn test_special_names() {
        for e in &EXPECTED {
            let policy =
                WhisperTokenLayout::new(WhisperSpecialIds::new(e.n_base, e.num_languages).unwrap());
            let ids = *policy.ids();
            let names: Vec<String> = policy.special_names().collect();
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

    /// A layout that is not upstream's: the names and the grid follow it,
    /// and the ids still fall in the same roles.
    #[test]
    fn test_custom_layout() {
        let layout = WhisperTokenLayoutConfig::new()
            .with_languages(vec!["xx".to_string(), "yy".to_string()])
            .with_control_tokens(
                CONTROL_TOKENS
                    .iter()
                    .map(|name| name.replace("<|", "<").replace("|>", ">"))
                    .collect(),
            )
            .with_timestamp_tokens(11)
            .with_timestamp_step_seconds(0.5);
        let policy = layout.policy(5, 2).unwrap();
        let ids = *policy.ids();

        assert_eq!(ids.n_vocab(), 5 + 2 + 2 + 6 + 11);
        assert!(ids.is_multilingual(), "5 is not the English-only base");
        assert_eq!(policy.language_token("yy"), Some(ids.language_begin + 1));
        assert_eq!(policy.language_code(ids.language_begin), Some("xx"));
        assert_eq!(policy.timestamp_seconds(ids.timestamp_begin + 3), Some(1.5));
        assert_eq!(policy.timestamp_token(11), None);

        let names: Vec<String> = policy.special_names().collect();
        assert_eq!(names[(ids.translate - ids.eot) as usize], "<translate>");
        assert_eq!(names.last().map(String::as_str), Some("<|5.00|>"));

        assert!(layout.special_ids(5, 3).is_err(), "only two languages");
        assert!(
            layout
                .clone()
                .with_control_tokens(Vec::new())
                .special_ids(5, 1)
                .is_err(),
            "a role is missing"
        );
    }

    #[test]
    fn test_policy_predicates() {
        let policy = WhisperTokenLayout::from_vocab_size(51865).unwrap();
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
        let multilingual = WhisperTokenLayout::from_vocab_size(51865).unwrap();

        assert_eq!(
            multilingual
                .sot_sequence(Some("en"), Some(WhisperTask::Transcribe), true)
                .unwrap(),
            [50258, 50259, 50359],
        );
        assert_eq!(
            multilingual
                .sot_sequence(Some("en"), Some(WhisperTask::Transcribe), false)
                .unwrap(),
            [50258, 50259, 50359, 50363],
        );
        assert_eq!(
            multilingual
                .sot_sequence(Some("ja"), Some(WhisperTask::Translate), false)
                .unwrap(),
            [50258, 50266, 50358, 50363],
        );
        assert_eq!(
            multilingual.sot_sequence(None, None, true).unwrap(),
            [50258]
        );
        assert_eq!(
            multilingual
                .sot_sequence(None, Some(WhisperTask::Transcribe), true)
                .unwrap(),
            [50258, 50359],
        );
        assert!(multilingual.sot_sequence(Some("xx"), None, true).is_err());
        assert!(
            multilingual.sot_sequence(Some("yue"), None, true).is_err(),
            "yue needs the 100-language layout",
        );

        let english = WhisperTokenLayout::from_vocab_size(51864).unwrap();
        assert_eq!(english.sot_sequence(None, None, true).unwrap(), [50257]);
        assert_eq!(
            english.sot_sequence(None, None, false).unwrap(),
            [50257, 50362]
        );
        assert!(english.sot_sequence(Some("en"), None, true).is_err());
        assert!(
            english
                .sot_sequence(None, Some(WhisperTask::Transcribe), true)
                .is_err()
        );
    }

    /// Five base ranks, including an empty one, laid out with two languages.
    fn tiny() -> (TiktokenRanks, WhisperTokenLayout) {
        let ranks = TiktokenRanks::parse("b2s= 0\nIPCfjg== 1\niQ== 2\nIGRvbmU= 3\n= 4\n").unwrap();
        let policy = WhisperTokenLayout::new(WhisperSpecialIds::new(5, 2).unwrap());
        (ranks, policy)
    }

    #[test]
    fn test_token_spans() {
        let (ranks, token_layout) = tiny();
        let ids = *token_layout.ids();
        let spans = token_layout.token_spans(&ranks).unwrap();

        assert_eq!(spans.len(), ids.n_vocab());
        assert_eq!(spans[0], b"ok");
        assert_eq!(spans[4], b"", "the empty token is present");
        assert_eq!(spans[ids.eot as usize], b"<|endoftext|>");
        assert_eq!(spans[ids.language_begin as usize + 1], b"<|zh|>");
        assert_eq!(spans[ids.timestamp_begin as usize + 100], b"<|2.00|>");
        assert_eq!(spans[ids.timestamp_end() as usize], b"<|30.00|>");
    }

    #[test]
    fn test_token_spans_rejects_a_mismatched_layout() {
        let (ranks, _) = tiny();
        let wrong = WhisperTokenLayout::new(WhisperSpecialIds::new(4, 2).unwrap());

        let err = wrong.token_spans(&ranks).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err:?}");
    }

    #[cfg(feature = "tokenizer")]
    mod wordchipper {
        use super::*;
        use crate::kits::tokens::Detokenizer;

        /// Specials render as their spellings, straight from the layout.
        #[test]
        fn test_detokenizer_renders_specials() {
            let (ranks, token_layout) = tiny();
            let ids = *token_layout.ids();
            let detok = token_layout.detokenizer(&ranks).unwrap();
            assert_eq!(detok.vocab_size(), ids.n_vocab());

            let window = [
                ids.sot,
                ids.language_begin,
                ids.transcribe,
                ids.timestamp_begin,
                0,
                4,
                ids.timestamp_begin + 100,
                ids.eot,
            ];
            assert_eq!(
                detok.detokenize(&window).unwrap(),
                "<|startoftranscript|><|en|><|transcribe|><|0.00|>ok<|2.00|><|endoftext|>",
            );
            assert_eq!(detok.detokenize(&[0, 1, 2, 3]).unwrap(), "ok 🎉 done");
            assert!(detok.detokenize(&[ids.n_vocab() as i64]).is_err());
        }

        /// Against the real assets, with ids produced by `whisper.tokenizer`.
        #[cfg(feature = "whisper-weights")]
        mod bundled {
            use super::*;

            #[test]
            fn test_multilingual_vocabulary() {
                let ranks =
                    TiktokenRanks::load(bunsen_bundled_whisper::multilingual_tiktoken()).unwrap();
                assert_eq!(
                    ranks.len(),
                    WhisperTokenLayoutConfig::new().multilingual_base_ranks
                );
                assert_eq!(ranks.get(50256), Some(&b""[..]), "the empty token");
                assert_eq!(
                    ranks.get(50255),
                    Some("场".as_bytes()),
                    "rank 50255 is `5Zy6`"
                );

                // The layout the checkpoint implies is the layout the file has.
                let token_layout = WhisperTokenLayout::from_vocab_size(51865).unwrap();
                let detok = token_layout.detokenizer(&ranks).unwrap();
                assert_eq!(detok.vocab_size(), 51865);

                // Plain text, including non-ASCII.
                let sentence = [
                    15947, 1002, 11, 341, 307, 257, 1500, 295, 264, 1667, 15487, 303, 979, 19866,
                    3466, 220, 27311, 31348, 886, 13,
                ];
                assert_eq!(
                    detok.detokenize(&sentence).unwrap(),
                    "Hello world, this is a test of the naïve decoder — 日本語 too.",
                );

                // A codepoint split across two ids: 19034 is `b" \xf0\x9f\x8e"`
                // and 231 is `b"\x89"`.
                assert_eq!(
                    detok.detokenize(&[453, 19034, 231, 1096]).unwrap(),
                    "ok 🎉 done"
                );

                // A window as the model emits it, with and without its
                // specials.
                let window = [50258, 50259, 50359, 50364, 15947, 1002, 13, 50464, 50257];
                assert_eq!(
                    detok.detokenize(&window).unwrap(),
                    "<|startoftranscript|><|en|><|transcribe|><|0.00|>Hello world.<|2.00|><|endoftext|>",
                );
                assert_eq!(
                    detok.detokenize(&token_layout.text_ids(&window)).unwrap(),
                    "Hello world."
                );

                // Every id renders; none is a hole.
                let all: Vec<i64> = (0..51865).collect();
                assert!(detok.detokenize(&all).is_ok());
            }

            #[test]
            fn test_english_vocabulary() {
                let ranks = TiktokenRanks::load(bunsen_bundled_whisper::gpt2_tiktoken()).unwrap();
                assert_eq!(
                    ranks.len(),
                    WhisperTokenLayoutConfig::new().english_base_ranks
                );
                assert_eq!(ranks.get(50255), Some(&b" gazed"[..]));

                let policy = WhisperTokenLayout::from_vocab_size(51864).unwrap();
                let detok = policy
                    .load_detokenizer(bunsen_bundled_whisper::gpt2_tiktoken())
                    .unwrap();
                assert_eq!(detok.vocab_size(), 51864);

                // The same text, tokenized differently: this vocabulary splits
                // the party popper across three ids.
                assert_eq!(detok.detokenize(&[15496, 995, 13]).unwrap(), "Hello world.");
                assert_eq!(
                    detok.detokenize(&[482, 12520, 236, 231, 1760]).unwrap(),
                    "ok 🎉 done"
                );
                assert_eq!(
                    detok.detokenize(&[50257, 50362, 50363, 50256]).unwrap(),
                    "<|startoftranscript|><|notimestamps|><|0.00|><|endoftext|>",
                );

                // The multilingual layout does not fit this file.
                let wrong = WhisperTokenLayout::from_vocab_size(51865).unwrap();
                assert!(wrong.detokenizer(&ranks).is_err());
            }
        }
    }
}
