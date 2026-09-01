//! # Logit filters: what the search may not pick.
//!
//! A filter rewrites the last position's logits before the search looks at
//! them, and it is consulted every step. Upstream applies two by default:
//! [`SuppressTokens`] over its non-speech list and the control tokens, and
//! [`SuppressBlank`] at the first sampled position. Both need the
//! vocabulary to know which ids they mean, so [`default_filters`] derives
//! them from the rank file and the layout &mdash; decode-only, without an
//! encoder: the symbols upstream encodes are either single tokens, which is
//! an exact byte match, or the seven music symbols, whose first byte-level
//! BPE token is the longest prefix of their bytes that is a token.
//!
//! The timestamp rules are a filter too, and arrive with timestamps.

use std::{
    collections::HashMap,
    fmt::Debug,
    sync::Arc,
};

use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
        TensorData,
    },
};

use crate::kits::speech::whisper::{
    tokens::WhisperSpecialIds,
    vocab::TiktokenRanks,
};

/// Rewrites the logits the search chooses from.
pub trait LogitFilter<B: Backend>: Send + Sync + Debug {
    /// Rewrites `logits`, `[rows, vocab]`, the last position's, given the
    /// sequences so far and how long their prompt is.
    ///
    /// # Arguments
    /// * `logits` - `[rows, vocab]`.
    /// * `tokens` - one sequence per row, prompt included.
    /// * `prompt_len` - the length of the prompt every row began with, so a
    ///   filter can tell the first sampled position.
    fn apply(
        &self,
        logits: Tensor<B, 2>,
        tokens: &[Vec<i64>],
        prompt_len: usize,
    ) -> Tensor<B, 2>;
}

/// Sets the given ids to `-inf` in every row.
fn suppress<B: Backend>(
    logits: Tensor<B, 2>,
    ids: &[i64],
) -> Tensor<B, 2> {
    let [rows, vocab] = logits.dims();
    let mut mask = vec![false; vocab];
    for &id in ids {
        if let Ok(i) = usize::try_from(id)
            && i < vocab
        {
            mask[i] = true;
        }
    }
    let mask: Tensor<B, 1, Bool> =
        Tensor::from_data(TensorData::new(mask, [vocab]), &logits.device());
    logits.mask_fill(
        mask.unsqueeze::<2>().expand([rows, vocab]),
        f32::NEG_INFINITY,
    )
}

/// Never picks these ids.
#[derive(Debug, Clone)]
pub struct SuppressTokens {
    ids: Vec<i64>,
}

impl SuppressTokens {
    /// Suppresses `ids`, in any order, duplicates allowed.
    pub fn new(ids: impl IntoIterator<Item = i64>) -> Self {
        let mut ids: Vec<i64> = ids.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        Self { ids }
    }

    /// The suppressed ids, sorted.
    pub fn ids(&self) -> &[i64] {
        &self.ids
    }
}

impl<B: Backend> LogitFilter<B> for SuppressTokens {
    fn apply(
        &self,
        logits: Tensor<B, 2>,
        _tokens: &[Vec<i64>],
        _prompt_len: usize,
    ) -> Tensor<B, 2> {
        suppress(logits, &self.ids)
    }
}

/// Never opens a transcript with a space or with nothing: at the first
/// sampled position, suppresses the blank token and `<|endoftext|>`.
#[derive(Debug, Clone)]
pub struct SuppressBlank {
    ids: Vec<i64>,
}

impl SuppressBlank {
    /// # Arguments
    /// * `blank` - the id of the single-space token.
    /// * `eot` - `<|endoftext|>`.
    pub fn new(
        blank: i64,
        eot: i64,
    ) -> Self {
        Self {
            ids: vec![blank, eot],
        }
    }
}

impl<B: Backend> LogitFilter<B> for SuppressBlank {
    fn apply(
        &self,
        logits: Tensor<B, 2>,
        tokens: &[Vec<i64>],
        prompt_len: usize,
    ) -> Tensor<B, 2> {
        if tokens.first().is_some_and(|t| t.len() == prompt_len) {
            suppress(logits, &self.ids)
        } else {
            logits
        }
    }
}

/// The symbols upstream's `non_speech_tokens` suppresses when they are a
/// single token, with or without a leading space.
const SYMBOLS: &[&str] = &[
    "\"",
    "#",
    "(",
    ")",
    "*",
    "+",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
    "\u{300c}",
    "\u{300d}",
    "\u{300e}",
    "\u{300f}",
    "<<",
    ">>",
    "<<<",
    ">>>",
    "--",
    "---",
    "-(",
    "-[",
    "('",
    "(\"",
    "((",
    "))",
    "(((",
    ")))",
    "[[",
    "]]",
    "{{",
    "}}",
    "\u{266a}\u{266a}",
    "\u{266a}\u{266a}\u{266a}",
];

/// The music symbols, suppressed by their first token whatever it is.
const MISCELLANEOUS: &[&str] = &[
    "\u{2669}", "\u{266a}", "\u{266b}", "\u{266c}", "\u{266d}", "\u{266e}", "\u{266f}",
];

/// The rank file as `{ bytes -> id }`.
fn lookup(ranks: &TiktokenRanks) -> HashMap<&[u8], i64> {
    ranks
        .iter()
        .enumerate()
        .map(|(id, bytes)| (bytes, id as i64))
        .collect()
}

/// The id of the longest prefix of `bytes` that is a token: what byte-level
/// BPE emits first for a string that is not itself a token.
fn first_token(
    table: &HashMap<&[u8], i64>,
    bytes: &[u8],
) -> Option<i64> {
    (1..=bytes.len())
        .rev()
        .find_map(|n| table.get(&bytes[..n]).copied())
}

/// Upstream's `Tokenizer.non_speech_tokens`, from the rank file alone:
/// the ids that would make a transcript say `[APPLAUSE]` or draw a music
/// note, plus the leading `-` and `'` that would start a word with one.
pub fn non_speech_tokens(ranks: &TiktokenRanks) -> Vec<i64> {
    let table = lookup(ranks);
    let mut ids: Vec<i64> = [" -", " '"]
        .iter()
        .filter_map(|s| table.get(s.as_bytes()).copied())
        .collect();

    for symbol in SYMBOLS {
        for candidate in [symbol.to_string(), format!(" {symbol}")] {
            if let Some(&id) = table.get(candidate.as_bytes()) {
                ids.push(id);
            }
        }
    }
    for symbol in MISCELLANEOUS {
        for candidate in [symbol.to_string(), format!(" {symbol}")] {
            if let Some(id) = first_token(&table, candidate.as_bytes()) {
                ids.push(id);
            }
        }
    }

    ids.sort_unstable();
    ids.dedup();
    ids
}

/// The id of the single-space token, which opens a blank transcript.
pub fn blank_token(ranks: &TiktokenRanks) -> Option<i64> {
    lookup(ranks).get(b" ".as_slice()).copied()
}

/// Upstream's default suppress list: the non-speech tokens, the control
/// tokens a decode must never emit, and `<|nospeech|>`, whose probability
/// is read separately.
pub fn default_suppress_tokens(
    ranks: &TiktokenRanks,
    ids: &WhisperSpecialIds,
) -> Vec<i64> {
    let mut all = non_speech_tokens(ranks);
    all.extend([
        ids.transcribe,
        ids.translate,
        ids.sot,
        ids.sot_prev,
        ids.sot_lm,
        ids.no_speech,
    ]);
    all.sort_unstable();
    all.dedup();
    all
}

/// The two filters upstream applies by default, in its order.
///
/// # Panics
/// If the vocabulary has no single-space token, which no Whisper vocabulary
/// lacks.
pub fn default_filters<B: Backend>(
    ranks: &TiktokenRanks,
    ids: &WhisperSpecialIds,
) -> Vec<Arc<dyn LogitFilter<B>>> {
    let blank = blank_token(ranks).expect("the vocabulary has a space token");
    vec![
        Arc::new(SuppressBlank::new(blank, ids.eot)),
        Arc::new(SuppressTokens::new(default_suppress_tokens(ranks, ids))),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;

    fn logits(rows: &[&[f32]]) -> Tensor<B, 2> {
        let vocab = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::from_data(
            TensorData::new(flat, [rows.len(), vocab]),
            &Default::default(),
        )
    }

    fn to_rows(t: Tensor<B, 2>) -> Vec<Vec<f32>> {
        let [rows, vocab] = t.dims();
        let flat = t.to_data().convert::<f32>().to_vec::<f32>().unwrap();
        flat.chunks(vocab).map(|c| c.to_vec()).collect::<Vec<_>>()[..rows].to_vec()
    }

    #[test]
    fn test_suppress_tokens() {
        let filter = SuppressTokens::new([3, 1, 3, 99, -1]);
        assert_eq!(filter.ids(), &[-1, 1, 3, 99], "sorted, deduplicated");

        let out = to_rows(LogitFilter::<B>::apply(
            &filter,
            logits(&[&[0.0, 1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0, 9.0]]),
            &[vec![7], vec![7]],
            1,
        ));
        assert_eq!(
            out[0],
            vec![0.0, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY, 4.0]
        );
        assert_eq!(
            out[1],
            vec![5.0, f32::NEG_INFINITY, 7.0, f32::NEG_INFINITY, 9.0]
        );
    }

    /// Only at the first sampled position: once anything has been sampled,
    /// the blank and the stop token are allowed again.
    #[test]
    fn test_suppress_blank() {
        let filter = SuppressBlank::new(2, 4);
        let first = to_rows(LogitFilter::<B>::apply(
            &filter,
            logits(&[&[0.0, 1.0, 2.0, 3.0, 4.0]]),
            &[vec![9, 9]],
            2,
        ));
        assert_eq!(
            first[0],
            vec![0.0, 1.0, f32::NEG_INFINITY, 3.0, f32::NEG_INFINITY]
        );

        let later = to_rows(LogitFilter::<B>::apply(
            &filter,
            logits(&[&[0.0, 1.0, 2.0, 3.0, 4.0]]),
            &[vec![9, 9, 1]],
            2,
        ));
        assert_eq!(later[0], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    /// The derivation on a toy vocabulary: exact matches for the symbols,
    /// the longest token prefix for a music symbol.
    #[test]
    fn test_non_speech_from_ranks() {
        // ranks: 0 "a", 1 "(", 2 " (", 3 " -", 4 "\xe2\x99" (the prefix the
        // music symbols share), 5 " ", 6 "<<", 7 "\xe2\x99\xaa" (a whole
        // note).
        let ranks = TiktokenRanks::parse(
            "YQ== 0\nKA== 1\nICg= 2\nIC0= 3\n4pk= 4\nIA== 5\nPDw= 6\n4pmq 7\n",
        )
        .unwrap();
        assert_eq!(blank_token(&ranks), Some(5));
        // "(" and " (" exactly; " -" exactly; "<<" exactly; the whole note
        // exactly, and every other music symbol by its shared prefix. With
        // a leading space none of the music symbols has a token prefix
        // beyond the space itself, so their first token is the blank, 5 —
        // as upstream's `encode(" ♭")[0]` would be in such a vocabulary.
        assert_eq!(non_speech_tokens(&ranks), vec![1, 2, 3, 4, 5, 6, 7]);
    }

    /// Against the real vocabularies: upstream's own `non_speech_tokens`
    /// and `encode(" ")`, read off `whisper.tokenizer`.
    #[cfg(feature = "whisper-weights")]
    mod bundled {
        use super::*;
        use crate::kits::speech::whisper::tokens::TokenPolicy;

        #[test]
        fn test_multilingual_matches_upstream() {
            let ranks =
                TiktokenRanks::load(bunsen_bundled_whisper::multilingual_tiktoken()).unwrap();
            assert_eq!(blank_token(&ranks), Some(220));
            assert_eq!(
                non_speech_tokens(&ranks),
                vec![
                    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91,
                    92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982,
                    2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273,
                    9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635,
                    15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130, 26161,
                    26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254,
                ]
            );

            let ids = *TokenPolicy::from_vocab_size(51865).unwrap().ids();
            let all = default_suppress_tokens(&ranks, &ids);
            assert!(all.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
            for id in [
                ids.transcribe,
                ids.translate,
                ids.sot,
                ids.sot_prev,
                ids.sot_lm,
                ids.no_speech,
            ] {
                assert!(all.contains(&id));
            }
            assert_eq!(all.len(), 82 + 6);
            assert_eq!(default_filters::<B>(&ranks, &ids).len(), 2);
        }

        #[test]
        fn test_english_matches_upstream() {
            let ranks = TiktokenRanks::load(bunsen_bundled_whisper::gpt2_tiktoken()).unwrap();
            assert_eq!(blank_token(&ranks), Some(220));
            assert_eq!(
                non_speech_tokens(&ranks),
                vec![
                    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91,
                    92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303,
                    1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600,
                    4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907,
                    13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724,
                    22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282,
                    49146,
                ]
            );
        }
    }
}
