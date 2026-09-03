//! # Logit filters.

use std::{
    fmt::Debug,
    ops::Range,
    sync::Arc,
};

use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
        TensorData,
    },
    tensor::activation::log_softmax,
};

use crate::kits::{
    speech::whisper::driver::WhisperSpecialIds,
    tokens,
    tokens::TiktokenRanks,
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

/// Upstream's default suppress list: the non-speech tokens, the control
/// tokens a decode must never emit, and `<|nospeech|>`, whose probability
/// is read separately.
pub fn default_suppress_tokens(
    ranks: &TiktokenRanks,
    ids: &WhisperSpecialIds,
) -> Vec<i64> {
    let mut all = tokens::non_speech_tokens(ranks);
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
    let blank = tokens::blank_token(ranks).expect("the vocabulary has a space token");
    vec![
        Arc::new(SuppressBlank::new(blank, ids.eot)),
        Arc::new(SuppressTokens::new(default_suppress_tokens(ranks, ids))),
    ]
}

/// Sets, per row, the id ranges `ranges[row]` to `-inf`.
fn suppress_rows<B: Backend>(
    logits: Tensor<B, 2>,
    ranges: &[Vec<Range<usize>>],
) -> Tensor<B, 2> {
    let [rows, vocab] = logits.dims();
    if ranges.iter().all(|r| r.is_empty()) {
        return logits;
    }
    let mut mask = vec![false; rows * vocab];
    for (row, ranges) in ranges.iter().enumerate() {
        for range in ranges {
            let (a, b) = (range.start.min(vocab), range.end.min(vocab));
            mask[row * vocab + a..row * vocab + b].fill(true);
        }
    }
    let mask: Tensor<B, 2, Bool> =
        Tensor::from_data(TensorData::new(mask, [rows, vocab]), &logits.device());
    logits.mask_fill(mask, f32::NEG_INFINITY)
}

/// Upstream's `ApplyTimestampRules`: the grammar of timestamp tokens.
///
/// Over the sampled history of a row: timestamps come in pairs (a start,
/// text, an end) except that a single timestamp may end the transcript;
/// they never decrease, and a segment is never empty; the first sampled
/// token is a timestamp, no later than `max_initial_timestamp_index`; and
/// `<|notimestamps|>` is never emitted. Over the logits: when the
/// probability mass on timestamps as a whole exceeds that of the likeliest
/// text token, only a timestamp may follow.
#[derive(Debug, Clone)]
pub struct ApplyTimestampRules {
    eot: i64,
    no_timestamps: i64,
    timestamp_begin: i64,
    max_initial_timestamp_index: Option<usize>,
}

impl ApplyTimestampRules {
    /// # Arguments
    /// * `ids` - the token layout.
    /// * `max_initial_timestamp_index` - the latest index the first timestamp
    ///   may have; upstream's default is one second, index 50.
    pub fn new(
        ids: &WhisperSpecialIds,
        max_initial_timestamp_index: Option<usize>,
    ) -> Self {
        Self {
            eot: ids.eot,
            no_timestamps: ids.no_timestamps,
            timestamp_begin: ids.timestamp_begin,
            max_initial_timestamp_index,
        }
    }

    /// The id ranges the history `sampled` (the row's tokens after the
    /// prompt) forbids next: every clause but the probability one, as a
    /// pure function.
    ///
    /// # Arguments
    /// * `sampled` - the tokens sampled so far in this row.
    /// * `first` - whether this is the first sampled position.
    pub fn forbidden(
        &self,
        sampled: &[i64],
        first: bool,
    ) -> Vec<Range<usize>> {
        let tb = self.timestamp_begin;
        let is_ts = |t: i64| t >= tb;
        let n = sampled.len();
        let mut out = Vec::with_capacity(4);

        // <|notimestamps|> is handled by the prompt, never sampled.
        let nt = self.no_timestamps as usize;
        out.push(nt..nt + 1);

        // Timestamps come in pairs, except directly before EOT.
        let last_was_timestamp = n >= 1 && is_ts(sampled[n - 1]);
        let penultimate_was_timestamp = n < 2 || is_ts(sampled[n - 2]);
        if last_was_timestamp {
            if penultimate_was_timestamp {
                // Has to be non-timestamp.
                out.push(tb as usize..usize::MAX);
            } else {
                // Cannot be normal text.
                out.push(0..self.eot as usize);
            }
        }

        // Timestamps never decrease, and a segment is never empty.
        if let Some(&last) = sampled.iter().rev().find(|&&t| is_ts(t)) {
            let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                last
            } else {
                last + 1
            };
            out.push(tb as usize..timestamp_last as usize);
        }

        // The first sampled token is a timestamp, and not a late one.
        if first {
            out.push(0..tb as usize);
            if let Some(max) = self.max_initial_timestamp_index {
                let last_allowed = tb as usize + max;
                out.push(last_allowed + 1..usize::MAX);
            }
        }

        out
    }
}

impl<B: Backend> LogitFilter<B> for ApplyTimestampRules {
    fn apply(
        &self,
        logits: Tensor<B, 2>,
        tokens: &[Vec<i64>],
        prompt_len: usize,
    ) -> Tensor<B, 2> {
        let [rows, vocab] = logits.dims();
        let ranges: Vec<Vec<Range<usize>>> = tokens
            .iter()
            .map(|t| self.forbidden(&t[prompt_len..], t.len() == prompt_len))
            .collect();
        let logits = suppress_rows(logits, &ranges);

        // If the probability mass on timestamps beats every text token,
        // sample a timestamp.
        let tb = (self.timestamp_begin as usize).min(vocab);
        if tb == 0 || tb == vocab {
            return logits;
        }
        let logprobs = log_softmax(logits.clone(), 1);
        let timestamps = logprobs.clone().slice_dim(1, tb as isize..vocab as isize);
        let peak = timestamps.clone().max_dim(1);
        let timestamp_logprob = (timestamps - peak.clone()).exp().sum_dim(1).log() + peak;
        let max_text_logprob = logprobs.slice_dim(1, 0..tb as isize).max_dim(1);
        // Compared on the host: a Bool tensor's storage differs by
        // backend, and two floats per row are nothing to move.
        let timestamp_logprob: Vec<f32> = timestamp_logprob
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let max_text_logprob: Vec<f32> = max_text_logprob
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        let prefer: Vec<bool> = timestamp_logprob
            .iter()
            .zip(&max_text_logprob)
            .map(|(ts, text)| ts > text)
            .collect();

        let ranges: Vec<Vec<Range<usize>>> = (0..rows)
            .map(|row| {
                if prefer[row] {
                    std::iter::once(0..tb).collect()
                } else {
                    Vec::new()
                }
            })
            .collect();
        suppress_rows(logits, &ranges)
    }
}

/// Language detection: nothing but the language block may be chosen.
///
/// Upstream's `detect_language` is one decoder step over
/// `<|startoftranscript|>` with every other id masked; as a filter over a
/// one-token decode it is the same step.
#[derive(Debug, Clone)]
pub struct RestrictToLanguages {
    begin: usize,
    count: usize,
}

impl RestrictToLanguages {
    /// # Panics
    /// If the layout has no language block.
    pub fn new(ids: &WhisperSpecialIds) -> Self {
        assert!(
            ids.is_multilingual(),
            "an English-only layout has no languages to detect"
        );
        Self {
            begin: ids.language_begin as usize,
            count: ids.num_languages,
        }
    }
}

impl<B: Backend> LogitFilter<B> for RestrictToLanguages {
    fn apply(
        &self,
        logits: Tensor<B, 2>,
        tokens: &[Vec<i64>],
        _prompt_len: usize,
    ) -> Tensor<B, 2> {
        let ranges = vec![vec![0..self.begin, self.begin + self.count..usize::MAX]; tokens.len()];
        suppress_rows(logits, &ranges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kits::tokens::{
            blank_token,
            non_speech_tokens,
        },
        support::testing::CpuBackend,
    };

    type B = CpuBackend;

    fn logits<B: Backend>(
        rows: &[&[f32]],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let vocab = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::from_data(TensorData::new(flat, [rows.len(), vocab]), device)
    }

    fn to_rows<B: Backend>(t: Tensor<B, 2>) -> Vec<Vec<f32>> {
        let [rows, vocab] = t.dims();
        let flat = t.to_data().convert::<f32>().to_vec::<f32>().unwrap();
        flat.chunks(vocab).map(|c| c.to_vec()).collect::<Vec<_>>()[..rows].to_vec()
    }

    #[test]
    fn test_suppress_tokens() {
        let device = Default::default();

        let filter = SuppressTokens::new([3, 1, 3, 99, -1]);
        assert_eq!(filter.ids(), &[-1, 1, 3, 99], "sorted, deduplicated");

        let out = to_rows(LogitFilter::<B>::apply(
            &filter,
            logits::<B>(
                &[&[0.0, 1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0, 9.0]],
                &device,
            ),
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

    /// A small layout for the timestamp grammar: eot 5, no_timestamps 13,
    /// timestamps from 14; a vocabulary of 20 leaves six timestamp ids.
    fn layout() -> WhisperSpecialIds {
        WhisperSpecialIds::new(5, 1).unwrap()
    }

    const NEG: f32 = f32::NEG_INFINITY;

    /// Every clause of the history rules, by hand, with no model in the
    /// loop.
    #[test]
    fn test_timestamp_rules_over_history() {
        let ids = layout();
        let (eot, nt, tb) = (
            ids.eot as usize,
            ids.no_timestamps as usize,
            ids.timestamp_begin,
        );
        assert_eq!((eot, nt, tb), (5, 13, 14));
        let rules = ApplyTimestampRules::new(&ids, Some(2));
        let ts = |i: i64| tb + i;

        // First position: only a timestamp, no later than index 2.
        assert_eq!(
            rules.forbidden(&[], true),
            vec![nt..nt + 1, 0..tb as usize, (tb as usize + 3)..usize::MAX]
        );
        // No cap.
        assert_eq!(
            ApplyTimestampRules::new(&ids, None).forbidden(&[], true),
            vec![nt..nt + 1, 0..tb as usize]
        );

        // After a lone opening timestamp: must be non-timestamp, and later
        // timestamps must exceed it (the segment is never empty).
        assert_eq!(
            rules.forbidden(&[ts(1)], false),
            vec![
                nt..nt + 1,
                tb as usize..usize::MAX,
                tb as usize..ts(2) as usize
            ]
        );

        // Text after the opening: anything but a timestamp before ts(2).
        assert_eq!(
            rules.forbidden(&[ts(1), 3], false),
            vec![nt..nt + 1, tb as usize..ts(2) as usize]
        );

        // A closing timestamp: no text may follow (a timestamp or eot), and
        // the next start may equal the close.
        assert_eq!(
            rules.forbidden(&[ts(1), 3, ts(4)], false),
            vec![nt..nt + 1, 0..eot, tb as usize..ts(4) as usize]
        );

        // A pair of timestamps just closed and reopened: text only, and
        // strictly increasing from here.
        assert_eq!(
            rules.forbidden(&[ts(1), 3, ts(4), ts(4)], false),
            vec![
                nt..nt + 1,
                tb as usize..usize::MAX,
                tb as usize..ts(5) as usize
            ]
        );

        // Text with no timestamp yet (a prompted continuation): only
        // <|notimestamps|> is out.
        assert_eq!(rules.forbidden(&[3, 4], false), vec![nt..nt + 1]);
    }

    /// The probability clause, on tensors: when the timestamp mass beats
    /// the best text token the text goes, row by row.
    #[test]
    fn test_timestamp_rules_probability_clause() {
        let device = Default::default();

        let ids = layout();
        let rules = ApplyTimestampRules::new(&ids, None);
        let (nt, tb) = (ids.no_timestamps as usize, ids.timestamp_begin as usize);
        let vocab = 20;
        // Row 0: one strong text token; timestamps weak. Row 1: text flat
        // and weak, timestamps each modest but six of them: their sum
        // wins. Both rows are mid-transcript with a text token last, so
        // only <|notimestamps|> and the first timestamp (already used, and
        // a segment is never empty) are forbidden by history.
        let mut row0 = vec![0.0f32; vocab];
        row0[3] = 8.0;
        let mut row1 = vec![0.0f32; vocab];
        for t in tb..vocab {
            row1[t] = 1.0;
        }
        let rows: Vec<&[f32]> = vec![&row0, &row1];
        let out = to_rows(LogitFilter::<B>::apply(
            &rules,
            logits::<B>(&rows, &device),
            &[vec![9, tb as i64, 3], vec![9, tb as i64, 3]],
            1,
        ));

        assert_eq!(out[0][nt], NEG, "<|notimestamps|> always");
        assert_eq!(out[0][tb], NEG, "before the last timestamp + 1");
        assert_eq!(out[0][3], 8.0, "row 0 keeps its text");
        assert_eq!(out[0][tb + 1], 0.0);

        assert!(
            out[1][..tb].iter().all(|&v| v == NEG),
            "row 1 loses all text: {:?}",
            out[1]
        );
        assert_eq!(out[1][tb + 1], 1.0);
        assert_eq!(out[1][tb], NEG, "history still applies");
    }

    /// Detection leaves only the language block standing.
    #[test]
    fn test_restrict_to_languages() {
        let device = Default::default();

        let ids = WhisperSpecialIds::new(5, 3).unwrap();
        let filter = RestrictToLanguages::new(&ids);
        let vocab = ids.n_vocab();
        let row: Vec<f32> = (0..vocab).map(|i| i as f32).collect();
        let out = to_rows(LogitFilter::<B>::apply(
            &filter,
            logits::<B>(&[&row], &device),
            &[vec![6]],
            1,
        ));
        let begin = ids.language_begin as usize;
        for (i, v) in out[0].iter().enumerate() {
            if (begin..begin + 3).contains(&i) {
                assert_eq!(*v, i as f32);
            } else {
                assert_eq!(*v, NEG, "id {i}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "no languages to detect")]
    fn test_restrict_to_languages_needs_a_block() {
        let _ = RestrictToLanguages::new(&WhisperSpecialIds::from_vocab_size(51864).unwrap());
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
        use crate::kits::speech::whisper::driver::WhisperTokenLayout;

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

            let ids = *WhisperTokenLayout::from_vocab_size(51865).unwrap().ids();
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
