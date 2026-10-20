//! Word error rates over normalized text: of one transcript
//! ([`word_error_rate`], [`text_error_rate`]), and of a speaker-attributed
//! one ([`cp_word_error_rate`], CHiME-6's cpWER).
//!
//! A speech model emits **token ids**; judging them against a transcript
//! needs text, which is [`Detokenizer`](crate::kits::tokens::Detokenizer)'s
//! job, and then a comparison that forgives formatting, which is this
//! module's.

/// Normalizes a transcript for comparison: lowercase, punctuation stripped,
/// whitespace collapsed.
///
/// This is deliberately much smaller than Whisper's own `EnglishTextNormalizer`
/// — it does **not** reconcile spelled-out numbers with digits, or expand
/// contractions. A threshold set against this normalizer is therefore counting
/// some formatting differences as errors; see the callers' notes.
pub fn normalize_transcript(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

/// The word edit distance: the fewest substitutions, insertions and
/// deletions that turn `hypothesis` into `reference`.
pub fn word_errors(
    hypothesis: &[String],
    reference: &[String],
) -> usize {
    // Two-row Levenshtein: the full matrix is O(n*m) memory for no benefit.
    let mut prev: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut cur = vec![0usize; hypothesis.len() + 1];

    for (r, refr) in reference.iter().enumerate() {
        cur[0] = r + 1;
        for (h, hyp) in hypothesis.iter().enumerate() {
            let cost = usize::from(refr != hyp);
            cur[h + 1] = (prev[h] + cost).min(prev[h + 1] + 1).min(cur[h] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[hypothesis.len()]
}

/// Word error rate: edit distance over words, divided by the reference length.
///
/// `(substitutions + insertions + deletions) / reference_words`. Not bounded
/// by 1 — a hypothesis longer than the reference can exceed it.
///
/// Returns `0.0` when both are empty, and `1.0` when only the reference is.
///
/// # Arguments
/// * `hypothesis` - normalized words produced by the model.
/// * `reference` - normalized words from the transcript.
pub fn word_error_rate(
    hypothesis: &[String],
    reference: &[String],
) -> f64 {
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    word_errors(hypothesis, reference) as f64 / reference.len() as f64
}

/// Word error rate between two raw strings, normalized by
/// [`normalize_transcript`].
pub fn text_error_rate(
    hypothesis: &str,
    reference: &str,
) -> f64 {
    word_error_rate(
        &normalize_transcript(hypothesis),
        &normalize_transcript(reference),
    )
}

/// The concatenated minimum-permutation word error rate (cpWER, CHiME-6):
/// each speaker's words concatenated in order, reference and hypothesis
/// speakers mapped one to one to minimize the total word errors, a speaker
/// left unmapped costing all its words; divided by the reference's words.
///
/// Words are normalized as [`normalize_transcript`] does.
///
/// # Arguments
/// * `reference` - `(speaker, text)`, in time order.
/// * `hypothesis` - `(speaker, text)`, in time order.
pub fn cp_word_error_rate(
    reference: &[(String, String)],
    hypothesis: &[(String, String)],
) -> f64 {
    let by_speaker = |turns: &[(String, String)]| -> Vec<Vec<String>> {
        let mut names: Vec<&str> = Vec::new();
        let mut words: Vec<Vec<String>> = Vec::new();
        for (speaker, text) in turns {
            let k = match names.iter().position(|n| n == speaker) {
                Some(k) => k,
                None => {
                    names.push(speaker);
                    words.push(Vec::new());
                    names.len() - 1
                }
            };
            words[k].extend(normalize_transcript(text));
        }
        words
    };
    let refs = by_speaker(reference);
    let hyps = by_speaker(hypothesis);
    let total: usize = refs.iter().map(Vec::len).sum();
    let errors: Vec<Vec<usize>> = refs
        .iter()
        .map(|r| hyps.iter().map(|h| word_errors(h, r)).collect())
        .collect();

    /// The fewest errors mapping refs `i..` to unused hyps (or none).
    fn best(
        i: usize,
        used: &mut Vec<bool>,
        refs: &[Vec<String>],
        hyps: &[Vec<String>],
        errors: &[Vec<usize>],
    ) -> usize {
        if i == refs.len() {
            return hyps
                .iter()
                .zip(used.iter())
                .filter(|(_, u)| !**u)
                .map(|(h, _)| h.len())
                .sum();
        }
        let mut least = refs[i].len() + best(i + 1, used, refs, hyps, errors);
        for j in 0..hyps.len() {
            if !used[j] {
                used[j] = true;
                least = least.min(errors[i][j] + best(i + 1, used, refs, hyps, errors));
                used[j] = false;
            }
        }
        least
    }
    let errors = best(0, &mut vec![false; hyps.len()], &refs, &hyps, &errors);
    if total == 0 {
        return if errors == 0 { 0.0 } else { f64::INFINITY };
    }
    errors as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        normalize_transcript(s)
    }

    #[test]
    fn test_normalize_strips_case_and_punctuation() {
        assert_eq!(
            words("  The Moon?? Why -- choose  this,  as our goal! "),
            ["the", "moon", "why", "choose", "this", "as", "our", "goal"],
        );
        // Apostrophes are kept: "we're" and "were" are different words.
        assert_eq!(words("we're"), ["we're"]);
    }

    #[test]
    fn test_wer_counts_each_edit_once() {
        let refr = words("we choose to go to the moon");

        assert_eq!(word_error_rate(&refr, &refr), 0.0);

        // One substitution out of seven.
        let sub = words("we choose to go to the sun");
        assert!((word_error_rate(&sub, &refr) - 1.0 / 7.0).abs() < 1e-12);

        // One deletion out of seven.
        let del = words("we choose to go to moon");
        assert!((word_error_rate(&del, &refr) - 1.0 / 7.0).abs() < 1e-12);

        // One insertion out of seven.
        let ins = words("we choose to go to the big moon");
        assert!((word_error_rate(&ins, &refr) - 1.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_wer_edge_cases() {
        assert_eq!(word_error_rate(&[], &[]), 0.0);
        assert_eq!(word_error_rate(&words("anything"), &[]), 1.0);
        // Nothing recognized at all is a full miss.
        assert_eq!(word_error_rate(&[], &words("we choose to go")), 1.0);
    }

    /// cpWER by hand: the mapping that makes the speakers agree best,
    /// unmapped speakers costing every word.
    #[test]
    fn test_cp_word_error_rate() {
        let t = |s: &str, x: &str| (s.to_string(), x.to_string());
        let reference = [t("A", "we choose"), t("B", "to go"), t("A", "to the moon")];

        // The same words under other names: no error.
        let renamed = [t("x", "we choose"), t("y", "to go"), t("x", "to the moon")];
        assert_eq!(cp_word_error_rate(&reference, &renamed), 0.0);

        // Speaker B's words given to A: missing from B and inserted into A,
        // two errors each.
        let merged = [t("x", "we choose to go to the moon")];
        assert!((cp_word_error_rate(&reference, &merged) - 4.0 / 7.0).abs() < 1e-12);

        // A third hypothesis speaker costs all its words.
        let extra = [
            t("x", "we choose to the moon"),
            t("y", "to go"),
            t("z", "applause"),
        ];
        assert!((cp_word_error_rate(&reference, &extra) - 1.0 / 7.0).abs() < 1e-12);

        assert_eq!(cp_word_error_rate(&[], &[]), 0.0);
    }
}
