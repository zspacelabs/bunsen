//! Speech-recognition test helpers: word error rate over normalized text.
//!
//! bunsen's Whisper emits **token ids**; judging them against a transcript
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

    prev[hypothesis.len()] as f64 / reference.len() as f64
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
}
