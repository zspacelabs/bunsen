//! # Ranking: which finished candidate a decode returns.
//!
//! A beam search ends with several finished sequences per audio; the ranker
//! picks one. Upstream's is the cumulative log probability normalized by
//! length, either plainly or with the Google NMT penalty, and that is the
//! one here.

use std::fmt::Debug;

/// Picks one candidate per audio.
pub trait SequenceRanker: Send + Sync + Debug {
    /// The index of the winner among `candidates`, each a generated
    /// sequence (prompt and stop token excluded) with its cumulative log
    /// probability.
    fn rank(
        &self,
        candidates: &[(Vec<i64>, f32)],
    ) -> usize;
}

/// The highest log probability per unit of length.
///
/// With `length_penalty` unset the penalty is the length itself; set, it is
/// `((5 + length) / 6) ^ length_penalty`, from the Google NMT paper.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MaximumLikelihoodRanker {
    /// The exponent of the NMT penalty, or `None` for plain normalization.
    pub length_penalty: Option<f64>,
}

impl MaximumLikelihoodRanker {
    /// The penalty a sequence of `length` tokens is divided by.
    pub fn penalty(
        &self,
        length: usize,
    ) -> f64 {
        match self.length_penalty {
            None => length as f64,
            Some(alpha) => ((5.0 + length as f64) / 6.0).powf(alpha),
        }
    }
}

impl SequenceRanker for MaximumLikelihoodRanker {
    fn rank(
        &self,
        candidates: &[(Vec<i64>, f32)],
    ) -> usize {
        assert!(!candidates.is_empty(), "nothing to rank");

        // The first of equals, as `argmax` picks.
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, (tokens, logprob)) in candidates.iter().enumerate() {
            let score = f64::from(*logprob) / self.penalty(tokens.len());
            if score > best_score {
                best = i;
                best_score = score;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_normalization_prefers_probability_per_token() {
        let ranker = MaximumLikelihoodRanker::default();
        // -3 over 3 tokens is -1 each; -2.5 over 2 is -1.25 each.
        let candidates = vec![(vec![1, 2, 3], -3.0), (vec![1, 2], -2.5)];
        assert_eq!(ranker.rank(&candidates), 0);
        assert_eq!(ranker.penalty(3), 3.0);
    }

    #[test]
    fn test_nmt_penalty() {
        let ranker = MaximumLikelihoodRanker {
            length_penalty: Some(1.0),
        };
        assert!((ranker.penalty(1) - 1.0).abs() < 1e-12);
        assert!((ranker.penalty(7) - 2.0).abs() < 1e-12);

        // Under the NMT penalty a longer sequence is penalized more gently
        // than by its raw length, so the same pair flips.
        let candidates = vec![(vec![1, 2, 3], -3.0), (vec![1, 2], -2.5)];
        assert_eq!(ranker.rank(&candidates), 1);
    }

    #[test]
    fn test_first_of_equals() {
        let ranker = MaximumLikelihoodRanker::default();
        let candidates = vec![(vec![1], -1.0), (vec![2], -1.0), (vec![3], -1.0)];
        assert_eq!(ranker.rank(&candidates), 0);
    }

    /// An empty candidate has zero length; plain normalization would divide
    /// by zero and must not pick it over a real one by accident.
    #[test]
    fn test_empty_candidate() {
        let ranker = MaximumLikelihoodRanker::default();
        // -0.0 / 0 is NaN, which never compares greater; -1 / 1 wins.
        let candidates = vec![(vec![], -0.0), (vec![5], -1.0)];
        assert_eq!(ranker.rank(&candidates), 1);
    }
}
