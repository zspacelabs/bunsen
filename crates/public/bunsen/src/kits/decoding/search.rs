//! The search seam, and the value object that picks a search.
use std::fmt::Debug;

use burn::{
    Tensor,
    config::Config,
};

use super::{
    BeamSearch,
    GreedySearch,
    MaximumLikelihoodRanker,
};

/// The search: what the next token of every row is, given the logits.
///
/// Rows are laid out `row = group * group_size + member`: a group is one
/// prompt being decoded, a greedy search has one member per group, a beam
/// search `k`, a sampled search one per trajectory. The decode loop owns the
/// tensors and the model's cache; the search owns the sequences' bookkeeping
/// and tells the loop, through `reorder`, how to permute the cache's rows
/// when its members branch.
pub trait TokenSearch: Send + Debug {
    /// Rows per group.
    fn group_size(&self) -> usize;

    /// Forgets any state from a previous decode.
    fn reset(&mut self);

    /// Consumes one step of logits.
    ///
    /// # Arguments
    /// * `tokens` - one sequence per row, prompt included; extended (or
    ///   replaced, as beams branch) in place.
    /// * `logits` - `[rows, vocab]`, the last position's, filtered.
    /// * `sum_logprobs` - one cumulative log probability per row, updated in
    ///   place.
    /// * `reorder` - permutes the cache: row `r` becomes row `sources[r]`.
    ///
    /// # Returns
    /// The token to feed each row next, and whether the search is complete.
    fn update(
        &mut self,
        tokens: &mut Vec<Vec<i64>>,
        logits: Tensor<2>,
        sum_logprobs: &mut [f32],
        reorder: &mut dyn FnMut(&[usize]),
    ) -> (Vec<i64>, bool);

    /// The finished candidates of every group.
    fn finalize(
        &mut self,
        tokens: Vec<Vec<i64>>,
        sum_logprobs: Vec<f32>,
        prompt_len: usize,
    ) -> Vec<Vec<Candidate>>;
}

/// One finished sequence of a search: what it generated, how probable that
/// was, and what stopped it.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// The generated ids, after the prompt, without the stop token.
    pub tokens: Vec<i64>,

    /// The cumulative log probability of `tokens`.
    pub sum_logprob: f32,

    /// The stop token that ended the sequence; `None` when the decode's
    /// budget did.
    pub stop: Option<i64>,
}

impl Candidate {
    /// The generated ids with the stop token that ended them, when one did.
    pub fn tokens_with_stop(&self) -> Vec<i64> {
        let mut out = self.tokens.clone();
        out.extend(self.stop);
        out
    }
}

/// How a decode searches: the width of a beam search, or the temperature
/// and trajectories of sampling, and how the finished candidates are ranked.
///
/// The value object a model's decode config embeds; it builds the
/// [`TokenSearch`] ([`init_search`](Self::init_search)) and the ranker
/// ([`init_ranker`](Self::init_ranker)).
#[derive(Config, Debug, PartialEq)]
pub struct SearchConfig {
    /// Beams per group; one is greedy.
    #[config(default = "1")]
    pub beam_size: usize,

    /// Finished candidates to collect before a beam search stops, as a
    /// multiple of the beam size; `None` is one.
    #[config(default = "None")]
    pub patience: Option<f64>,

    /// The exponent of the ranker's length penalty; `None` normalizes by
    /// length.
    #[config(default = "None")]
    pub length_penalty: Option<f64>,

    /// Sampling temperature; zero is the argmax. Above zero the search is
    /// greedy sampling, `best_of` trajectories per group, whatever
    /// `beam_size` says.
    #[config(default = "0.0")]
    pub temperature: f64,

    /// Independent sample trajectories per group when `temperature` is
    /// above zero; `None` is one.
    #[config(default = "None")]
    pub best_of: Option<usize>,
}

impl SearchConfig {
    /// The search this config asks for: sampling above temperature zero,
    /// else a beam search when `beam_size` is more than one, else the
    /// argmax.
    ///
    /// # Arguments
    /// * `stops` - the tokens that end a sequence.
    /// * `filler` - what a greedy search feeds a finished row; must be a valid
    ///   id, which a stop token need not be.
    pub fn init_search(
        &self,
        stops: &[i64],
        filler: i64,
    ) -> Box<dyn TokenSearch> {
        let stops = stops.iter().copied();
        if self.temperature > 0.0 {
            Box::new(
                GreedySearch::new(stops, filler)
                    .with_temperature(self.temperature)
                    .with_group(self.best_of.unwrap_or(1)),
            )
        } else if self.beam_size == 1 {
            Box::new(GreedySearch::new(stops, filler))
        } else {
            Box::new(BeamSearch::new(self.beam_size, stops, self.patience))
        }
    }

    /// The ranker this config asks for.
    pub fn init_ranker(&self) -> MaximumLikelihoodRanker {
        MaximumLikelihoodRanker {
            length_penalty: self.length_penalty,
        }
    }

    /// This search at `temperature`: above zero the beam is off and
    /// `best_of` trajectories are sampled, as a temperature ladder's
    /// sampling rungs run; at zero the search is as configured, one
    /// trajectory.
    pub fn at_temperature(
        &self,
        temperature: f64,
        best_of: Option<usize>,
    ) -> Self {
        let mut config = self.clone().with_temperature(temperature);
        if temperature > 0.0 {
            config.beam_size = 1;
            config.patience = None;
            config.best_of = best_of;
        } else {
            config.best_of = None;
        }
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_search_and_rungs() {
        let base = SearchConfig::new()
            .with_beam_size(5)
            .with_patience(Some(2.0))
            .with_length_penalty(Some(0.6));
        assert_eq!(base.init_search(&[9], 0).group_size(), 5);
        assert_eq!(base.init_ranker().length_penalty, Some(0.6));

        let sampled = base.at_temperature(0.4, Some(3));
        assert_eq!(sampled.beam_size, 1);
        assert_eq!(sampled.patience, None);
        assert_eq!(sampled.best_of, Some(3));
        assert_eq!(sampled.init_search(&[9], 0).group_size(), 3);

        let searched = base.at_temperature(0.0, Some(3));
        assert_eq!(searched.beam_size, 5);
        assert_eq!(searched.best_of, None);
        assert_eq!(SearchConfig::new().init_search(&[9], 0).group_size(), 1);
    }
}
