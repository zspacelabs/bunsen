//! The cached decode loop.
use std::sync::Arc;

use burn::{
    Tensor,
    prelude::{
        Int,
        TensorData,
    },
};

use super::{
    Candidate,
    LogitFilter,
    TokenSearch,
};

/// A model a cached decode steps: one token per row through its cache, and
/// the cache's rows permuted when beams branch.
///
/// The model is immutable; what a decode carries between steps (the KV cache,
/// a rotary table sized for the decode) is its [`State`](Self::State),
/// which the caller makes, prefills, and passes in.
pub trait CachedStep {
    /// What a decode carries between steps.
    type State;

    /// Runs one token per row through the model at the state's position,
    /// advancing it.
    ///
    /// # Arguments
    /// * `ids` - `[rows, 1]`.
    ///
    /// # Returns
    /// `[rows, vocab]`, the logits of the next token.
    fn step(
        &self,
        ids: Tensor<2, Int>,
        state: &mut Self::State,
    ) -> Tensor<2>;

    /// Permutes the state's rows: row `r` becomes row `sources[r]`.
    fn reorder(
        &self,
        state: &mut Self::State,
        sources: &[usize],
    );
}

/// Decodes from a prefilled state: filters the logits, lets the search
/// choose, steps the model on its choice, until the search completes or
/// `max_new_tokens` have been chosen.
///
/// The caller prefills: it runs each row's prompt through the model into
/// `state` (as embeddings, for a model whose prompt is not all tokens), and
/// hands over the logits that followed. Rows are laid out as the search
/// groups them (`group * group_size + member`), so a beam search of `k`
/// over one prompt starts from `k` identical rows.
///
/// # Arguments
/// * `prompts` - one per row, every one the same length: what a filter sees as
///   the sequence so far. Empty prompts are allowed.
/// * `first_logits` - `[rows, vocab]`, the logits after each row's prompt.
/// * `max_new_tokens` - the budget: the most tokens chosen per row, stop tokens
///   included.
///
/// # Returns
/// The search's finished candidates, per group.
///
/// # Panics
/// If the prompts differ in length, or their count is not `first_logits`'
/// rows.
pub fn decode_cached<M: CachedStep>(
    model: &M,
    state: &mut M::State,
    prompts: Vec<Vec<i64>>,
    first_logits: Tensor<2>,
    max_new_tokens: usize,
    search: &mut dyn TokenSearch,
    filters: &[Arc<dyn LogitFilter>],
) -> Vec<Vec<Candidate>> {
    let [rows, _] = first_logits.dims();
    assert_eq!(prompts.len(), rows, "one prompt per row of logits");
    let prompt_len = prompts.first().map_or(0, Vec::len);
    assert!(
        prompts.iter().all(|p| p.len() == prompt_len),
        "every prompt the same length"
    );
    let device = first_logits.device();

    search.reset();
    let mut tokens = prompts;
    let mut sum_logprobs = vec![0f32; rows];
    let mut logits = first_logits;
    for step in 0..max_new_tokens {
        for filter in filters {
            logits = filter.apply(logits, &tokens, prompt_len);
        }
        let (feed, completed) =
            search.update(&mut tokens, logits, &mut sum_logprobs, &mut |sources| {
                model.reorder(state, sources)
            });
        if completed || step + 1 == max_new_tokens {
            break;
        }
        let ids = Tensor::<2, Int>::from_data(TensorData::new(feed, [rows, 1]), &device);
        logits = model.step(ids, state);
    }
    search.finalize(tokens, sum_logprobs, prompt_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kits::decoding::{
            GreedySearch,
            SuppressTokens,
        },
        support::testing::cpu_device,
    };

    /// A model whose next token is always the last fed plus one, its state
    /// every id it was fed, by row.
    struct Counter {
        vocab: usize,
    }

    impl CachedStep for Counter {
        type State = Vec<Vec<i64>>;

        fn step(
            &self,
            ids: Tensor<2, Int>,
            state: &mut Self::State,
        ) -> Tensor<2> {
            let fed: Vec<i64> = ids.into_data().try_into_vec_as::<i64>().unwrap();
            let mut data = vec![0f32; fed.len() * self.vocab];
            for (row, &id) in fed.iter().enumerate() {
                state[row].push(id);
                data[row * self.vocab + ((id + 1) as usize % self.vocab)] = 10.0;
            }
            Tensor::from_data(
                TensorData::new(data, [fed.len(), self.vocab]),
                &cpu_device(),
            )
        }

        fn reorder(
            &self,
            state: &mut Self::State,
            sources: &[usize],
        ) {
            *state = sources.iter().map(|&s| state[s].clone()).collect();
        }
    }

    fn first(
        rows: usize,
        next: usize,
        vocab: usize,
    ) -> Tensor<2> {
        let mut data = vec![0f32; rows * vocab];
        for row in 0..rows {
            data[row * vocab + next] = 10.0;
        }
        Tensor::from_data(TensorData::new(data, [rows, vocab]), &cpu_device())
    }

    /// The loop steps the model on each choice until a stop token, which
    /// the candidate records; the budget cuts a run that never stops.
    #[test]
    fn test_decode_until_a_stop_or_the_budget() {
        let model = Counter { vocab: 8 };
        let mut state = vec![Vec::new()];
        let mut search = GreedySearch::new([5], 0);
        let out = decode_cached(
            &model,
            &mut state,
            vec![vec![9, 9]],
            first(1, 2, 8),
            16,
            &mut search,
            &[],
        );
        assert_eq!(out[0][0].tokens, vec![2, 3, 4]);
        assert_eq!(out[0][0].stop, Some(5));
        assert_eq!(state, vec![vec![2, 3, 4]], "the stop token is never fed");

        let mut state = vec![Vec::new()];
        let out = decode_cached(
            &model,
            &mut state,
            vec![Vec::new()],
            first(1, 2, 8),
            2,
            &mut search,
            &[],
        );
        assert_eq!(out[0][0].tokens, vec![2, 3]);
        assert_eq!(out[0][0].stop, None);
        assert_eq!(state, vec![vec![2]], "the last choice is not stepped");
    }

    /// Filters see the logits before the search does.
    #[test]
    fn test_filters_reach_the_search() {
        let model = Counter { vocab: 8 };
        let mut state = vec![Vec::new()];
        let mut search = GreedySearch::new([7], 0);
        let filters: Vec<Arc<dyn LogitFilter>> = vec![Arc::new(SuppressTokens::new([2]))];
        let out = decode_cached(
            &model,
            &mut state,
            vec![Vec::new()],
            first(1, 2, 8),
            1,
            &mut search,
            &filters,
        );
        assert_ne!(out[0][0].tokens, vec![2], "2 was suppressed");
    }
}
