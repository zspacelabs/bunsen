//! Greedy search and sampling, through the search seam.

use burn::{
    Tensor,
    prelude::Int,
    tensor::{
        Distribution,
        activation::log_softmax,
    },
};

use super::{
    Candidate,
    TokenSearch,
};

/// The argmax, one row per group; or sampling, `group` rows per group.
///
/// Greedy search through the [`TokenSearch`] seam: one row per group, the
/// argmax every step; or, above temperature zero, a sample from the softmax
/// of the logits over the temperature, drawn by the Gumbel-max trick on the
/// backend's own random numbers (so `device.seed(..)` makes it repeatable),
/// with `group` independent trajectories per group for the ranker to choose
/// among: `openai-whisper`'s `best_of`.
///
/// Rows finish independently: a row that emits one of the stop tokens stops
/// contributing, and its candidate records which one stopped it, but the batch
/// keeps stepping until every row has finished or the cap is reached, so a
/// finished row is fed a filler token whose output is discarded.
/// A caller makes the filler a valid id (Whisper's decode feeds the first
/// prompt token), because the stop token need not be a valid embedding
/// index.
#[derive(Debug, Clone)]
pub struct GreedySearch {
    stops: Vec<i64>,
    filler: i64,
    temperature: f64,
    group: usize,
    /// Per row, once the first update has said how many there are: `None`
    /// while it runs, then the stop token that ended it.
    stopped: Vec<Option<i64>>,
}

impl GreedySearch {
    /// # Arguments
    /// * `stops` - the tokens that end a row.
    /// * `filler` - what a finished row is fed; must be a valid id.
    pub fn new(
        stops: impl IntoIterator<Item = i64>,
        filler: i64,
    ) -> Self {
        Self {
            stops: stops.into_iter().collect(),
            filler,
            temperature: 0.0,
            group: 1,
            stopped: Vec::new(),
        }
    }

    /// Samples at `temperature` instead of taking the argmax; zero is the
    /// argmax.
    pub fn with_temperature(
        mut self,
        temperature: f64,
    ) -> Self {
        assert!(temperature >= 0.0, "a temperature is not negative");
        self.temperature = temperature;
        self
    }

    /// Trajectories per group; only meaningful when sampling.
    pub fn with_group(
        mut self,
        group: usize,
    ) -> Self {
        assert!(group >= 1, "at least one trajectory");
        self.group = group;
        self
    }

    /// The sampling temperature; zero is the argmax.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}

impl TokenSearch for GreedySearch {
    fn group_size(&self) -> usize {
        self.group
    }

    fn reset(&mut self) {
        self.stopped.clear();
    }

    fn update(
        &mut self,
        tokens: &mut Vec<Vec<i64>>,
        logits: Tensor<2>,
        sum_logprobs: &mut [f32],
        _reorder: &mut dyn FnMut(&[usize]),
    ) -> (Vec<i64>, bool) {
        let [rows, _] = logits.dims();
        if self.stopped.len() != rows {
            self.stopped = vec![None; rows];
        }

        let picked: Tensor<2, Int> = if self.temperature > 0.0 {
            // Gumbel-max: argmax(logits / t + g), g = -log(-log u), is a
            // sample from softmax(logits / t). Suppressed ids stay -inf.
            let uniform: Tensor<2> = Tensor::random(
                logits.dims(),
                Distribution::Uniform(0.0, 1.0),
                &logits.device(),
            )
            .clamp(1e-20, 1.0);
            let gumbel = uniform.log().neg().log().neg();
            (logits.clone() / self.temperature + gumbel).argmax(1)
        } else {
            logits.clone().argmax(1)
        };
        let chosen: Vec<i64> = picked.clone().into_data().try_into_vec_as::<i64>().unwrap();
        let logprobs: Vec<f32> = log_softmax(logits, 1)
            .gather(1, picked)
            .into_data()
            .try_into_vec_as::<f32>()
            .unwrap();

        let mut feed = Vec::with_capacity(rows);
        for row in 0..rows {
            let token = chosen[row];
            if self.stopped[row].is_some() {
                feed.push(self.filler);
            } else if self.stops.contains(&token) {
                self.stopped[row] = Some(token);
                feed.push(self.filler);
            } else {
                tokens[row].push(token);
                sum_logprobs[row] += logprobs[row];
                feed.push(token);
            }
        }

        let completed = self.stopped.iter().all(Option::is_some);
        (feed, completed)
    }

    fn finalize(
        &mut self,
        tokens: Vec<Vec<i64>>,
        sum_logprobs: Vec<f32>,
        prompt_len: usize,
    ) -> Vec<Vec<Candidate>> {
        let group = self.group.max(1);
        let stopped = std::mem::take(&mut self.stopped);
        tokens
            .into_iter()
            .zip(sum_logprobs)
            .enumerate()
            .map(|(row, (sequence, sum_logprob))| Candidate {
                tokens: sequence[prompt_len..].to_vec(),
                sum_logprob,
                stop: stopped.get(row).copied().flatten(),
            })
            .collect::<Vec<_>>()
            .chunks(group)
            .map(|group| group.to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::support::testing::cpu_device;

    fn logits(rows: &[&[f32]]) -> Tensor<2> {
        let vocab = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::from_data(TensorData::new(flat, [rows.len(), vocab]), &cpu_device())
    }

    /// Two rows: one keeps going, one stops on the stop token and is fed
    /// the filler from then on; the batch completes when both have stopped.
    #[test]
    fn test_rows_finish_independently() {
        let eot = 3;
        let mut decoder = GreedySearch::new([eot], 9);
        let mut tokens = vec![vec![9, 9], vec![9, 9]];
        let mut sums = vec![0.0, 0.0];
        let mut reorders = 0;
        let mut reorder = |_: &[usize]| reorders += 1;

        let (feed, done) = TokenSearch::update(
            &mut decoder,
            &mut tokens,
            logits(&[&[0.0, 5.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 5.0]]),
            &mut sums,
            &mut reorder,
        );
        assert_eq!(feed, vec![1, 9], "row 1 stopped and is fed the filler");
        assert!(!done);
        assert_eq!(tokens, vec![vec![9, 9, 1], vec![9, 9]]);
        assert!(sums[0] < 0.0 && sums[0] > -0.1, "log p of a confident pick");
        assert_eq!(sums[1], 0.0, "a stop token adds nothing");

        let (feed, done) = TokenSearch::update(
            &mut decoder,
            &mut tokens,
            logits(&[&[0.0, 0.0, 0.0, 5.0], &[5.0, 0.0, 0.0, 0.0]]),
            &mut sums,
            &mut reorder,
        );
        assert_eq!(feed, vec![9, 9]);
        assert!(done);
        assert_eq!(
            tokens,
            vec![vec![9, 9, 1], vec![9, 9]],
            "a finished row ignores its logits"
        );
        assert_eq!(reorders, 0, "greedy never permutes the cache");

        let out = TokenSearch::finalize(&mut decoder, tokens, sums, 2);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0].tokens, vec![1]);
        assert_eq!(out[1][0].tokens, Vec::<i64>::new());
        assert_eq!(out[0][0].stop, Some(eot));
        assert_eq!(out[1][0].tokens_with_stop(), vec![eot]);

        TokenSearch::reset(&mut decoder);
        assert!(decoder.stopped.is_empty());
    }

    /// Any of the stop tokens ends a row, and the candidate records which;
    /// a row the budget cut records none.
    #[test]
    fn test_stop_sets() {
        let mut decoder = GreedySearch::new([3, 2], 9);
        let mut tokens = vec![vec![9], vec![9], vec![9]];
        let mut sums = vec![0.0; 3];
        let (_, done) = TokenSearch::update(
            &mut decoder,
            &mut tokens,
            logits(&[
                &[0.0, 0.0, 5.0, 0.0],
                &[0.0, 0.0, 0.0, 5.0],
                &[0.0, 5.0, 0.0, 0.0],
            ]),
            &mut sums,
            &mut |_| {},
        );
        assert!(!done, "the last row runs on");
        let stops: Vec<Option<i64>> = TokenSearch::finalize(&mut decoder, tokens, sums, 1)
            .into_iter()
            .map(|group| group[0].stop)
            .collect();
        assert_eq!(stops, [Some(2), Some(3), None]);
    }

    /// Sampling: a decisive distribution is picked every time, with the
    /// unscaled log probability recorded; a two-way tie is broken by the
    /// backend's random numbers, both ways over many draws.
    #[test]
    fn test_sampling() {
        let mut decoder = GreedySearch::new([3], 9).with_temperature(0.5);
        assert_eq!(decoder.temperature(), 0.5);
        let mut reorder = |_: &[usize]| {};

        for _ in 0..8 {
            let mut tokens = vec![vec![9]];
            let mut sums = vec![0.0];
            let (feed, _) = TokenSearch::update(
                &mut decoder,
                &mut tokens,
                logits(&[&[0.0, 30.0, 0.0, 0.0]]),
                &mut sums,
                &mut reorder,
            );
            assert_eq!(feed, vec![1]);
            assert!(
                sums[0] > -1e-6 && sums[0] <= 0.0,
                "log p of a certain pick: {}",
                sums[0]
            );
            TokenSearch::reset(&mut decoder);
        }

        let mut seen = [0usize; 4];
        for _ in 0..64 {
            let mut tokens = vec![vec![9]];
            let mut sums = vec![0.0];
            let (feed, _) = TokenSearch::update(
                &mut decoder,
                &mut tokens,
                logits(&[&[5.0, -30.0, 5.0, -30.0]]),
                &mut sums,
                &mut reorder,
            );
            seen[feed[0] as usize] += 1;
            TokenSearch::reset(&mut decoder);
        }
        assert!(seen[0] > 0 && seen[2] > 0, "both sides of a tie: {seen:?}");
        assert!(
            seen[1] == 0 && seen[3] == 0,
            "never the improbable: {seen:?}"
        );
    }

    /// Trajectories per group come back grouped, in row order.
    #[test]
    fn test_group_finalize() {
        let mut decoder = GreedySearch::new([3], 9)
            .with_temperature(0.5)
            .with_group(2);
        assert_eq!(TokenSearch::group_size(&decoder), 2);
        let out = TokenSearch::finalize(
            &mut decoder,
            vec![vec![9, 1], vec![9, 2], vec![9, 1, 1], vec![9]],
            vec![-1.0, -2.0, -3.0, -4.0],
            1,
        );
        assert_eq!(out.len(), 2);
        let summary = |group: &[Candidate]| -> Vec<(Vec<i64>, f32, Option<i64>)> {
            group
                .iter()
                .map(|c| (c.tokens.clone(), c.sum_logprob, c.stop))
                .collect()
        };
        assert_eq!(
            summary(&out[0]),
            vec![(vec![1], -1.0, None), (vec![2], -2.0, None)]
        );
        assert_eq!(
            summary(&out[1]),
            vec![(vec![1, 1], -3.0, None), (vec![], -4.0, None)]
        );
    }
}
