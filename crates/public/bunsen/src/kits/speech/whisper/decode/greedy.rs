//! # Greedy search, expressed through the search seam.
//!
//! One row per audio, the argmax every step; or, above temperature zero,
//! a sample from the softmax of the logits over the temperature, drawn by
//! the Gumbel-max trick on the backend's own random numbers (so `B::seed`
//! makes it repeatable), with `group` independent trajectories per audio
//! for the ranker to choose among &mdash; upstream's `best_of`. Rows finish
//! independently: a row that emits `<|endoftext|>` stops contributing, but the
//! batch keeps stepping until every row has finished or the cap is reached, so
//! a finished row is fed a filler token whose output is discarded. The filler
//! is the first prompt token rather than the stop token, because the stop
//! token need not be a valid embedding index.

use burn::{
    Tensor,
    prelude::{
        Backend,
        Int,
    },
    tensor::{
        Distribution,
        activation::log_softmax,
    },
};

use crate::kits::speech::whisper::decode::TokenDecoder;

/// The argmax, one row per audio; or sampling, `group` rows per audio.
#[derive(Debug, Clone)]
pub struct GreedyDecoder {
    eot: i64,
    filler: i64,
    temperature: f64,
    group: usize,
    finished: Vec<bool>,
}

impl GreedyDecoder {
    /// # Arguments
    /// * `eot` - the stop token.
    /// * `filler` - what a finished row is fed; must be a valid id.
    pub fn new(
        eot: i64,
        filler: i64,
    ) -> Self {
        Self {
            eot,
            filler,
            temperature: 0.0,
            group: 1,
            finished: Vec::new(),
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

    /// Trajectories per audio; only meaningful when sampling.
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

impl<B: Backend> TokenDecoder<B> for GreedyDecoder {
    fn group_size(&self) -> usize {
        self.group
    }

    fn reset(&mut self) {
        self.finished.clear();
    }

    fn update(
        &mut self,
        tokens: &mut Vec<Vec<i64>>,
        logits: Tensor<B, 2>,
        sum_logprobs: &mut [f32],
        _reorder: &mut dyn FnMut(&[usize]),
    ) -> (Vec<i64>, bool) {
        let [rows, _] = logits.dims();
        if self.finished.len() != rows {
            self.finished = vec![false; rows];
        }

        let picked: Tensor<B, 2, Int> = if self.temperature > 0.0 {
            // Gumbel-max: argmax(logits / t + g), g = -log(-log u), is a
            // sample from softmax(logits / t). Suppressed ids stay -inf.
            let uniform: Tensor<B, 2> = Tensor::random(
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
        let chosen: Vec<i64> = picked
            .clone()
            .into_data()
            .convert::<i64>()
            .to_vec()
            .unwrap();
        let logprobs: Vec<f32> = log_softmax(logits, 1)
            .gather(1, picked)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();

        let mut feed = Vec::with_capacity(rows);
        for row in 0..rows {
            let token = chosen[row];
            if self.finished[row] {
                feed.push(self.filler);
            } else if token == self.eot {
                self.finished[row] = true;
                feed.push(self.filler);
            } else {
                tokens[row].push(token);
                sum_logprobs[row] += logprobs[row];
                feed.push(token);
            }
        }

        let completed = self.finished.iter().all(|&done| done);
        (feed, completed)
    }

    fn finalize(
        &mut self,
        tokens: Vec<Vec<i64>>,
        sum_logprobs: Vec<f32>,
        prompt_len: usize,
    ) -> Vec<Vec<(Vec<i64>, f32)>> {
        let group = self.group.max(1);
        tokens
            .into_iter()
            .zip(sum_logprobs)
            .map(|(sequence, logprob)| (sequence[prompt_len..].to_vec(), logprob))
            .collect::<Vec<_>>()
            .chunks(group)
            .map(|audio| audio.to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

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

    /// Two rows: one keeps going, one stops on the stop token and is fed
    /// the filler from then on; the batch completes when both have stopped.
    #[test]
    fn test_rows_finish_independently() {
        let eot = 3;
        let mut decoder = GreedyDecoder::new(eot, 9);
        let mut tokens = vec![vec![9, 9], vec![9, 9]];
        let mut sums = vec![0.0, 0.0];
        let mut reorders = 0;
        let mut reorder = |_: &[usize]| reorders += 1;

        let (feed, done) = TokenDecoder::<B>::update(
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

        let (feed, done) = TokenDecoder::<B>::update(
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

        let out = TokenDecoder::<B>::finalize(&mut decoder, tokens, sums, 2);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0].0, vec![1]);
        assert_eq!(out[1][0].0, Vec::<i64>::new());

        TokenDecoder::<B>::reset(&mut decoder);
        assert!(decoder.finished.is_empty());
    }

    /// Sampling: a decisive distribution is picked every time, with the
    /// unscaled log probability recorded; a two-way tie is broken by the
    /// backend's random numbers, both ways over many draws.
    #[test]
    fn test_sampling() {
        let mut decoder = GreedyDecoder::new(3, 9).with_temperature(0.5);
        assert_eq!(decoder.temperature(), 0.5);
        let mut reorder = |_: &[usize]| {};

        for _ in 0..8 {
            let mut tokens = vec![vec![9]];
            let mut sums = vec![0.0];
            let (feed, _) = TokenDecoder::<B>::update(
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
            TokenDecoder::<B>::reset(&mut decoder);
        }

        let mut seen = [0usize; 4];
        for _ in 0..64 {
            let mut tokens = vec![vec![9]];
            let mut sums = vec![0.0];
            let (feed, _) = TokenDecoder::<B>::update(
                &mut decoder,
                &mut tokens,
                logits(&[&[5.0, -30.0, 5.0, -30.0]]),
                &mut sums,
                &mut reorder,
            );
            seen[feed[0] as usize] += 1;
            TokenDecoder::<B>::reset(&mut decoder);
        }
        assert!(seen[0] > 0 && seen[2] > 0, "both sides of a tie: {seen:?}");
        assert!(
            seen[1] == 0 && seen[3] == 0,
            "never the improbable: {seen:?}"
        );
    }

    /// Trajectories per audio come back grouped, in row order.
    #[test]
    fn test_group_finalize() {
        let mut decoder = GreedyDecoder::new(3, 9).with_temperature(0.5).with_group(2);
        assert_eq!(TokenDecoder::<B>::group_size(&decoder), 2);
        let out = TokenDecoder::<B>::finalize(
            &mut decoder,
            vec![vec![9, 1], vec![9, 2], vec![9, 1, 1], vec![9]],
            vec![-1.0, -2.0, -3.0, -4.0],
            1,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], vec![(vec![1], -1.0), (vec![2], -2.0)]);
        assert_eq!(out[1], vec![(vec![1, 1], -3.0), (vec![], -4.0)]);
    }
}
