//! # Greedy search, expressed through the search seam.
//!
//! One row per audio, the argmax every step. Rows finish independently: a
//! row that emits `<|endoftext|>` stops contributing, but the batch keeps
//! stepping until every row has finished or the cap is reached, so a
//! finished row is fed a filler token whose output is discarded. The filler
//! is the first prompt token rather than the stop token, because the stop
//! token need not be a valid embedding index.

use burn::{
    Tensor,
    prelude::{
        Backend,
        Int,
    },
    tensor::activation::log_softmax,
};

use crate::kits::speech::whisper::decode::TokenDecoder;

/// The argmax, one row per audio.
#[derive(Debug, Clone)]
pub struct GreedyDecoder {
    eot: i64,
    filler: i64,
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
            finished: Vec::new(),
        }
    }
}

impl<B: Backend> TokenDecoder<B> for GreedyDecoder {
    fn group_size(&self) -> usize {
        1
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

        let picked: Tensor<B, 2, Int> = logits.clone().argmax(1);
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
        tokens
            .into_iter()
            .zip(sum_logprobs)
            .map(|(sequence, logprob)| vec![(sequence[prompt_len..].to_vec(), logprob)])
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
}
