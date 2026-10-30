//! Sampling truncations: top-k, top-p and min-p.
use burn::{
    Tensor,
    prelude::Int,
    tensor::activation::softmax,
};

use super::LogitFilter;

/// Keeps each row's `k` most probable tokens, `-inf` for the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopK {
    /// Tokens kept per row; at least one.
    pub k: usize,
}

impl LogitFilter for TopK {
    fn apply(
        &self,
        logits: Tensor<2>,
        _tokens: &[Vec<i64>],
        _prompt_len: usize,
    ) -> Tensor<2> {
        let [rows, vocab] = logits.dims();
        let k = self.k.clamp(1, vocab);
        let kth = logits.clone().topk(k, 1).slice_dim(1, k - 1..k);
        let below = logits.clone().lower(kth.expand([rows, vocab]));
        logits.mask_fill(below, f32::NEG_INFINITY)
    }
}

/// Keeps each row's nucleus: its most probable tokens, at `temperature`,
/// until their probabilities sum to `p`; the most probable token always.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopP {
    /// The probability mass kept, in `(0, 1]`.
    pub p: f64,

    /// The temperature the probabilities are taken at: the sampler's.
    pub temperature: f64,
}

impl LogitFilter for TopP {
    fn apply(
        &self,
        logits: Tensor<2>,
        _tokens: &[Vec<i64>],
        _prompt_len: usize,
    ) -> Tensor<2> {
        let [rows, vocab] = logits.dims();
        let sorted = logits.clone().sort_descending(1);
        let probs = softmax(sorted.clone() / self.temperature.max(f64::MIN_POSITIVE), 1);
        // A token is kept while the mass before it is under p.
        let before = probs.clone().cumsum(1) - probs;
        let kept: Tensor<2, Int> = before.lower_elem(self.p).int().sum_dim(1).clamp_min(1);
        let threshold = sorted.gather(1, kept - 1);
        let below = logits.clone().lower(threshold.expand([rows, vocab]));
        logits.mask_fill(below, f32::NEG_INFINITY)
    }
}

/// Keeps the tokens at least `p` times as probable, at `temperature`, as
/// each row's most probable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinP {
    /// The fraction of the top probability a token must reach, in `(0, 1]`.
    pub p: f64,

    /// The temperature the probabilities are taken at: the sampler's.
    pub temperature: f64,
}

impl LogitFilter for MinP {
    fn apply(
        &self,
        logits: Tensor<2>,
        _tokens: &[Vec<i64>],
        _prompt_len: usize,
    ) -> Tensor<2> {
        let [rows, vocab] = logits.dims();
        // p(x) >= p * p(top) is (l - l_top) / t >= ln p.
        let floor = logits.clone().max_dim(1) + self.temperature * self.p.ln();
        let below = logits.clone().lower(floor.expand([rows, vocab]));
        logits.mask_fill(below, f32::NEG_INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::support::testing::cpu_device;

    /// Which ids of a row survive a filter.
    fn kept(
        filter: &dyn LogitFilter,
        row: &[f32],
    ) -> Vec<usize> {
        let logits =
            Tensor::<2>::from_data(TensorData::new(row.to_vec(), [1, row.len()]), &cpu_device());
        let out: Vec<f32> = filter
            .apply(logits, &[], 0)
            .into_data()
            .try_into_vec_as::<f32>()
            .unwrap();
        (0..row.len()).filter(|&i| out[i].is_finite()).collect()
    }

    #[test]
    fn test_top_k() {
        let row = [1.0, 4.0, 3.0, 2.0, 0.0];
        assert_eq!(kept(&TopK { k: 2 }, &row), [1, 2]);
        assert_eq!(kept(&TopK { k: 9 }, &row), [0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_top_p() {
        // ln of 0.5, 0.3, 0.15, 0.05.
        let row = [0.5f32.ln(), 0.3f32.ln(), 0.15f32.ln(), 0.05f32.ln()];
        let at = |p| TopP {
            p,
            temperature: 1.0,
        };
        assert_eq!(kept(&at(0.5), &row), [0]);
        assert_eq!(kept(&at(0.6), &row), [0, 1]);
        assert_eq!(kept(&at(0.9), &row), [0, 1, 2]);
        assert_eq!(kept(&at(1.0), &row), [0, 1, 2, 3]);
        assert_eq!(kept(&at(0.0001), &row), [0], "the top token always");
        // A colder temperature concentrates the mass.
        assert_eq!(
            kept(
                &TopP {
                    p: 0.6,
                    temperature: 0.25
                },
                &row
            ),
            [0]
        );
    }

    #[test]
    fn test_min_p() {
        let row = [0.5f32.ln(), 0.3f32.ln(), 0.15f32.ln(), 0.05f32.ln()];
        assert_eq!(
            kept(
                &MinP {
                    p: 0.5,
                    temperature: 1.0
                },
                &row
            ),
            [0, 1]
        );
        assert_eq!(
            kept(
                &MinP {
                    p: 0.2,
                    temperature: 1.0
                },
                &row
            ),
            [0, 1, 2]
        );
        assert_eq!(
            kept(
                &MinP {
                    p: 0.5,
                    temperature: 0.5
                },
                &row
            ),
            [0]
        );
    }
}
