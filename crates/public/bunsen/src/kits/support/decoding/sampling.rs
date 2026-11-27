//! Sampling: the truncations top-k, top-p and min-p, and the draw.
use alloc::vec::Vec;

use burn::{
    Tensor,
    prelude::{
        Int,
        TensorData,
    },
    tensor::{
        DType,
        Distribution,
        activation::softmax,
    },
};

use super::LogitFilter;
use crate::ops::reduce::{
    argmax_rows,
    max_rows,
    softmax_normalizer_rows,
    topk_rows,
    topk_rows_host,
};

/// The candidates [`TopP`] finds a nucleus among before it sorts the
/// whole row: the top 64.
pub const NUCLEUS_CANDIDATES: usize = 64;

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
        let top = topk_rows_host(logits.clone(), k);
        let kth: Vec<f32> = top.values.chunks(k).map(|row| row[k - 1]).collect();
        // The logits' own values, widened: narrowing them back is exact.
        let kth = Tensor::<2>::from_data(
            TensorData::new(kth, [rows, 1]),
            (&logits.device(), DType::F32),
        )
        .cast(logits.dtype());
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
        if let Some(threshold) = self.candidate_threshold(logits.clone()) {
            let below = logits.clone().lower(threshold.expand([rows, vocab]));
            return logits.mask_fill(below, f32::NEG_INFINITY);
        }
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

impl TopP {
    /// Each row's threshold, `[rows, 1]` in the logits' dtype, from its top
    /// [`NUCLEUS_CANDIDATES`] candidates alone (by chunked reductions, without
    /// the full sort): their probabilities at the temperature over the
    /// whole row's normalizer, the nucleus counted as the sort counts it.
    ///
    /// `None` when a row's nucleus may run past the candidates (it keeps
    /// all of them), for the full sort to decide. After a [`TopK`] of at
    /// most [`NUCLEUS_CANDIDATES`], a nucleus never does.
    fn candidate_threshold(
        &self,
        logits: Tensor<2>,
    ) -> Option<Tensor<2>> {
        let [_, vocab] = logits.dims();
        let m = vocab.min(NUCLEUS_CANDIDATES);
        let temperature = self.temperature.max(f64::MIN_POSITIVE);
        let (max, sum) = softmax_normalizer_rows(logits.clone().cast(DType::F32) / temperature);
        let (top, _) = topk_rows(logits.clone(), m);
        let probs = ((top.clone() / temperature) - max).exp() / sum;
        // A token is kept while the mass before it is under p.
        let before = probs.clone().cumsum(1) - probs;
        let kept: Tensor<2, Int> = before.lower_elem(self.p).int().sum_dim(1).clamp_min(1);
        let counts: Vec<i64> = kept
            .clone()
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .expect("the nucleus counts read back");
        if m < vocab && counts.iter().any(|&c| c as usize >= m) {
            return None;
        }
        Some(top.gather(1, kept - 1).cast(logits.dtype()))
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
        let floor = max_rows(logits.clone()).cast(logits.dtype()) + self.temperature * self.p.ln();
        let below = logits.clone().lower(floor.expand([rows, vocab]));
        logits.mask_fill(below, f32::NEG_INFINITY)
    }
}

/// One id per row, `[rows, 1]`, drawn from `softmax(logits / temperature)`;
/// at temperature zero, the argmax.
///
/// The draw is the Gumbel-max trick, `argmax(logits / t + g)` with `g =
/// -log(-log u)`, on the backend's own random numbers, so `device.seed(..)`
/// makes it repeatable. A suppressed (`-inf`) id is never drawn.
///
/// The noise is drawn and added in f32, whatever the logits' dtype, and `u`
/// stays below 1: a `u` of 1 is noise of +inf, which outranks every logit.
/// In bf16, a draw within 2^-9 of 1 rounds to 1, so one id in 512 would be
/// picked whatever its logit.
///
/// # Panics
/// If `temperature` is negative.
pub fn sample_softmax(
    logits: Tensor<2>,
    temperature: f64,
) -> Tensor<2, Int> {
    assert!(temperature >= 0.0, "a temperature is not negative");
    if temperature == 0.0 {
        return argmax_rows(logits);
    }
    let logits = logits.cast(DType::F32);
    let uniform: Tensor<2> = Tensor::random(
        logits.dims(),
        Distribution::Uniform(0.0, 1.0),
        (&logits.device(), DType::F32),
    )
    .clamp(1e-20, 1.0 - f64::from(f32::EPSILON) / 2.0);
    let gumbel = uniform.log().neg().log().neg();
    argmax_rows(logits / temperature + gumbel)
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::support::testing::{
        cpu_device,
        int_tensor_to_vec,
        tensor_to_vec_as,
    };

    /// Which ids of a row survive a filter.
    fn kept(
        filter: &dyn LogitFilter,
        row: &[f32],
    ) -> Vec<usize> {
        let logits =
            Tensor::<2>::from_data(TensorData::new(row.to_vec(), [1, row.len()]), &cpu_device());
        let out: Vec<f32> = tensor_to_vec_as::<f32, _>(&filter.apply(logits, &[], 0));
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

    #[test]
    fn test_sample_softmax() {
        let device = cpu_device();
        let logits = Tensor::<2>::from_data(
            TensorData::new(
                vec![0.0, 3.0, 1.0, f32::NEG_INFINITY, 2.0, 0.0, 1.0, 0.5],
                [2, 4],
            ),
            &device,
        );
        let ids = |t: Tensor<2, Int>| int_tensor_to_vec(&t);
        assert_eq!(
            ids(sample_softmax(logits.clone(), 0.0)),
            [1, 0],
            "the argmax"
        );

        device.seed(5);
        let first = ids(sample_softmax(logits.clone(), 1.0));
        device.seed(5);
        assert_eq!(ids(sample_softmax(logits.clone(), 1.0)), first, "seeded");

        for _ in 0..64 {
            assert_ne!(
                ids(sample_softmax(logits.clone(), 4.0))[0],
                3,
                "never a -inf id"
            );
        }
    }
}
