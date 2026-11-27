//! # Choosing codes: one filter chain, then a draw

use alloc::vec::Vec;

use burn::{
    Tensor,
    prelude::Int,
    tensor::DType,
};

use super::{
    LogitFilter,
    MinP,
    RepetitionPenalty,
    SuppressTokens,
    TopK,
    TopP,
    sample_softmax,
};

/// A code per row from rows of logits, through one fixed chain of filters,
/// each stage optional, in this order:
///
/// 1. [`suppress`](Self::suppress): ids never chosen, `-inf`;
/// 2. [`penalty`](Self::penalty): the repetition penalty over each row's
///    history, on every row or the first
///    [`penalized_rows`](Self::penalized_rows);
/// 3. the call's blocked ids, `-inf`, for that choice alone;
/// 4. the truncations: [`top_k`](Self::top_k), [`top_p`](Self::top_p),
///    [`min_p`](Self::min_p);
/// 5. a draw at [`temperature`](Self::temperature) ([`sample_softmax`], on the
///    backend's random numbers), or the argmax at zero.
///
/// The logits are cast to f32 first. A TTS model's sampler builds one from
/// its own config: which stages it has, and their settings.
#[derive(Clone, Debug, Default)]
pub struct CodeSampler {
    /// Ids never chosen.
    pub suppress: Option<SuppressTokens>,

    /// The repetition penalty over each row's history.
    pub penalty: Option<RepetitionPenalty>,

    /// The rows the penalty covers, the first ones; `None` covers every row.
    pub penalized_rows: Option<usize>,

    /// Codes kept, the most probable.
    pub top_k: Option<TopK>,

    /// The probability mass kept.
    pub top_p: Option<TopP>,

    /// The least probability, relative to the most probable, kept.
    pub min_p: Option<MinP>,

    /// The temperature it draws at; zero takes the argmax.
    pub temperature: f64,
}

impl CodeSampler {
    /// A code per row.
    ///
    /// # Arguments
    /// - `logits`: `[rows, V]`.
    /// - `history`: per row, the codes chosen before, oldest first.
    /// - `blocked`: ids never chosen this time.
    ///
    /// # Returns
    /// `[rows]`.
    pub fn choose(
        &self,
        logits: Tensor<2>,
        history: &[Vec<i64>],
        blocked: &[i64],
    ) -> Tensor<1, Int> {
        let [rows] =
            crate::contracts::unpack_shape_contract!(["rows", "vocab"], &logits.dims(), &["rows"]);
        let mut logits = logits.cast(DType::F32);
        if let Some(suppress) = &self.suppress {
            logits = suppress.apply(logits, &[], 0);
        }
        if let Some(penalty) = &self.penalty {
            logits = match self.penalized_rows {
                None => penalty.apply(logits, history, 0),
                Some(rows_covered) => {
                    let covered = rows_covered.min(rows);
                    if covered == 0 {
                        logits
                    } else {
                        let penalized = penalty.apply(
                            logits.clone().slice_dim(0, 0..covered),
                            &history[..covered.min(history.len())],
                            0,
                        );
                        if covered < rows {
                            Tensor::cat(vec![penalized, logits.slice_dim(0, covered..rows)], 0)
                        } else {
                            penalized
                        }
                    }
                }
            };
        }
        if !blocked.is_empty() {
            logits = SuppressTokens::new(blocked.iter().copied()).apply(logits, &[], 0);
        }
        if let Some(top_k) = &self.top_k {
            logits = top_k.apply(logits, &[], 0);
        }
        if let Some(top_p) = &self.top_p {
            logits = top_p.apply(logits, &[], 0);
        }
        if let Some(min_p) = &self.min_p {
            logits = min_p.apply(logits, &[], 0);
        }
        sample_softmax(logits, self.temperature).reshape([rows])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::{
        cpu_device,
        int_tensor_to_vec,
    };

    /// At temperature zero, the argmax after each stage: the penalty on the
    /// rows it covers alone, then the blocked ids.
    #[test]
    fn test_choose_in_order() {
        let device = cpu_device();
        let logits = || Tensor::<2>::from_floats([[1.0, 3.0, 2.5], [1.0, 3.0, 2.5]], &device);
        let history = vec![vec![1], vec![1]];
        let penalized = |rows| CodeSampler {
            penalty: Some(RepetitionPenalty::new(4.0)),
            penalized_rows: rows,
            ..CodeSampler::default()
        };
        let chosen = |sampler: &CodeSampler, blocked: &[i64]| {
            int_tensor_to_vec(&sampler.choose(logits(), &history, blocked))
        };
        assert_eq!(chosen(&CodeSampler::default(), &[]), [1, 1]);
        assert_eq!(chosen(&penalized(None), &[]), [2, 2], "every row");
        assert_eq!(chosen(&penalized(Some(1)), &[]), [2, 1], "the first row");
        assert_eq!(chosen(&penalized(Some(0)), &[]), [1, 1], "no row");
        assert_eq!(
            chosen(&penalized(Some(1)), &[2]),
            [0, 1],
            "then the blocked"
        );
    }
}
