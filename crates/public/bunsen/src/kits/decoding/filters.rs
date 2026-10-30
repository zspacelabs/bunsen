//! Logit filters: rewriting the logits a search chooses from.
use std::fmt::Debug;

use burn::{
    Tensor,
    prelude::{
        Bool,
        TensorData,
    },
};

/// Rewrites the logits the search chooses from.
pub trait LogitFilter: Send + Sync + Debug {
    /// Rewrites `logits`, `[rows, vocab]`, the last position's, given the
    /// sequences so far and how long their prompt is.
    ///
    /// # Arguments
    /// * `logits` - `[rows, vocab]`.
    /// * `tokens` - one sequence per row, prompt included.
    /// * `prompt_len` - the length of the prompt every row began with, so a
    ///   filter can tell the first sampled position.
    fn apply(
        &self,
        logits: Tensor<2>,
        tokens: &[Vec<i64>],
        prompt_len: usize,
    ) -> Tensor<2>;
}

/// Sets the given ids to `-inf` in every row: what [`SuppressTokens`] does,
/// for a filter that suppresses ids only sometimes.
pub fn suppress(
    logits: Tensor<2>,
    ids: &[i64],
) -> Tensor<2> {
    let [rows, vocab] = logits.dims();
    let mut mask = vec![false; vocab];
    for &id in ids {
        if let Ok(i) = usize::try_from(id)
            && i < vocab
        {
            mask[i] = true;
        }
    }
    let mask: Tensor<1, Bool> = Tensor::from_data(TensorData::new(mask, [vocab]), &logits.device());
    logits.mask_fill(
        mask.unsqueeze::<2>().expand([rows, vocab]),
        f32::NEG_INFINITY,
    )
}

/// Never picks these ids.
#[derive(Debug, Clone)]
pub struct SuppressTokens {
    ids: Vec<i64>,
}

impl SuppressTokens {
    /// Suppresses `ids`, in any order, duplicates allowed.
    pub fn new(ids: impl IntoIterator<Item = i64>) -> Self {
        let mut ids: Vec<i64> = ids.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        Self { ids }
    }

    /// The suppressed ids, sorted.
    pub fn ids(&self) -> &[i64] {
        &self.ids
    }
}

impl LogitFilter for SuppressTokens {
    fn apply(
        &self,
        logits: Tensor<2>,
        _tokens: &[Vec<i64>],
        _prompt_len: usize,
    ) -> Tensor<2> {
        suppress(logits, &self.ids)
    }
}
