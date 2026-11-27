//! Logit filters: rewriting the logits a search chooses from.
use alloc::sync::Arc;
use core::fmt::Debug;
use std::sync::Mutex;

use burn::{
    Tensor,
    prelude::{
        Bool,
        TensorData,
    },
    tensor::Device,
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
    let [_, vocab] = logits.dims();
    let mask = suppress_mask(ids, vocab, &logits.device());
    fill_masked(logits, mask)
}

/// `[vocab]`: true at each of `ids` within the vocabulary.
fn suppress_mask(
    ids: &[i64],
    vocab: usize,
    device: &Device,
) -> Tensor<1, Bool> {
    let mut mask = vec![false; vocab];
    for &id in ids {
        if let Ok(i) = usize::try_from(id)
            && i < vocab
        {
            mask[i] = true;
        }
    }
    Tensor::from_data(TensorData::new(mask, [vocab]), device)
}

/// `logits` with every row's `mask`ed ids at `-inf`.
fn fill_masked(
    logits: Tensor<2>,
    mask: Tensor<1, Bool>,
) -> Tensor<2> {
    let [rows, vocab] = logits.dims();
    logits.mask_fill(
        mask.unsqueeze::<2>().expand([rows, vocab]),
        f32::NEG_INFINITY,
    )
}

/// A mask, with the vocabulary width and the device it was built for.
type CachedMask = (usize, Device, Tensor<1, Bool>);

/// Never picks these ids.
///
/// Its mask is built once per vocabulary width and device, on the first
/// step, and reused: a decode does not rebuild and upload it every step.
/// Clones share it.
#[derive(Debug, Clone)]
pub struct SuppressTokens {
    ids: Vec<i64>,
    mask: Arc<Mutex<Option<CachedMask>>>,
}

impl SuppressTokens {
    /// Suppresses `ids`, in any order, duplicates allowed.
    pub fn new(ids: impl IntoIterator<Item = i64>) -> Self {
        let mut ids: Vec<i64> = ids.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        Self {
            ids,
            mask: Arc::new(Mutex::new(None)),
        }
    }

    /// The mask for a vocabulary `vocab` wide on `device`: the one built
    /// before, or built now and kept.
    fn mask(
        &self,
        vocab: usize,
        device: &Device,
    ) -> Tensor<1, Bool> {
        let mut cached = self.mask.lock().expect("the suppress mask's lock");
        if let Some((width, at, mask)) = cached.as_ref()
            && *width == vocab
            && at == device
        {
            return mask.clone();
        }
        let mask = suppress_mask(&self.ids, vocab, device);
        *cached = Some((vocab, device.clone(), mask.clone()));
        mask
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
        let [_, vocab] = logits.dims();
        let mask = self.mask(vocab, &logits.device());
        fill_masked(logits, mask)
    }
}

/// Penalizes the ids a row has recently generated: CTRL's repetition penalty
/// (Keskar et al. 2019), a positive logit divided by `penalty` and a
/// negative one multiplied by it, so a penalty above one makes a repeat less
/// likely whatever its logit's sign.
///
/// It counts presence, not repeats: an id generated once is penalized as
/// much as one generated ten times. Only generated tokens count, the row's
/// tokens from `prompt_len` on (`transformers`'
/// `RepetitionPenaltyLogitsProcessor` counts the prompt too), at most the
/// last [`window`](Self::with_window) of them; the
/// [`exempt`](Self::with_exempt) ids are never penalized. A row without a
/// sequence is left as it is.
#[derive(Debug, Clone, PartialEq)]
pub struct RepetitionPenalty {
    penalty: f64,
    window: Option<usize>,
    exempt: Vec<i64>,
}

impl RepetitionPenalty {
    /// Every generated token counts, and no id is exempt.
    ///
    /// # Panics
    /// If `penalty` is not positive.
    pub fn new(penalty: f64) -> Self {
        assert!(penalty > 0.0, "a repetition penalty is positive");
        Self {
            penalty,
            window: None,
            exempt: Vec::new(),
        }
    }

    /// Counts only the last `window` generated tokens.
    pub fn with_window(
        mut self,
        window: usize,
    ) -> Self {
        self.window = Some(window);
        self
    }

    /// Never penalizes `ids`, in any order, duplicates allowed.
    pub fn with_exempt(
        mut self,
        ids: impl IntoIterator<Item = i64>,
    ) -> Self {
        self.exempt = ids.into_iter().collect();
        self.exempt.sort_unstable();
        self.exempt.dedup();
        self
    }

    /// The penalty.
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// The generated tokens counted, the last ones; `None`, all of them.
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    /// The ids never penalized, sorted.
    pub fn exempt(&self) -> &[i64] {
        &self.exempt
    }
}

impl LogitFilter for RepetitionPenalty {
    fn apply(
        &self,
        logits: Tensor<2>,
        tokens: &[Vec<i64>],
        prompt_len: usize,
    ) -> Tensor<2> {
        let [rows, vocab] = logits.dims();
        let mut mask = vec![false; rows * vocab];
        let mut any = false;
        for (row, sequence) in tokens.iter().enumerate().take(rows) {
            let generated = sequence.get(prompt_len..).unwrap_or_default();
            let start = self
                .window
                .map_or(0, |window| generated.len().saturating_sub(window));
            for &id in &generated[start..] {
                if let Ok(i) = usize::try_from(id)
                    && i < vocab
                    && self.exempt.binary_search(&id).is_err()
                {
                    mask[row * vocab + i] = true;
                    any = true;
                }
            }
        }
        if !any {
            return logits;
        }
        let mask: Tensor<2, Bool> =
            Tensor::from_data(TensorData::new(mask, [rows, vocab]), &logits.device());
        // l / p where l > 0, l * p elsewhere (-inf stays -inf).
        let penalized = logits.clone().clamp_min(0.0) / self.penalty
            + logits.clone().clamp_max(0.0) * self.penalty;
        logits.mask_where(mask, penalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::{
        cpu_device,
        tensor_to_vec_as,
    };

    /// `filter` over `rows`, read back.
    fn filtered(
        filter: &dyn LogitFilter,
        rows: &[[f32; 4]],
        tokens: &[Vec<i64>],
        prompt_len: usize,
    ) -> Vec<Vec<f32>> {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let logits = Tensor::<2>::from_data(TensorData::new(flat, [rows.len(), 4]), &cpu_device());
        let out: Vec<f32> = tensor_to_vec_as::<f32, _>(&filter.apply(logits, tokens, prompt_len));
        out.chunks(4).map(<[f32]>::to_vec).collect()
    }

    #[test]
    fn test_repetition_penalty() {
        let rows = [[2.0, -2.0, 1.0, f32::NEG_INFINITY], [2.0, -2.0, 1.0, 0.5]];
        let penalty = RepetitionPenalty::new(2.0);

        // Presence, not repeats: id 0 twice is penalized once; the sign
        // decides divide or multiply; -inf stays -inf.
        assert_eq!(
            filtered(&penalty, &rows, &[vec![0, 0, 1, 3], vec![2]], 0),
            [
                vec![1.0, -4.0, 1.0, f32::NEG_INFINITY],
                vec![2.0, -2.0, 0.5, 0.5],
            ]
        );

        // The prompt never counts.
        assert_eq!(
            filtered(&penalty, &rows, &[vec![0, 1, 2], vec![0, 1, 2]], 2)[0],
            [2.0, -2.0, 0.5, f32::NEG_INFINITY]
        );

        // Only the window's last tokens; an exempt id never.
        let windowed = RepetitionPenalty::new(2.0)
            .with_window(2)
            .with_exempt([2, 2]);
        assert_eq!(windowed.exempt(), [2]);
        assert_eq!(
            filtered(&windowed, &rows, &[vec![0, 1, 2], vec![]], 0),
            [
                vec![2.0, -4.0, 1.0, f32::NEG_INFINITY],
                vec![2.0, -2.0, 1.0, 0.5],
            ]
        );

        // No sequences, or nothing generated: the logits as they are.
        assert_eq!(filtered(&penalty, &rows, &[], 0)[1], rows[1]);
    }

    /// The mask is built once and reused, and built again for another
    /// vocabulary width.
    #[test]
    fn test_suppress_tokens_reuses_its_mask() {
        let filter = SuppressTokens::new([1, 4]);
        let device = cpu_device();
        let row =
            |n: usize| Tensor::<2>::from_data(TensorData::new(vec![1.0f32; n], [1, n]), &device);
        let kept = |out: Tensor<2>| -> Vec<bool> {
            tensor_to_vec_as::<f32, _>(&out)
                .iter()
                .map(|x| x.is_finite())
                .collect()
        };
        for _ in 0..2 {
            assert_eq!(
                kept(filter.apply(row(4), &[], 0)),
                [true, false, true, true]
            );
        }
        assert_eq!(
            kept(filter.clone().apply(row(6), &[], 0)),
            [true, false, true, true, false, true],
            "a wider vocabulary, its own mask"
        );
    }
}
