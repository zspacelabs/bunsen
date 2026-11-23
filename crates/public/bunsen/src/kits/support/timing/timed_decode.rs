//! [`TimedDecode`] and its synthetic input.

use burn::{
    Tensor,
    prelude::Int,
    tensor::{
        Device,
        TensorData,
    },
};

use crate::ops::transformers::attention::KVCacheStorage;

/// A decode a timing harness drives: its state made for `rows` rows of at
/// most `positions` positions, its input encoded (a model with an encoder),
/// a prompt of `len` positions prefilled, then one step at a time, each on
/// the model's own synthetic input.
///
/// Every call returns what a real decode reads back after it (the last
/// position's logits), so a harness can close a timing window on a read of
/// it, as a decode does when it takes its token. The input is fixed (the same
/// on every run) and nothing stops a decode early, so two runs do the same
/// work.
pub trait TimedDecode {
    /// What the decode carries between calls.
    type State;

    /// What one step makes, for a report's units: `"token"` for a language
    /// model.
    fn step_unit(&self) -> &'static str {
        "token"
    }

    /// A state for `rows` rows of at most `positions` positions each, its KV
    /// cache under `storage`.
    fn timed_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> Self::State;

    /// Encodes every row's synthetic input into `state`, for a model whose
    /// decode attends to an encoder's output (Whisper's audio); `None`, and
    /// nothing done, for a model without one (the default). A harness calls
    /// it on every new state, before the prefill.
    ///
    /// # Returns
    /// `[rows, d]`: a slice of the encoder's output (its first frame), for
    /// the harness's read.
    fn timed_encode(
        &self,
        _state: &mut Self::State,
        _rows: usize,
        _device: &Device,
    ) -> Option<Tensor<2>> {
        None
    }

    /// Prefills `len` positions of every row into `state`, after whatever
    /// it holds.
    ///
    /// # Returns
    /// `[rows, n_logits]`: the logits after the last position.
    fn timed_prefill(
        &self,
        state: &mut Self::State,
        rows: usize,
        len: usize,
        device: &Device,
    ) -> Tensor<2>;

    /// One step of every row, at `state`'s position.
    ///
    /// # Returns
    /// `[rows, n_logits]`: the step's logits.
    fn timed_step(
        &self,
        state: &mut Self::State,
        rows: usize,
        device: &Device,
    ) -> Tensor<2>;
}

/// The ids no token decode is short of: a vocabulary of fewer is the one
/// model that must make its own input.
pub const SYNTHETIC_ID_RANGE: i64 = 1000;

/// `[rows, len]` fixed token ids for positions `from..from + len`: the same
/// on every call, different by row and position, all in
/// `1..SYNTHETIC_ID_RANGE`.
///
/// The values don't matter to a timing (a decode reads its token and goes
/// on); a fixed pattern keeps two runs alike.
pub fn synthetic_ids(
    rows: usize,
    from: usize,
    len: usize,
    device: &Device,
) -> Tensor<2, Int> {
    let ids: Vec<i64> = (0..rows)
        .flat_map(|r| {
            (from..from + len)
                .map(move |t| 1 + (t as i64 * 7919 + r as i64 * 104_729) % (SYNTHETIC_ID_RANGE - 1))
        })
        .collect();
    Tensor::from_data(TensorData::new(ids, [rows, len]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// `[rows, len]` ids, read back as `i64` whatever the device's int dtype.
    fn host(ids: Tensor<2, Int>) -> Vec<i64> {
        ids.into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap()
    }

    /// The ids are fixed, in range, and differ by row and by position.
    #[test]
    fn test_synthetic_ids() {
        let device = cpu_device();
        let a = host(synthetic_ids(2, 3, 4, &device));
        assert_eq!(
            a,
            host(synthetic_ids(2, 3, 4, &device)),
            "the same on every call"
        );
        assert!(a.iter().all(|&id| (1..SYNTHETIC_ID_RANGE).contains(&id)));
        assert_ne!(a[..4], a[4..], "rows differ");
        let later = host(synthetic_ids(1, 4, 1, &device));
        assert_eq!(later[0], a[1], "position 4 of row 0, whatever the window");
    }
}
