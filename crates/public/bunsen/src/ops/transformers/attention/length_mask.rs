//! Validity and window masks.
use alloc::vec::Vec;

use burn::{
    Tensor,
    prelude::{
        Bool,
        Int,
        TensorData,
    },
    tensor::Device,
};

/// `[batch, time]`: true at the positions `j < lengths[b]`, the valid frames
/// of a padded batch.
///
/// The mask every padded batch of different-length items needs, to zero
/// padding, to keep it out of attention, or to normalize over the valid
/// frames only.
pub fn length_mask(
    lengths: &[usize],
    time: usize,
    device: &Device,
) -> Tensor<2, Bool> {
    let lens: Vec<i64> = lengths.iter().map(|&l| l as i64).collect();
    let lens = Tensor::<1, Int>::from_data(TensorData::new(lens, [lengths.len()]), device);
    Tensor::<1, Int>::arange(0..time as i64, device)
        .unsqueeze_dim::<2>(0)
        .lower(lens.unsqueeze_dim::<2>(1))
}

/// `[n, n]`: true where positions `i` and `j` fall in different
/// `window`-wide blocks, `i / window != j / window`.
///
/// The mask of block-diagonal (windowed) self-attention, in the attention
/// convention: true is masked out. Each position attends only within its own
/// window; the last window is short when `window` does not divide `n`.
///
/// # Panics
/// If `window` is 0.
pub fn window_block_mask(
    n: usize,
    window: usize,
    device: &Device,
) -> Tensor<2, Bool> {
    assert!(window > 0, "window_block_mask: window is 0");
    let block = Tensor::<1, Int>::arange(0..n as i64, device).div_scalar(window as i64);
    block
        .clone()
        .unsqueeze_dim::<2>(1)
        .not_equal(block.unsqueeze_dim::<2>(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_length_mask() {
        let mask = length_mask(&[3, 1], 4, &cpu_device());
        let values: Vec<bool> = mask.into_data().iter::<bool>().collect();
        assert_eq!(values, [true, true, true, false, true, false, false, false]);
    }

    #[test]
    fn test_window_block_mask() {
        let mask = window_block_mask(5, 2, &cpu_device());
        let values: Vec<bool> = mask.into_data().iter::<bool>().collect();
        let expected: Vec<bool> = (0..25).map(|k| (k / 5) / 2 != (k % 5) / 2).collect();
        assert_eq!(values, expected);
    }
}
