//! Causal mask utilities.

use burn::{
    Tensor,
    prelude::Bool,
    tensor::Device,
};

/// Generates a Bool causal mask `[1, seq_len, n_past + seq_len]`.
/// `true` = masked (future positions blocked), `false` = attend.
pub fn causal_mask(
    seq_len: usize,
    n_past: usize,
    device: &Device,
) -> Tensor<3, Bool> {
    let total = n_past + seq_len;
    Tensor::<2, Bool>::tril_mask([seq_len, total], n_past as i64, device).unsqueeze::<3>()
}
