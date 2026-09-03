//! Causal mask utilities.

use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
    },
};

/// Generates a Bool causal mask `[1, seq_len, n_past + seq_len]`.
/// `true` = masked (future positions blocked), `false` = attend.
pub fn causal_mask<B: Backend>(
    seq_len: usize,
    n_past: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    Tensor::<B, 3, Bool>::tril_mask([seq_len, n_past + seq_len], n_past as i64, device)
        .unsqueeze::<3>()
}
