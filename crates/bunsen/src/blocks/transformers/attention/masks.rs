use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
    },
};

/// Generate a Bool causal mask [1, `seq_len`, `n_past` + `seq_len`].
/// `true` = masked (future positions blocked), `false` = attend.
pub fn causal_mask<B: Backend>(
    seq_len: usize,
    n_past: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let total = n_past + seq_len;
    Tensor::<B, 2, Bool>::tril_mask([seq_len, total], n_past as i64, device).unsqueeze::<3>()
}
