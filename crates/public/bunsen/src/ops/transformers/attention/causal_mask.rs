//! Causal mask utilities.

use burn::{
    Tensor,
    prelude::{
        Bool,
        Int,
    },
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

/// `[t_q, t_kv]`: true where key `j` is visible to query `i`, for queries
/// that are the last `t_q` of `t_kv` positions. Query `i` sits at position
/// `p = t_kv - t_q + i`, and sees key `j` when `j <= p` and, under a sliding
/// window of `w` positions, `j > p - w`: itself and the `w - 1` positions
/// before it.
///
/// In [`scaled_dot_product_attention`](super::scaled_dot_product_attention)'s
/// mask convention, `true` attends; [`causal_mask`]'s is the reverse.
///
/// # Panics
/// If `t_q > t_kv`, or `window` is `Some(0)`.
pub fn causal_window_visible(
    t_q: usize,
    t_kv: usize,
    window: Option<usize>,
    device: &Device,
) -> Tensor<2, Bool> {
    assert!(t_q <= t_kv, "{t_q} queries over {t_kv} keys");
    assert!(window != Some(0), "a sliding window of 0 positions");
    let rows =
        Tensor::<1, Int>::arange((t_kv - t_q) as i64..t_kv as i64, device).unsqueeze_dim::<2>(1);
    let cols = Tensor::<1, Int>::arange(0..t_kv as i64, device).unsqueeze_dim::<2>(0);
    let causal = cols.clone().lower_equal(rows.clone());
    match window {
        None => causal,
        Some(w) => causal.bool_and(cols.greater(rows.sub_scalar(w as i64))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    fn rows(mask: Tensor<2, Bool>) -> Vec<Vec<bool>> {
        let [_, t_kv] = mask.dims();
        let values: Vec<bool> = mask.into_data().iter::<bool>().collect();
        values.chunks(t_kv).map(<[bool]>::to_vec).collect()
    }

    #[test]
    fn test_causal_window_visible() {
        let device = cpu_device();
        // Two queries at positions 3 and 4 of 5, no window: causal.
        assert_eq!(
            rows(causal_window_visible(2, 5, None, &device)),
            [
                [true, true, true, true, false],
                [true, true, true, true, true],
            ]
        );
        // A window of 2: each query sees itself and one position back.
        assert_eq!(
            rows(causal_window_visible(3, 3, Some(2), &device)),
            [
                [true, false, false],
                [true, true, false],
                [false, true, true],
            ]
        );
        // The window spans the queries' prefix.
        assert_eq!(
            rows(causal_window_visible(1, 4, Some(3), &device)),
            [[false, true, true, true]]
        );
    }

    #[test]
    #[should_panic(expected = "sliding window of 0")]
    fn test_causal_window_visible_refuses_an_empty_window() {
        let _ = causal_window_visible(1, 1, Some(0), &cpu_device());
    }
}
