//! # Attention residuals
//!
//! [`attn_res_route`]: Full Attention Residuals' routing (ZONOS3's
//! `full_attnres`), which takes the place of a residual stream. Each sublayer
//! reads a learned, input-dependent convex mixture of a bank of sources (the
//! embedding and every earlier sublayer's output), instead of their running
//! sum.

use burn::{
    Tensor,
    tensor::activation::softmax,
};

use crate::{
    burner::tensor::Widening,
    contracts::unpack_shape_contract,
    ops::norm::RmsNormOptions,
};

/// The mixture of `sources` that `query` routes to, per position:
///
/// ```text
/// keys   = rms_norm(bank, eps)            (weightless, over the width)
/// scores = Σ_d keys[s, .., d] · query[d]  (widened to f32)
/// w      = softmax_s(scores)              (widened, then the bank's dtype)
/// out    = Σ_s w[s, ..] · bank[s, ..]
/// ```
///
/// Each position weighs the sources by its own scores, so the mixture is
/// per position, and causal when every source is.
///
/// # Arguments
/// - `sources`: the bank, `S` tensors of `[B, T, D]`, in one dtype.
/// - `query`: `[D]`, the route's learned query.
/// - `eps`: the keys' RMS-norm epsilon.
///
/// # Returns
/// - `[B, T, D]`, in the sources' dtype.
///
/// # Panics
/// If `sources` is empty.
pub fn attn_res_route(
    sources: &[Tensor<3>],
    query: Tensor<1>,
    eps: f32,
) -> Tensor<3> {
    assert!(!sources.is_empty(), "a route over no sources");
    let [b, t, d] = unpack_shape_contract!(["B", "T", "D"], &sources[0].dims(), &["B", "T", "D"]);
    assert_eq!(
        query.dims(),
        [d],
        "a route's query is as wide as its sources"
    );
    let w = Widening::F32.scope(sources[0].dtype());

    // [S, B, T, D]
    let bank = Tensor::stack::<4>(sources.to_vec(), 0);
    let keys = RmsNormOptions::default().with_eps(eps).norm(bank.clone());
    // [S, B, T, 1]
    let scores = (w.widen(keys) * w.widen(query).reshape([1, 1, 1, d])).sum_dim(3);
    let weights = w.narrow(softmax(scores, 0));
    let out = (bank * weights).sum_dim(0).reshape([b, t, d]);

    #[cfg(any(debug_assertions, test))]
    crate::contracts::assert_shape_contract_periodically!(
        ["B", "T", "D"],
        &out.dims(),
        &[("B", b), ("T", t), ("D", d)]
    );
    out
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        DType,
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    /// The route written out per source: each source's keys normalized by
    /// hand, scored against the query, softmaxed across the sources.
    #[test]
    fn test_route_is_the_formula() {
        let device = cpu_device();
        let eps = 1e-5;
        let sources: Vec<Tensor<3>> = (0..3)
            .map(|i| seeded_tensor(i, [2, 4, 6], Distribution::Default, &device))
            .collect();
        let query: Tensor<1> = seeded_tensor(9, [6], Distribution::Default, &device);

        let scores: Vec<Tensor<3>> = sources
            .iter()
            .map(|s| {
                let rms = s
                    .clone()
                    .square()
                    .mean_dim(2)
                    .add_scalar(f64::from(eps))
                    .sqrt();
                ((s.clone() / rms) * query.clone().reshape([1, 1, 6])).sum_dim(2)
            })
            .collect();
        let weights = softmax(Tensor::cat(scores, 2), 2);
        let expected = sources
            .iter()
            .enumerate()
            .map(|(i, s)| s.clone() * weights.clone().slice_dim(2, i..i + 1))
            .reduce(|a, b| a + b)
            .unwrap();

        assert_tensors_close(
            &attn_res_route(&sources, query, eps),
            &expected,
            Tolerance::default(),
        );
    }

    /// One source routes to itself; a zero query mixes the sources evenly.
    #[test]
    fn test_route_edges() {
        let device = cpu_device();
        let sources: Vec<Tensor<3>> = (0..4)
            .map(|i| seeded_tensor(i, [1, 3, 5], Distribution::Default, &device))
            .collect();
        let query: Tensor<1> = seeded_tensor(7, [5], Distribution::Default, &device);
        assert_tensors_close(
            &attn_res_route(&sources[..1], query, 1e-5),
            &sources[0],
            Tolerance::default(),
        );
        let mean = sources.iter().cloned().reduce(|a, b| a + b).unwrap() / 4.0;
        assert_tensors_close(
            &attn_res_route(&sources, Tensor::zeros([5], &device), 1e-5),
            &mean,
            Tolerance::default(),
        );
    }

    /// A half bank is scored and softmaxed in f32, its weights rounded once
    /// to its dtype; an f64 bank is routed in f64 throughout.
    #[test]
    fn test_route_widens() {
        let device = cpu_device();
        let eps = 1e-5;
        let sources: Vec<Tensor<3>> = (0..3)
            .map(|i| seeded_tensor(i, [2, 4, 6], Distribution::Default, &device))
            .collect();
        let query: Tensor<1> = seeded_tensor(9, [6], Distribution::Default, &device);
        let route_by_hand = |dtype: DType, scores_in: DType| {
            let bank =
                Tensor::stack::<4>(sources.iter().map(|s| s.clone().cast(dtype)).collect(), 0);
            let keys = RmsNormOptions::default().with_eps(eps).norm(bank.clone());
            let scores = (keys.cast(scores_in)
                * query
                    .clone()
                    .cast(dtype)
                    .cast(scores_in)
                    .reshape([1, 1, 1, 6]))
            .sum_dim(3);
            (bank * softmax(scores, 0).cast(dtype))
                .sum_dim(0)
                .reshape([2, 4, 6])
        };
        let route = |dtype: DType| {
            let bank: Vec<Tensor<3>> = sources.iter().map(|s| s.clone().cast(dtype)).collect();
            attn_res_route(&bank, query.clone().cast(dtype), eps)
        };

        let half = route(DType::BF16);
        assert_eq!(half.dtype(), DType::BF16);
        half.into_data()
            .assert_eq(&route_by_hand(DType::BF16, DType::F32).into_data(), true);

        assert!(device.supports_dtype(DType::F64));
        let wide = route(DType::F64);
        assert_eq!(wide.dtype(), DType::F64);
        wide.into_data()
            .assert_eq(&route_by_hand(DType::F64, DType::F64).into_data(), true);
    }
}
