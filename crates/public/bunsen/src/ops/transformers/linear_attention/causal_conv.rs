//! The causal depthwise convolution over time.

use burn::{
    Tensor,
    prelude::s,
};

use crate::contracts::{
    assert_shape_contract_periodically,
    unpack_shape_contract,
};

/// A causal depthwise convolution over time: each channel's own kernel over
/// its own past, `y[t, c] = Σ_j w[c, j] · x[t - (K - 1) + j, c]`, the inputs
/// before the first position taken from `state` (zeros without one).
///
/// The gated `DeltaNet`'s short convolution (`transformers`' `causal_conv1d_fn`
/// and `causal_conv1d_update`, without the activation, which is the
/// caller's): a `Conv1d` with `groups = channels`, `padding = K - 1`, its
/// output cut to the input's length. Run in `x`'s dtype, the kernel cast to
/// it.
///
/// # Arguments
/// - `x`: `[B, T, C]`, the inputs at the next `T` positions.
/// - `weight`: `[C, K]`, each channel's kernel, oldest tap first; `K ≥ 2`.
/// - `state`: `[B, K - 1, C]`, the `K - 1` inputs before `x`, oldest first;
///   `None`, zeros (the sequence starts at `x`).
///
/// # Returns
/// - `[B, T, C]`, the convolution at `x`'s positions, in `x`'s dtype;
/// - `[B, K - 1, C]`, the last `K - 1` inputs, `x`'s end (with the state's tail
///   before it when `T < K - 1`): the next call's `state`, in `x`'s dtype.
///
/// # Panics
/// If the shapes disagree, or `K < 2`.
pub fn causal_depthwise_conv1d(
    x: Tensor<3>,
    weight: Tensor<2>,
    state: Option<Tensor<3>>,
) -> (Tensor<3>, Tensor<3>) {
    let [b, t, c] = unpack_shape_contract!(["B", "T", "C"], &x.dims(), &["B", "T", "C"]);
    let [k] = unpack_shape_contract!(["C", "K"], &weight.dims(), &["K"], &[("C", c)]);
    assert!(k >= 2, "a causal convolution of at least 2 taps; got {k}");
    let dtype = x.dtype();
    let device = x.device();
    let prev = match state {
        Some(state) => {
            assert_shape_contract_periodically!(
                ["B", "K1", "C"],
                &state.dims(),
                &[("B", b), ("K1", k - 1), ("C", c)]
            );
            state.cast(dtype)
        }
        None => Tensor::zeros([b, k - 1, c], (&device, dtype)),
    };
    // [B, K - 1 + T, C]: the window's inputs, oldest first.
    let padded = Tensor::cat(vec![prev, x], 1);
    let weight = weight.cast(dtype);
    let y = (0..k)
        .map(|j| {
            padded.clone().slice(s![.., j..j + t, ..])
                * weight.clone().slice(s![.., j..=j]).reshape([1, 1, c])
        })
        .reduce(|acc, term| acc + term)
        .expect("at least two taps");
    let next = padded.slice(s![.., t.., ..]);
    (y, next)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    fn values<const D: usize>(t: Tensor<D>) -> Vec<f64> {
        t.into_data().try_into_vec_as::<f64>().unwrap()
    }

    /// The convolution as written out, one output at a time, against the
    /// op: zeros before the first input, each channel its own kernel.
    #[test]
    fn test_against_the_sum() {
        let device = cpu_device();
        let (b, t, c, k) = (2, 5, 3, 4);
        let x = seeded_tensor::<3>(1, [b, t, c], Distribution::Default, &device);
        let w = seeded_tensor::<2>(2, [c, k], Distribution::Default, &device);
        let (y, next) = causal_depthwise_conv1d(x.clone(), w.clone(), None);
        assert_eq!(y.dims(), [b, t, c]);
        assert_eq!(next.dims(), [b, k - 1, c]);

        let (xs, ws, ys) = (values(x.clone()), values(w), values(y));
        for row in 0..b {
            for pos in 0..t {
                for ch in 0..c {
                    let mut sum = 0.0;
                    for j in 0..k {
                        let src = pos as isize - (k as isize - 1) + j as isize;
                        if src >= 0 {
                            sum += ws[ch * k + j] * xs[(row * t + src as usize) * c + ch];
                        }
                    }
                    let got = ys[(row * t + pos) * c + ch];
                    assert!(
                        (got - sum).abs() < 1e-5,
                        "[{row}, {pos}, {ch}]: {got} vs {sum}"
                    );
                }
            }
        }
        // The state is the last K - 1 inputs.
        assert_tensors_close(
            &next,
            &x.slice(s![.., t - (k - 1).., ..]),
            Tolerance::default(),
        );
    }

    /// Run in parts, each from the state the last left, the convolution is
    /// the one-pass convolution: parts shorter than the kernel too, one
    /// position at a time (a decode) included.
    #[test]
    fn test_parts_agree_with_one_pass() {
        let device = cpu_device();
        let (b, t, c, k) = (2, 9, 4, 4);
        let x = seeded_tensor::<3>(3, [b, t, c], Distribution::Default, &device);
        let w = seeded_tensor::<2>(4, [c, k], Distribution::Default, &device);
        let (whole, whole_next) = causal_depthwise_conv1d(x.clone(), w.clone(), None);
        for cuts in [
            vec![0, 4, 9],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            vec![0, 2, 7, 9],
        ] {
            let mut state = None;
            let mut parts = Vec::new();
            for span in cuts.windows(2) {
                let (y, next) = causal_depthwise_conv1d(
                    x.clone().slice(s![.., span[0]..span[1], ..]),
                    w.clone(),
                    state,
                );
                parts.push(y);
                state = Some(next);
            }
            assert_tensors_close(&Tensor::cat(parts, 1), &whole, Tolerance::default());
            assert_tensors_close(&state.unwrap(), &whole_next, Tolerance::default());
        }
    }

    #[test]
    #[should_panic(expected = "at least 2 taps")]
    fn test_one_tap() {
        let device = cpu_device();
        let _ = causal_depthwise_conv1d(
            Tensor::zeros([1, 2, 3], &device),
            Tensor::zeros([3, 1], &device),
            None,
        );
    }
}
