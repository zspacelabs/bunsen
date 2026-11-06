//! The gated delta rule.

use burn::{
    Tensor,
    prelude::s,
};

use crate::{
    burner::tensor::widen_half,
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
};

/// The gated delta rule's options, and the rule as methods
/// ([`recurrent`](Self::recurrent)).
///
/// Per head, a `[d_k, d_v]` state `S` is decayed, corrected toward the
/// value a key should read, and read by the query, at each position:
///
/// ```text
/// S ← exp(g) · S
/// S ← S + k ⊗ β (v - Sᵀ k)
/// o = Sᵀ q
/// ```
///
/// with `q` scaled by `1 / sqrt(d_k)`, and `q` and `k` first L2-normalized
/// when [`qk_l2norm_eps`](Self::qk_l2norm_eps) is set (FLA's
/// `use_qk_l2norm_in_kernel`, which Qwen3.5 sets). `g ≤ 0` is the decay in
/// log space, `β ∈ (0, 1)` the write strength. The rule runs in f32, or the
/// inputs' dtype when it is wider, as `transformers`' reference forms do;
/// the output is narrowed to the queries' dtype, and the state is kept in
/// the dtype it ran in.
///
/// Queries and keys come with as many heads as the values: a model with
/// fewer key heads repeats them before the rule, as `transformers` does.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct GatedDeltaRule {
    /// L2-normalize each query and key head, `x / sqrt(Σ x² + eps)`, with
    /// this epsilon; `None` takes them as given.
    pub qk_l2norm_eps: Option<f64>,
}

impl Default for GatedDeltaRule {
    /// The queries and keys L2-normalized, with FLA's epsilon of `1e-6`.
    fn default() -> Self {
        Self {
            qk_l2norm_eps: Some(1e-6),
        }
    }
}

/// The rule's inputs, `[B, H, T, ·]` in the dtype it runs in: the queries
/// normalized and scaled, the keys normalized.
struct Prepared {
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    decay: Tensor<3>,
    beta: Tensor<3>,
}

impl GatedDeltaRule {
    /// Sets [`qk_l2norm_eps`](Self::qk_l2norm_eps).
    pub fn with_qk_l2norm_eps(
        mut self,
        eps: Option<f64>,
    ) -> Self {
        self.qk_l2norm_eps = eps;
        self
    }

    /// `x / sqrt(Σ x² + eps)` over the last axis, when the rule normalizes.
    fn l2norm(
        &self,
        x: Tensor<4>,
    ) -> Tensor<4> {
        match self.qk_l2norm_eps {
            Some(eps) => {
                let norm = x.clone().square().sum_dim(3).add_scalar(eps).sqrt();
                x / norm
            }
            None => x,
        }
    }

    /// The inputs widened, transposed to `[B, H, T, ·]`, `q` and `k`
    /// normalized and `q` scaled; and their dims.
    fn prepare(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<3>,
        beta: Tensor<3>,
    ) -> ([usize; 5], Prepared) {
        let [b, t, h, d_k] =
            unpack_shape_contract!(["B", "T", "H", "Dk"], &q.dims(), &["B", "T", "H", "Dk"]);
        assert_shape_contract_periodically!(
            ["B", "T", "H", "Dk"],
            &k.dims(),
            &[("B", b), ("T", t), ("H", h), ("Dk", d_k)]
        );
        let [d_v] = unpack_shape_contract!(
            ["B", "T", "H", "Dv"],
            &v.dims(),
            &["Dv"],
            &[("B", b), ("T", t), ("H", h)]
        );
        for (name, x) in [("g", &g), ("beta", &beta)] {
            assert_eq!(x.dims(), [b, t, h], "the rule's `{name}` is [B, T, H]");
        }
        let compute = widen_half(q.dtype());
        let to = |x: Tensor<4>| x.cast(compute).swap_dims(1, 2);
        let q = self.l2norm(to(q)).div_scalar((d_k as f64).sqrt());
        let k = self.l2norm(to(k));
        let gates = |x: Tensor<3>| x.cast(compute).swap_dims(1, 2);
        let prepared = Prepared {
            q,
            k,
            v: to(v),
            decay: gates(g).exp(),
            beta: gates(beta),
        };
        ([b, t, h, d_k, d_v], prepared)
    }

    /// The rule, one position at a time, from `state`: the per-token form a
    /// decode steps, and the reference the chunked form is held to.
    ///
    /// # Arguments
    /// - `q`, `k`: `[B, T, H, d_k]`.
    /// - `v`: `[B, T, H, d_v]`.
    /// - `g`: `[B, T, H]`, the log decay (`≤ 0`).
    /// - `beta`: `[B, T, H]`, the write strength.
    /// - `state`: `[B, H, d_k, d_v]`, the state before the first position;
    ///   `None`, zeros.
    ///
    /// # Returns
    /// - `[B, T, H, d_v]`, the outputs, in `q`'s dtype;
    /// - `[B, H, d_k, d_v]`, the state after the last position, in the dtype
    ///   the rule ran in.
    pub fn recurrent(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<3>,
        beta: Tensor<3>,
        state: Option<Tensor<4>>,
    ) -> (Tensor<4>, Tensor<4>) {
        let dtype = q.dtype();
        let device = q.device();
        let ([b, t, h, d_k, d_v], p) = self.prepare(q, k, v, g, beta);
        let compute = p.q.dtype();
        let mut s = match state {
            Some(state) => {
                assert_shape_contract_periodically!(
                    ["B", "H", "Dk", "Dv"],
                    &state.dims(),
                    &[("B", b), ("H", h), ("Dk", d_k), ("Dv", d_v)]
                );
                state.cast(compute)
            }
            None => Tensor::zeros([b, h, d_k, d_v], (&device, compute)),
        };
        let mut outputs = Vec::with_capacity(t);
        for i in 0..t {
            let at = |x: &Tensor<4>| x.clone().slice(s![.., .., i..=i, ..]);
            // [B, H, 1, 1]
            let gate = |x: &Tensor<3>| x.clone().slice(s![.., .., i..=i]).unsqueeze_dim::<4>(3);
            let (q_t, k_t, v_t) = (at(&p.q), at(&p.k), at(&p.v));
            s = s * gate(&p.decay);
            // [B, H, 1, d_v]: what the state reads at the key.
            let read = k_t.clone().matmul(s.clone());
            let delta = (v_t - read) * gate(&p.beta);
            s = s + k_t.swap_dims(2, 3).matmul(delta);
            outputs.push(q_t.matmul(s.clone()));
        }
        let out = Tensor::cat(outputs, 2).swap_dims(1, 2).cast(dtype);
        (out, s)
    }
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

    /// Seeded inputs of the rule's shapes: `g` in `(-1, 0)`, `β` in `(0, 1)`.
    fn inputs(
        seed: u64,
        [b, t, h, d_k, d_v]: [usize; 5],
    ) -> (Tensor<4>, Tensor<4>, Tensor<4>, Tensor<3>, Tensor<3>) {
        let device = cpu_device();
        (
            seeded_tensor::<4>(seed, [b, t, h, d_k], Distribution::Default, &device),
            seeded_tensor::<4>(seed + 1, [b, t, h, d_k], Distribution::Default, &device),
            seeded_tensor::<4>(seed + 2, [b, t, h, d_v], Distribution::Default, &device),
            seeded_tensor::<3>(
                seed + 3,
                [b, t, h],
                Distribution::Uniform(-1.0, 0.0),
                &device,
            ),
            seeded_tensor::<3>(
                seed + 4,
                [b, t, h],
                Distribution::Uniform(0.0, 1.0),
                &device,
            ),
        )
    }

    /// The rule written out in scalars, one head at a time, in f64:
    /// `transformers`' `torch_recurrent_gated_delta_rule` loop.
    #[allow(clippy::too_many_arguments)]
    fn scalar_rule(
        [b, t, h, d_k, d_v]: [usize; 5],
        q: &[f64],
        k: &[f64],
        v: &[f64],
        g: &[f64],
        beta: &[f64],
        state: Option<&[f64]>,
        l2norm: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut out = vec![0.0; b * t * h * d_v];
        let mut s = state.map_or_else(|| vec![0.0; b * h * d_k * d_v], <[f64]>::to_vec);
        let norm = |x: &[f64]| -> Vec<f64> {
            if !l2norm {
                return x.to_vec();
            }
            let n = (x.iter().map(|a| a * a).sum::<f64>() + 1e-6).sqrt();
            x.iter().map(|a| a / n).collect()
        };
        for row in 0..b {
            for head in 0..h {
                let st = &mut s[(row * h + head) * d_k * d_v..][..d_k * d_v];
                for pos in 0..t {
                    let base = (row * t + pos) * h + head;
                    let q_t: Vec<f64> = norm(&q[base * d_k..][..d_k])
                        .iter()
                        .map(|a| a / (d_k as f64).sqrt())
                        .collect();
                    let k_t = norm(&k[base * d_k..][..d_k]);
                    let v_t = &v[base * d_v..][..d_v];
                    let decay = g[base].exp();
                    st.iter_mut().for_each(|x| *x *= decay);
                    for j in 0..d_v {
                        let read: f64 = (0..d_k).map(|i| st[i * d_v + j] * k_t[i]).sum();
                        let delta = (v_t[j] - read) * beta[base];
                        for i in 0..d_k {
                            st[i * d_v + j] += k_t[i] * delta;
                        }
                    }
                    for j in 0..d_v {
                        out[base * d_v + j] = (0..d_k).map(|i| st[i * d_v + j] * q_t[i]).sum();
                    }
                }
            }
        }
        (out, s)
    }

    /// The recurrent form against the scalar rule: with and without an
    /// initial state, with and without the L2 norm.
    #[test]
    fn test_recurrent_against_the_scalar_rule() {
        let device = cpu_device();
        let dims @ [b, t, h, d_k, d_v] = [2, 5, 3, 4, 6];
        let (q, k, v, g, beta) = inputs(10, dims);
        let start = seeded_tensor::<4>(20, [b, h, d_k, d_v], Distribution::Default, &device);
        for initial in [false, true] {
            for eps in [Some(1e-6), None] {
                let rule = GatedDeltaRule::default().with_qk_l2norm_eps(eps);
                let (out, state) = rule.recurrent(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    g.clone(),
                    beta.clone(),
                    initial.then(|| start.clone()),
                );
                assert_eq!(out.dims(), [b, t, h, d_v]);
                assert_eq!(state.dims(), [b, h, d_k, d_v]);
                let start_values = values(start.clone());
                let (want_out, want_state) = scalar_rule(
                    dims,
                    &values(q.clone()),
                    &values(k.clone()),
                    &values(v.clone()),
                    &values(g.clone()),
                    &values(beta.clone()),
                    initial.then_some(start_values.as_slice()),
                    eps.is_some(),
                );
                for (got, want, what) in [
                    (values(out), want_out, "out"),
                    (values(state), want_state, "state"),
                ] {
                    let worst = got
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        worst < 1e-4,
                        "{what} (initial {initial}, eps {eps:?}): off by {worst}"
                    );
                }
            }
        }
    }

    /// Run in parts, each from the state the last left, the rule is the
    /// one-pass rule: one position at a time (a decode) included.
    #[test]
    fn test_parts_agree_with_one_pass() {
        let dims @ [_, t, ..] = [2, 7, 2, 4, 3];
        let (q, k, v, g, beta) = inputs(30, dims);
        let rule = GatedDeltaRule::default();
        let (whole, whole_state) = rule.recurrent(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
        );
        for cuts in [vec![0, 3, 7], vec![0, 1, 2, 3, 4, 5, 6, 7]] {
            let mut state = None;
            let mut parts = Vec::new();
            for span in cuts.windows(2) {
                let r = span[0]..span[1];
                let (y, next) = rule.recurrent(
                    q.clone().slice(s![.., r.clone(), .., ..]),
                    k.clone().slice(s![.., r.clone(), .., ..]),
                    v.clone().slice(s![.., r.clone(), .., ..]),
                    g.clone().slice(s![.., r.clone(), ..]),
                    beta.clone().slice(s![.., r, ..]),
                    state,
                );
                parts.push(y);
                state = Some(next);
            }
            assert_eq!(cuts.last(), Some(&t));
            assert_tensors_close(&Tensor::cat(parts, 1), &whole, Tolerance::default());
            assert_tensors_close(&state.unwrap(), &whole_state, Tolerance::default());
        }
    }

    /// A zero write strength leaves the state the decayed state; a zero
    /// decay (`g = 0`) with it, the state unchanged.
    #[test]
    fn test_zero_beta_only_decays() {
        let device = cpu_device();
        let dims @ [b, t, h, d_k, d_v] = [1, 3, 2, 4, 3];
        let (q, k, v, _, _) = inputs(40, dims);
        let start = seeded_tensor::<4>(41, [b, h, d_k, d_v], Distribution::Default, &device);
        let (_, state) = GatedDeltaRule::default().recurrent(
            q,
            k,
            v,
            Tensor::zeros([b, t, h], &device),
            Tensor::zeros([b, t, h], &device),
            Some(start.clone()),
        );
        assert_tensors_close(&state, &start, Tolerance::default());
    }
}
