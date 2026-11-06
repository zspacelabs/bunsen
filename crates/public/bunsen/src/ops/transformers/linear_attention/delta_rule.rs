//! The gated delta rule.

use burn::{
    Tensor,
    prelude::{
        Bool,
        s,
    },
};

use crate::{
    burner::tensor::{
        Widening,
        WideningScope,
    },
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
};

/// The gated delta rule's options, and the rule as methods: one position at
/// a time ([`recurrent`](Self::recurrent)), or a chunk of positions at a
/// time ([`chunked`](Self::chunked)), which agree.
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

    /// The positions [`chunked`](Self::chunked) takes at a time.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
}

/// [`GatedDeltaRule::chunk_size`]'s default, `transformers`' and FLA's.
fn default_chunk_size() -> usize {
    64
}

impl Default for GatedDeltaRule {
    /// The queries and keys L2-normalized, with FLA's epsilon of `1e-6`;
    /// chunks of 64.
    fn default() -> Self {
        Self {
            qk_l2norm_eps: Some(1e-6),
            chunk_size: default_chunk_size(),
        }
    }
}

/// The rule's inputs, `[B, H, T, ·]` in the dtype it runs in: the queries
/// normalized and scaled, the keys normalized.
struct Prepared {
    /// The rule's dtypes: the one it runs in, and the queries', which its
    /// outputs are narrowed to.
    scope: WideningScope,
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    /// The log decay.
    g: Tensor<3>,
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

    /// Sets [`chunk_size`](Self::chunk_size).
    pub fn with_chunk_size(
        mut self,
        chunk_size: usize,
    ) -> Self {
        self.chunk_size = chunk_size;
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
        let scope = Widening::F32.scope(q.dtype());
        let to = |x: Tensor<4>| scope.widen(x).swap_dims(1, 2);
        let q = self.l2norm(to(q)).div_scalar((d_k as f64).sqrt());
        let k = self.l2norm(to(k));
        let gates = |x: Tensor<3>| scope.widen(x).swap_dims(1, 2);
        let prepared = Prepared {
            scope,
            q,
            k,
            v: to(v),
            g: gates(g),
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
        let (dims @ [_, t, ..], p) = self.prepare(q, k, v, g, beta);
        let mut s = initial_state(state, dims, &p);
        let decay = p.g.clone().exp();
        let mut outputs = Vec::with_capacity(t);
        for i in 0..t {
            let at = |x: &Tensor<4>| x.clone().slice(s![.., .., i..=i, ..]);
            // [B, H, 1, 1]
            let gate = |x: &Tensor<3>| x.clone().slice(s![.., .., i..=i]).unsqueeze_dim::<4>(3);
            let (q_t, k_t, v_t) = (at(&p.q), at(&p.k), at(&p.v));
            s = s * gate(&decay);
            // [B, H, 1, d_v]: what the state reads at the key.
            let read = k_t.clone().matmul(s.clone());
            let delta = (v_t - read) * gate(&p.beta);
            s = s + k_t.swap_dims(2, 3).matmul(delta);
            outputs.push(q_t.matmul(s.clone()));
        }
        let out = p.scope.narrow(Tensor::cat(outputs, 2).swap_dims(1, 2));
        (out, s)
    }

    /// The rule, [`chunk_size`](Self::chunk_size) positions at a time, from
    /// `state`: the form a prefill runs, a chunk's updates condensed into a
    /// few matmuls (the WY form of Yang et al., *Parallelizing Linear
    /// Transformers with the Delta Rule over Sequence Length*, as
    /// `transformers`' `torch_chunk_gated_delta_rule` computes it), and a
    /// step per chunk, not per position. Agrees with
    /// [`recurrent`](Self::recurrent), whose arguments and returns it
    /// takes; the last chunk may be short.
    ///
    /// Within a chunk the updates solve a unit lower-triangular system,
    /// here by forward substitution, one row at a time (`transformers`'
    /// export path), for every chunk at once.
    pub fn chunked(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        g: Tensor<3>,
        beta: Tensor<3>,
        state: Option<Tensor<4>>,
    ) -> (Tensor<4>, Tensor<4>) {
        let c = self.chunk_size;
        assert!(c > 0, "a chunk of at least one position");
        let (dims @ [b, t, h, ..], p) = self.prepare(q, k, v, g, beta);
        let scope = p.scope;
        let mut s = initial_state(state, dims, &p);
        let device = p.q.device();
        let n = t.div_ceil(c);
        let pad = n * c - t;

        // Padded to whole chunks with zeros, which neither write nor decay;
        // then `[B, H, n, C, ·]`.
        let chunks4 = |x: Tensor<4>| {
            let x = if pad > 0 {
                let [_, _, _, d] = x.dims();
                let zeros = Tensor::zeros([b, h, pad, d], (&device, x.dtype()));
                Tensor::cat(vec![x, zeros], 2)
            } else {
                x
            };
            let [_, _, _, d] = x.dims();
            x.reshape([b, h, n, c, d])
        };
        let chunks3 = |x: Tensor<3>| {
            let x = if pad > 0 {
                let zeros = Tensor::zeros([b, h, pad], (&device, x.dtype()));
                Tensor::cat(vec![x, zeros], 2)
            } else {
                x
            };
            x.reshape([b, h, n, c])
        };
        let beta = chunks3(p.beta).unsqueeze_dim::<5>(4);
        let q = chunks4(p.q);
        let k = chunks4(p.k);
        let v_beta = chunks4(p.v) * beta.clone();
        let k_beta = k.clone() * beta;

        // [B, H, n, C]: the log of the decay from a chunk's start to each
        // of its positions; [B, H, n, C, C]: from position j to position i,
        // zero above the diagonal.
        let cum = chunks3(p.g).cumsum(3);
        let upper = Tensor::<2, Bool>::tril_mask([c, c], 0, &device).unsqueeze::<5>();
        let pairwise = (cum.clone().unsqueeze_dim::<5>(4) - cum.clone().unsqueeze_dim::<5>(3))
            .mask_fill(upper, f64::NEG_INFINITY)
            .exp();
        let k_t = k.clone().swap_dims(3, 4);
        let system = k_beta.clone().matmul(k_t.clone()) * pairwise.clone();
        let intra = q.clone().matmul(k_t) * pairwise;
        let cum_exp = cum.clone().exp().unsqueeze_dim::<5>(4);
        let decayed_k_beta = k_beta * cum_exp.clone();

        let inverse = unit_lower_inverse(system, c);
        let new_values = inverse.clone().matmul(v_beta);
        let k_cumdecay = inverse.matmul(decayed_k_beta);

        let q = q * cum_exp;
        let last = cum.clone().slice(s![.., .., .., c - 1..c]);
        let k = k * (last.clone() - cum).exp().unsqueeze_dim::<5>(4);
        // [B, H, n, 1, 1]
        let chunk_decay = last.exp().unsqueeze_dim::<5>(4);

        let at = |x: &Tensor<5>, i: usize| {
            x.clone()
                .slice(s![.., .., i..=i, .., ..])
                .squeeze_dim::<4>(2)
        };
        let mut outputs = Vec::with_capacity(n);
        for i in 0..n {
            let v_new = at(&new_values, i) - at(&k_cumdecay, i).matmul(s.clone());
            outputs.push(at(&q, i).matmul(s.clone()) + at(&intra, i).matmul(v_new.clone()));
            s = s * at(&chunk_decay, i) + at(&k, i).swap_dims(2, 3).matmul(v_new);
        }
        let out = scope.narrow(
            Tensor::cat(outputs, 2)
                .slice(s![.., .., ..t, ..])
                .swap_dims(1, 2),
        );
        (out, s)
    }
}

/// The rule's starting state: `state`, cast to the dtype the rule runs in,
/// or zeros.
fn initial_state(
    state: Option<Tensor<4>>,
    [b, _, h, d_k, d_v]: [usize; 5],
    p: &Prepared,
) -> Tensor<4> {
    let compute = p.scope.compute();
    match state {
        Some(state) => {
            assert_shape_contract_periodically!(
                ["B", "H", "Dk", "Dv"],
                &state.dims(),
                &[("B", b), ("H", h), ("Dk", d_k), ("Dv", d_v)]
            );
            state.cast(compute)
        }
        None => Tensor::zeros([b, h, d_k, d_v], (&p.q.device(), compute)),
    }
}

/// `(I + L)⁻¹` for `L`, `system`'s part below the diagonal (its diagonal
/// and above ignored), over the last two axes of `[.., C, C]`: forward
/// substitution, one row at a time.
fn unit_lower_inverse(
    system: Tensor<5>,
    c: usize,
) -> Tensor<5> {
    let device = system.device();
    let dtype = system.dtype();
    let mut a = system.tril(-1).neg();
    for i in 1..c {
        let row = a.clone().slice(s![.., .., .., i..=i, ..i]);
        let sub = a.clone().slice(s![.., .., .., ..i, ..i]);
        let next = row.clone() + (row.swap_dims(3, 4) * sub).sum_dim(3);
        a = a.slice_assign(s![.., .., .., i..=i, ..i], next);
    }
    a + Tensor::<2>::eye(c, &device).cast(dtype).unsqueeze::<5>()
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

    /// The worst absolute difference between two tensors.
    fn worst<const D: usize>(
        a: Tensor<D>,
        b: Tensor<D>,
    ) -> f64 {
        values(a)
            .iter()
            .zip(values(b))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// The chunked form against the recurrent one, on `device`: chunks that
    /// divide the sequence, that do not, longer than it; with and without
    /// an initial state and the L2 norm.
    fn check_chunked_is_recurrent(
        device: &burn::tensor::Device,
        tolerance: f64,
    ) {
        let mut largest = 0.0f64;
        for (t, chunk) in [(8, 4), (10, 4), (3, 4), (70, 64), (1, 64)] {
            let dims @ [b, _, h, d_k, d_v] = [2, t, 3, 4, 5];
            let (q, k, v, g, beta) = inputs(50 + t as u64, dims);
            let to = |x: Tensor<4>| x.to_device(device);
            let to3 = |x: Tensor<3>| x.to_device(device);
            let (q, k, v, g, beta) = (to(q), to(k), to(v), to3(g), to3(beta));
            let start = seeded_tensor::<4>(60, [b, h, d_k, d_v], Distribution::Default, device);
            for initial in [false, true] {
                for eps in [Some(1e-6), None] {
                    let rule = GatedDeltaRule::default()
                        .with_qk_l2norm_eps(eps)
                        .with_chunk_size(chunk);
                    let run = |chunked: bool| {
                        let args = (
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            g.clone(),
                            beta.clone(),
                            initial.then(|| start.clone()),
                        );
                        if chunked {
                            rule.chunked(args.0, args.1, args.2, args.3, args.4, args.5)
                        } else {
                            rule.recurrent(args.0, args.1, args.2, args.3, args.4, args.5)
                        }
                    };
                    let (out_c, state_c) = run(true);
                    let (out_r, state_r) = run(false);
                    assert_eq!(out_c.dims(), [b, t, h, d_v]);
                    let what = format!("T {t}, chunk {chunk}, initial {initial}, eps {eps:?}");
                    let (o, st) = (worst(out_c, out_r), worst(state_c, state_r));
                    assert!(o < tolerance, "{what}: outputs off by {o}");
                    assert!(st < tolerance, "{what}: states off by {st}");
                    largest = largest.max(o).max(st);
                }
            }
        }
        eprintln!("chunked against recurrent on {device:?}: off by at most {largest:.2e}");
    }

    /// The chunked form is the recurrent form, on the CPU.
    #[test]
    fn test_chunked_is_recurrent() {
        check_chunked_is_recurrent(&cpu_device(), 1e-5);
    }

    /// The chunked form is the recurrent form, on the performance device.
    #[test]
    #[serial_test::serial]
    fn test_chunked_is_recurrent_on_the_performance_device() {
        let device = crate::support::testing::performance_device();
        let _memory = crate::support::testing::DeviceMemoryGuard::new(&device);
        check_chunked_is_recurrent(&device, 1e-5);
    }

    /// A chunked prefill in parts, each from the state the last left, is
    /// the one-pass prefill: a part ending mid-chunk included.
    #[test]
    fn test_chunked_parts_agree_with_one_pass() {
        let dims = [1, 13, 2, 4, 3];
        let (q, k, v, g, beta) = inputs(70, dims);
        let rule = GatedDeltaRule::default().with_chunk_size(4);
        let (whole, whole_state) = rule.chunked(
            q.clone(),
            k.clone(),
            v.clone(),
            g.clone(),
            beta.clone(),
            None,
        );
        let mut state = None;
        let mut parts = Vec::new();
        for span in [0, 5, 6, 13].windows(2) {
            let r = span[0]..span[1];
            let (y, next) = rule.chunked(
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
        assert_tensors_close(&Tensor::cat(parts, 1), &whole, Tolerance::default());
        assert_tensors_close(&state.unwrap(), &whole_state, Tolerance::default());
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
