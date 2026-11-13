//! # Mixture-of-experts dispatch
//!
//! [`expert_dispatch`]: the one seam every sparse mixture of experts runs its
//! experts through. Given each row's chosen experts and their weights, it
//! returns the weighted sum of those experts' outputs; the router, the
//! shared expert and the block around it call only this, never a strategy
//! behind it.
//!
//! The experts are `SwiGLU` feed-forwards packed in one pair of tensors,
//! [`PackedExperts`], whose layout is part of the contract (a kernel reads the
//! same parameters):
//!
//! ```text
//! gate_up [E, 2H, W]: rows [0, H) expert e's gate, [H, 2H) its up projection
//! down    [E, W, H]
//! expert_e(x) = down[e] · (silu(gate_e · x) ⊙ (up_e · x))
//! ```
//!
//! `W` is the width the experts run at: the model's, or a latent's.
//!
//! How the rows reach their experts is an [`ExpertDispatchStrategy`]. Every
//! strategy computes the same function; [`DenseMasked`] is its definition,
//! and a strategy is accepted by agreeing with it.
//!
//! [`DenseMasked`]: ExpertDispatchStrategy::DenseMasked

use alloc::vec::Vec;

use burn::{
    Tensor,
    config::Config,
    prelude::{
        Int,
        TensorData,
        s,
    },
    tensor::{
        DType,
        IndexingUpdateOp,
        activation::silu,
    },
};

use crate::contracts::unpack_shape_contract;

/// A packed stack of `SwiGLU` experts, as [`expert_dispatch`] reads them.
#[derive(Clone, Debug)]
pub struct PackedExperts {
    /// `[E, 2H, W]`: each expert's gate rows, then its up rows.
    pub gate_up: Tensor<3>,

    /// `[E, W, H]`: each expert's down projection.
    pub down: Tensor<3>,
}

impl PackedExperts {
    /// The expert count, `E`.
    pub fn n_experts(&self) -> usize {
        self.gate_up.dims()[0]
    }

    /// The experts' hidden width, `H`.
    pub fn hidden(&self) -> usize {
        self.down.dims()[2]
    }

    /// The width the experts read and write, `W`.
    pub fn width(&self) -> usize {
        self.down.dims()[1]
    }

    /// `silu(gate) ⊙ up` of a `[.., 2H]` projection, `[.., H]`.
    fn activate<const D: usize>(
        projected: Tensor<D>,
        hidden: usize,
    ) -> Tensor<D> {
        let gate = projected.clone().slice_dim(D - 1, 0..hidden);
        let up = projected.slice_dim(D - 1, hidden..2 * hidden);
        silu(gate) * up
    }
}

/// How [`expert_dispatch`] brings rows to their experts.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum ExpertDispatchStrategy {
    /// Every expert on every row, weighted by a dense `[N, E]` matrix that is
    /// zero off the chosen experts: the seam's definition, and the reference
    /// every other strategy is held to. `E` times the work; for tests.
    DenseMasked,

    /// Rows grouped by expert: the chosen ids read on the host once, then
    /// every expert that any row chose run in one batched matmul over its
    /// rows, gathered, and scattered back. The buckets are padded to the
    /// largest one's size and the experts' count up, each to a power of two
    /// (into a scratch row, at weight zero), so a shape-keyed autotune sees
    /// recurring shapes; exact, since the padding is sized from the buckets.
    /// A prefill's form.
    Bucketed,

    /// Each row's `k` experts' weights gathered into a batch of their own,
    /// then one batched matmul: no host read, and `N k` copies of an
    /// expert's weights. A decode step's form.
    Gathered,

    /// [`Gathered`](Self::Gathered) for at most
    /// [`AUTO_GATHER_ROWS`](ExpertDispatchStrategy::AUTO_GATHER_ROWS) rows,
    /// [`Bucketed`](Self::Bucketed) past that. The default.
    Auto,
}

// By hand: `Config`'s derive copies the enum without `Default`'s, so a
// `#[default]` variant does not compile.
#[allow(clippy::derivable_impls)]
impl Default for ExpertDispatchStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

impl ExpertDispatchStrategy {
    /// The most rows [`Auto`](Self::Auto) gathers for.
    pub const AUTO_GATHER_ROWS: usize = 8;

    /// The strategy that runs `rows` rows: `Auto` resolved, the others as
    /// they are.
    pub fn resolve(
        self,
        rows: usize,
    ) -> Self {
        match self {
            Self::Auto if rows <= Self::AUTO_GATHER_ROWS => Self::Gathered,
            Self::Auto => Self::Bucketed,
            other => other,
        }
    }
}

/// The weighted sum of each row's chosen experts' outputs:
/// `y[n] = Σ_j weights[n, j] · expert_{ids[n, j]}(x[n])`.
///
/// The sum is taken in f32 and returned in `x`'s dtype.
///
/// # Arguments
/// - `x`: `[N, W]`, the rows.
/// - `ids`: `[N, k]`, each row's chosen experts, distinct within a row.
/// - `weights`: `[N, k]`, their weights.
/// - `experts`: the packed experts.
/// - `strategy`: how rows reach their experts.
///
/// # Returns
/// - `[N, W]`.
///
/// # Panics
/// If the shapes disagree, or (for
/// [`Bucketed`](ExpertDispatchStrategy::Bucketed)) an id is not an expert.
pub fn expert_dispatch(
    x: Tensor<2>,
    ids: Tensor<2, Int>,
    weights: Tensor<2>,
    experts: &PackedExperts,
    strategy: ExpertDispatchStrategy,
) -> Tensor<2> {
    let [n, w] = unpack_shape_contract!(
        ["N", "W"],
        &x.dims(),
        &["N", "W"],
        &[("W", experts.width())]
    );
    let [k] = unpack_shape_contract!(["N", "k"], &ids.dims(), &["k"], &[("N", n)]);
    crate::contracts::assert_shape_contract_periodically!(
        ["N", "k"],
        &weights.dims(),
        &[("N", n), ("k", k)]
    );
    let dtype = x.dtype();
    let y = match strategy.resolve(n) {
        ExpertDispatchStrategy::DenseMasked => dense_masked(x, ids, weights, experts),
        ExpertDispatchStrategy::Bucketed => bucketed(x, ids, weights, experts),
        ExpertDispatchStrategy::Gathered | ExpertDispatchStrategy::Auto => {
            gathered(x, ids, weights, experts)
        }
    };
    let y = y.cast(dtype);
    crate::contracts::assert_shape_contract_periodically!(
        ["N", "W"],
        &y.dims(),
        &[("N", n), ("W", w)]
    );
    y
}

/// Every expert on every row, `[E, N, W]`, weighted by the dense routing
/// matrix and summed.
fn dense_masked(
    x: Tensor<2>,
    ids: Tensor<2, Int>,
    weights: Tensor<2>,
    experts: &PackedExperts,
) -> Tensor<2> {
    let [n, w] = x.dims();
    let (e, h) = (experts.n_experts(), experts.hidden());
    // [E, N, 2H] -> [E, N, H] -> [E, N, W]
    let projected = x
        .unsqueeze::<3>()
        .expand([e, n, w])
        .matmul(experts.gate_up.clone().swap_dims(1, 2));
    let outputs = PackedExperts::activate(projected, h)
        .matmul(experts.down.clone().swap_dims(1, 2))
        .cast(DType::F32);
    // [N, E]: each row's weights at its chosen experts, zero elsewhere.
    let dense = Tensor::<2>::zeros([n, e], (&weights.device(), DType::F32)).scatter(
        1,
        ids,
        weights.cast(DType::F32),
        IndexingUpdateOp::Add,
    );
    (outputs * dense.swap_dims(0, 1).unsqueeze_dim::<3>(2))
        .sum_dim(0)
        .reshape([n, w])
}

/// Each row's `k` experts gathered: `[N k, 2H, W]` and `[N k, W, H]`, one
/// batched matmul each, weighted and summed over `k`.
fn gathered(
    x: Tensor<2>,
    ids: Tensor<2, Int>,
    weights: Tensor<2>,
    experts: &PackedExperts,
) -> Tensor<2> {
    let [n, w] = x.dims();
    let k = ids.dims()[1];
    let h = experts.hidden();
    let flat = ids.reshape([n * k]);
    let gate_up = experts.gate_up.clone().select(0, flat.clone());
    let down = experts.down.clone().select(0, flat);
    // [N k, 1, W]
    let rows = x
        .unsqueeze_dim::<3>(1)
        .expand([n, k, w])
        .reshape([n * k, 1, w]);
    let projected = rows.matmul(gate_up.swap_dims(1, 2));
    let outputs = PackedExperts::activate(projected, h)
        .matmul(down.swap_dims(1, 2))
        .reshape([n, k, w])
        .cast(DType::F32);
    (outputs * weights.cast(DType::F32).unsqueeze_dim::<3>(2))
        .sum_dim(1)
        .reshape([n, w])
}

/// The rows grouped by expert, all the hit experts run as one batched
/// matmul: `[E', M, W]`, `M` the largest bucket and `E'` the hit experts'
/// count, each rounded up to a power of two. The padding reads row 0 and a
/// zero weight, and is scattered into an `[N + 1, W]` sum's last row.
fn bucketed(
    x: Tensor<2>,
    ids: Tensor<2, Int>,
    weights: Tensor<2>,
    experts: &PackedExperts,
) -> Tensor<2> {
    let [n, w] = x.dims();
    let k = ids.dims()[1];
    let (e, h) = (experts.n_experts(), experts.hidden());
    let device = x.device();

    let chosen: Vec<i64> = ids
        .into_data()
        .try_into_vec_as::<i64>()
        .expect("expert ids as integers");
    // Per expert, the flat positions `row * k + j` that chose it.
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); e];
    for (position, &id) in chosen.iter().enumerate() {
        let id = usize::try_from(id)
            .ok()
            .filter(|&id| id < e)
            .unwrap_or_else(|| panic!("expert id {id} of {e} experts"));
        buckets[id].push(position);
    }
    let hit: Vec<usize> = (0..e).filter(|&i| !buckets[i].is_empty()).collect();
    let slots = hit.len().next_power_of_two();
    let depth = hit
        .iter()
        .map(|&i| buckets[i].len())
        .max()
        .unwrap_or(1)
        .next_power_of_two();

    // Per slot and depth: the row read, the row written, the weight's flat
    // position. Padding reads row 0, writes the scratch row `n`, and weighs
    // by the appended zero at `n k`.
    let cells = slots * depth;
    let (mut reads, mut writes, mut picks) = (
        Vec::with_capacity(cells),
        Vec::with_capacity(cells),
        Vec::with_capacity(cells),
    );
    let mut slot_experts = Vec::with_capacity(slots);
    for slot in 0..slots {
        let bucket = hit.get(slot).map_or(&[][..], |&i| &buckets[i][..]);
        slot_experts.push(hit.get(slot).copied().unwrap_or(hit[0]) as i64);
        for d in 0..depth {
            match bucket.get(d) {
                Some(&p) => {
                    reads.push((p / k) as i64);
                    writes.push((p / k) as i64);
                    picks.push(p as i64);
                }
                None => {
                    reads.push(0);
                    writes.push(n as i64);
                    picks.push((n * k) as i64);
                }
            }
        }
    }
    let index = |v: Vec<i64>| {
        let len = v.len();
        Tensor::<1, Int>::from_data(TensorData::new(v, [len]), &device)
    };

    // [E', 2H, W] and [E', W, H]: the stack itself when every expert is hit
    // in order, else the hit experts gathered.
    let (gate_up, down) = if slots == e && hit.len() == e {
        (experts.gate_up.clone(), experts.down.clone())
    } else {
        let slot_experts = index(slot_experts);
        (
            experts.gate_up.clone().select(0, slot_experts.clone()),
            experts.down.clone().select(0, slot_experts),
        )
    };
    // [E', M, W]
    let batch = x.select(0, index(reads)).reshape([slots, depth, w]);
    let out = PackedExperts::activate(batch.matmul(gate_up.swap_dims(1, 2)), h)
        .matmul(down.swap_dims(1, 2))
        .reshape([cells, w])
        .cast(DType::F32);

    let flat_weights = Tensor::cat(
        vec![
            weights.cast(DType::F32).reshape([n * k]),
            Tensor::zeros([1], (&device, DType::F32)),
        ],
        0,
    );
    let scaled = out * flat_weights.select(0, index(picks)).unsqueeze_dim(1);
    Tensor::<2>::zeros([n + 1, w], (&device, DType::F32))
        .select_assign(0, index(writes), scaled, IndexingUpdateOp::Add)
        .slice(s![..n])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Device,
        Distribution,
        Tolerance,
    };
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        assert_tensors_close,
        cpu_device,
        performance_device,
        seeded_tensor,
    };

    fn experts(
        e: usize,
        h: usize,
        w: usize,
        device: &Device,
    ) -> PackedExperts {
        PackedExperts {
            gate_up: seeded_tensor(1, [e, 2 * h, w], Distribution::Normal(0.0, 0.3), device),
            down: seeded_tensor(2, [e, w, h], Distribution::Normal(0.0, 0.3), device),
        }
    }

    /// `DenseMasked` written out row by row: each chosen expert's `SwiGLU`,
    /// weighted.
    #[test]
    fn test_dense_masked_is_the_definition() {
        let device = cpu_device();
        let (e, h, w) = (4, 6, 5);
        let experts = experts(e, h, w, &device);
        let x: Tensor<2> = seeded_tensor(3, [3, w], Distribution::Default, &device);
        let ids = Tensor::<2, Int>::from_data([[0, 2], [3, 0], [2, 1]], &device);
        let weights = Tensor::<2>::from_data([[0.7f32, 0.3], [0.5, 0.5], [0.1, 0.9]], &device);

        let expert = |e: usize, row: Tensor<2>| {
            let gate_up = experts
                .gate_up
                .clone()
                .slice(s![e..e + 1])
                .reshape([2 * h, w]);
            let down = experts.down.clone().slice(s![e..e + 1]).reshape([w, h]);
            let p = row.matmul(gate_up.transpose());
            let act = silu(p.clone().slice(s![.., ..h])) * p.slice(s![.., h..]);
            act.matmul(down.transpose())
        };
        let chosen = [[0, 2], [3, 0], [2, 1]];
        let ws = [[0.7f32, 0.3], [0.5, 0.5], [0.1, 0.9]];
        let expected = Tensor::cat(
            (0..3)
                .map(|r| {
                    let row = x.clone().slice(s![r..r + 1]);
                    expert(chosen[r][0], row.clone()) * ws[r][0]
                        + expert(chosen[r][1], row) * ws[r][1]
                })
                .collect(),
            0,
        );
        assert_tensors_close(
            &expert_dispatch(
                x,
                ids,
                weights,
                &experts,
                ExpertDispatchStrategy::DenseMasked,
            ),
            &expected,
            Tolerance::default(),
        );
    }

    /// Every strategy agrees with `DenseMasked`: on a batch where one expert
    /// is chosen by no row, one by most rows, with equal weights (ties), and
    /// bucket sizes that are not powers of two; and on a single row. On the
    /// CPU and on the performance device, whose scatters differ.
    #[test]
    #[serial]
    fn test_strategies_agree() {
        let gpu = performance_device();
        let _memory = DeviceMemoryGuard::new(&gpu);
        for device in [cpu_device(), gpu.clone()] {
            strategies_agree(&device);
            all_experts_hit_agree(&device);
        }
    }

    /// Every expert of a power-of-two stack chosen: `Bucketed` runs the stack
    /// as it is, without gathering it.
    fn all_experts_hit_agree(device: &Device) {
        let experts = experts(4, 3, 5, device);
        let ids = Tensor::<2, Int>::from_data([[0, 1], [2, 3], [3, 0]], device);
        let weights = Tensor::<2>::from_data([[0.4f32, 0.6], [0.5, 0.5], [0.2, 0.8]], device);
        let x: Tensor<2> = seeded_tensor(6, [3, 5], Distribution::Default, device);
        assert_tensors_close(
            &expert_dispatch(
                x.clone(),
                ids.clone(),
                weights.clone(),
                &experts,
                ExpertDispatchStrategy::Bucketed,
            ),
            &expert_dispatch(
                x,
                ids,
                weights,
                &experts,
                ExpertDispatchStrategy::DenseMasked,
            ),
            Tolerance::rel_abs(1e-5, 1e-6),
        );
    }

    fn strategies_agree(device: &Device) {
        let device = device.clone();
        let (e, h, w) = (5, 6, 4);
        let experts = experts(e, h, w, &device);
        // Expert 4 is never chosen; expert 0 by six of seven rows.
        let ids = Tensor::<2, Int>::from_data(
            [[0, 1], [0, 2], [3, 0], [0, 1], [2, 0], [1, 3], [0, 2]],
            &device,
        );
        let weights = Tensor::<2>::from_data(
            [
                [0.5f32, 0.5],
                [0.9, 0.1],
                [0.2, 0.8],
                [0.5, 0.5],
                [0.6, 0.4],
                [0.3, 0.7],
                [1.0, 0.0],
            ],
            &device,
        );
        let x: Tensor<2> = seeded_tensor(4, [7, w], Distribution::Default, &device);
        let tolerance = Tolerance::rel_abs(1e-5, 1e-6);
        for (rows, x, ids, weights) in [
            (7, x.clone(), ids.clone(), weights.clone()),
            (
                1,
                x.slice(s![2..3]),
                ids.slice(s![2..3]),
                weights.slice(s![2..3]),
            ),
        ] {
            let reference = expert_dispatch(
                x.clone(),
                ids.clone(),
                weights.clone(),
                &experts,
                ExpertDispatchStrategy::DenseMasked,
            );
            for strategy in [
                ExpertDispatchStrategy::Bucketed,
                ExpertDispatchStrategy::Gathered,
                ExpertDispatchStrategy::Auto,
            ] {
                assert_tensors_close(
                    &expert_dispatch(x.clone(), ids.clone(), weights.clone(), &experts, strategy),
                    &reference,
                    tolerance,
                );
                assert_eq!(
                    expert_dispatch(x.clone(), ids.clone(), weights.clone(), &experts, strategy)
                        .dims(),
                    [rows, w]
                );
            }
        }
    }

    /// `Auto` gathers for a few rows and buckets for more.
    #[test]
    fn test_auto_resolves_by_rows() {
        let auto = ExpertDispatchStrategy::default();
        assert_eq!(auto, ExpertDispatchStrategy::Auto);
        assert_eq!(auto.resolve(1), ExpertDispatchStrategy::Gathered);
        assert_eq!(
            auto.resolve(ExpertDispatchStrategy::AUTO_GATHER_ROWS),
            ExpertDispatchStrategy::Gathered
        );
        assert_eq!(
            auto.resolve(ExpertDispatchStrategy::AUTO_GATHER_ROWS + 1),
            ExpertDispatchStrategy::Bucketed
        );
        assert_eq!(
            ExpertDispatchStrategy::DenseMasked.resolve(1),
            ExpertDispatchStrategy::DenseMasked
        );
    }
}
