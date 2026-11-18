//! The mixture-of-experts dispatch seam at ZONOS3-Mini's shapes, per
//! strategy, cold and warm.
//!
//! 64 experts of hidden width 1536 in a 256-wide latent, top 2: a decode
//! step (1 row), a speaker-guided step (2 rows), a prefill (72 rows) and a
//! long one (512 rows, `Bucketed` alone). Random weights and random
//! distinct ids. The first
//! call of each case is timed once and printed as `cold` (the backend's
//! autotune is shape-keyed, and `Bucketed`'s batch sizes vary with the
//! routing); criterion's numbers are the warm ones.
//!
//! `cargo bench -p bunsen --features wgpu --bench moe_dispatch`.

use std::time::Instant;

use bunsen::{
    ops::transformers::moe::{
        ExpertDispatchStrategy,
        PackedExperts,
        expert_dispatch,
    },
    support::testing::performance_device,
};
use burn::{
    Tensor,
    prelude::{
        Device,
        Int,
        TensorData,
    },
    tensor::Distribution,
};
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

const EXPERTS: usize = 64;
const WIDTH: usize = 256;
const HIDDEN: usize = 1536;
const TOP_K: usize = 2;

/// `rows` rows of `TOP_K` distinct experts each, from a fixed linear
/// congruential stream.
fn routes(
    rows: usize,
    device: &Device,
) -> (Tensor<2, Int>, Tensor<2>) {
    let mut state = 0x2545_f491_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as usize % EXPERTS
    };
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for _ in 0..rows {
        let first = next();
        let mut second = next();
        while second == first {
            second = next();
        }
        ids.extend([first as i64, second as i64]);
    }
    let weights = Tensor::random([rows, TOP_K], Distribution::Uniform(0.0, 1.0), device);
    (
        Tensor::from_data(TensorData::new(ids, [rows, TOP_K]), device),
        weights,
    )
}

fn bench_moe_dispatch(c: &mut Criterion) {
    let device = performance_device();
    let experts = PackedExperts {
        gate_up: Tensor::random(
            [EXPERTS, 2 * HIDDEN, WIDTH],
            Distribution::Normal(0.0, 0.02),
            &device,
        ),
        down: Tensor::random(
            [EXPERTS, WIDTH, HIDDEN],
            Distribution::Normal(0.0, 0.02),
            &device,
        ),
    };
    for rows in [1, 2, 72, 512] {
        let x: Tensor<2> = Tensor::random([rows, WIDTH], Distribution::Default, &device);
        let (ids, weights) = routes(rows, &device);
        // Gathered copies two slabs of weights per row (~9.4 MB each here),
        // and DenseMasked runs every expert on every row: both only up to a
        // prefill's size.
        let mut strategies = vec![ExpertDispatchStrategy::Bucketed];
        if rows <= 72 {
            strategies.extend([
                ExpertDispatchStrategy::Gathered,
                ExpertDispatchStrategy::DenseMasked,
            ]);
        }
        for strategy in strategies {
            let name = format!("{strategy:?} {rows} rows");
            let run = || {
                expert_dispatch(x.clone(), ids.clone(), weights.clone(), &experts, strategy)
                    .into_data()
            };
            let cold = Instant::now();
            let _ = run();
            eprintln!("cold {name}: {:?}", cold.elapsed());
            c.bench_function(&name, |b| b.iter(run));
        }
    }
}

criterion_group! {
    name = moe_dispatch;
    config = Criterion::default().sample_size(10);
    targets = bench_moe_dispatch
}
criterion_main!(moe_dispatch);
