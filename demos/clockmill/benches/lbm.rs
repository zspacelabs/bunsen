use std::hint::black_box;

use bunsen::support::testing::PerfTestBackend;
use burn::{
    Tensor,
    prelude::{
        Bool,
        s,
    },
    tensor::{
        DType::{
            F32,
            F64,
        },
        Distribution,
    },
};
use clockmill::simulations::surface::fluids::lbm::d2q9::{
    collision::{
        bgk_collision,
        bgk_collision_with_spherical_reflection,
    },
    reflection::with_spherical_reflection,
    relaxation::RelaxationParam,
    space::LbmTables,
    streaming::stream_interior_windows,
};
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn bench_lbm_d2q9(c: &mut Criterion) {
    type B = PerfTestBackend;
    let device = Default::default();

    let n = 1000;

    let mut group = c.benchmark_group(format!("lbm:d2q9: {n}x{n}"));

    let relaxation = RelaxationParam::Omega(1.5);

    for dtype in [F32, F64] {
        let dist = Tensor::<B, 4>::random([n, n, 3, 3], Distribution::Default, &device);
        let dist = dist.cast(dtype);

        let solid_mask = Tensor::<B, 2, Bool>::full([n, n], false, &device);

        let lbm_tables = LbmTables::for_dist(&dist);

        group.bench_function(format!("{:?} bgk_collision", dtype).as_str(), |b| {
            b.iter(|| {
                let dist_col = bgk_collision(dist.clone(), relaxation, None, &lbm_tables);

                black_box(dist_col.mean().into_scalar());
            })
        });

        group.bench_function(format!("{:?} isotropic collision", dtype).as_str(), |b| {
            b.iter(|| {
                let dist_col = bgk_collision_with_spherical_reflection(
                    dist.clone(),
                    solid_mask.clone(),
                    relaxation,
                    None,
                    &lbm_tables,
                );

                black_box(dist_col.mean().into_scalar());
            })
        });

        group.bench_function(format!("{:?} streaming", dtype).as_str(), |b| {
            b.iter(|| {
                let stream_result = stream_interior_windows(dist.clone());

                black_box(stream_result);
            })
        });

        group.bench_function(format!("{:?} update", dtype).as_str(), |b| {
            b.iter(|| {
                let dist = with_spherical_reflection(
                    dist.clone(),
                    bgk_collision(dist.clone(), relaxation, None, &lbm_tables),
                    solid_mask.clone(),
                );

                let stream_result = stream_interior_windows(dist.clone());
                let dist = dist.slice_assign(s![1..-1, 1..-1], stream_result);

                black_box(dist.mean().into_scalar());
            })
        });
    }
}

criterion_group!(benches, bench_lbm_d2q9);
criterion_main!(benches);
