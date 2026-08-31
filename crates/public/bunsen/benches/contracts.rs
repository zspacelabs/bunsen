use bunsen::contracts::{
    ShapeContract,
    run_periodically,
    shape_contract,
};
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

static BATCH: usize = 2;
static HEIGHT: usize = 3;
static WIDTH: usize = 2;
static PADDING: usize = 4;
static CHANNELS: usize = 5;
static COLOR: usize = 4;

fn bench_unpack_shape_bchpwp(c: &mut Criterion) {
    static PATTERN: ShapeContract = shape_contract!["b", "c", "h" * "p", "w" * "p",];

    let shape = [BATCH, CHANNELS, HEIGHT * PADDING, WIDTH * PADDING];
    let env = [("p", PADDING), ("c", CHANNELS)];

    c.bench_function("unpack_shape hwp", |b| {
        b.iter(|| {
            let [b, h, w, c] = PATTERN.unpack_shape(&shape, &["b", "h", "w", "c"], &env);

            assert_eq!(b, BATCH);
            assert_eq!(h, HEIGHT);
            assert_eq!(w, WIDTH);
            assert_eq!(c, CHANNELS);
        })
    });
}
fn bench_unpack_shape(c: &mut Criterion) {
    static PATTERN: ShapeContract = shape_contract![
        _,
        "b",
        ...,
        "h"*"p",
        "w"*"p",
        "z"^3,
        "c"
    ];

    let shape = [
        12,
        BATCH,
        1,
        2,
        3,
        HEIGHT * PADDING,
        WIDTH * PADDING,
        COLOR * COLOR * COLOR,
        CHANNELS,
    ];
    let env = [("p", PADDING), ("c", CHANNELS)];

    c.bench_function("unpack_shape", |b| {
        b.iter(|| {
            let [b, h, w, c] = PATTERN.unpack_shape(&shape, &["b", "h", "w", "z"], &env);

            assert_eq!(b, BATCH);
            assert_eq!(h, HEIGHT);
            assert_eq!(w, WIDTH);
            assert_eq!(c, COLOR);
        })
    });
}

fn bench_assert_shape(c: &mut Criterion) {
    static PATTERN: ShapeContract = shape_contract![
        _,
        "b",
        ...,
        "h"*"p",
        "w"*"p",
        "z"^3,
        "c"
    ];

    let shape = [
        12,
        BATCH,
        1,
        2,
        3,
        HEIGHT * PADDING,
        WIDTH * PADDING,
        COLOR * COLOR * COLOR,
        CHANNELS,
    ];
    let env = [("p", PADDING), ("c", CHANNELS)];

    c.bench_function("assert_shape", |b| {
        b.iter(|| {
            PATTERN.assert_shape(&shape, &env);
        })
    });
}

fn bench_assert_shape_every_nth(c: &mut Criterion) {
    static PATTERN: ShapeContract = shape_contract![
        _,
        "b",
        ...,
        "h"*"p",
        "w"*"p",
        "z"^3,
        "c"
    ];

    let shape = [
        12,
        BATCH,
        1,
        2,
        3,
        HEIGHT * PADDING,
        WIDTH * PADDING,
        COLOR * COLOR * COLOR,
        CHANNELS,
    ];
    let env = [("p", PADDING), ("c", CHANNELS)];

    let mut group = c.benchmark_group("assert_shape_every_nth");
    group.warm_up_time(std::time::Duration::from_nanos(1));

    group.bench_function("assert_shape_every_nth", |b| {
        b.iter(|| {
            run_periodically!(PATTERN.assert_shape(&shape, &env));
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_unpack_shape,
    bench_unpack_shape_bchpwp,
    bench_assert_shape,
    bench_assert_shape_every_nth
);
criterion_main!(benches);
