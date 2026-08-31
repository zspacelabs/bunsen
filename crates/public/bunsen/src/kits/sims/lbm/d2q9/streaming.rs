//! # Streaming Operations

use burn::{
    Tensor,
    prelude::{
        Backend,
        s,
    },
};

use crate::kits::sims::lbm::d2q9::space;

/// Applies the streaming update step to the non-border cells of a population.
///
/// # Arguments
///
/// - `dist`: a `[H, W, VY=3, VX=3]` population distribution.
///
/// # Returns
/// - The updated `[H[1:-1], W[1:-1], VY=3, VX=3]` interior.
pub fn stream_interior_windows<B: Backend>(dist: Tensor<B, 4>) -> Tensor<B, 4> {
    #[cfg(debug_assertions)]
    let [h, w] = crate::contracts::unpack_shape_contract!(
        ["H", "W", "VY", "VX"],
        dist.shape().as_slice(),
        &["H", "W"],
        &[("VY", 3), ("VX", 3)]
    );

    let windows = space::dist_windows(dist);

    // Timing: crutcher, Oct 2025:
    // cat([cat([tensor,]),]) is ~10% faster than cat([tensor,]).reshape([...,
    // 3, 3]) Timing: crutcher, Oct 2025:
    // This also beats empty() + 0..3 0..3 slice_assign by ~8%
    let result: Tensor<B, 4> = Tensor::cat(
        (0..3)
            .map(|vy| -> Tensor<B, 4> {
                let source_vy = 2 - vy;

                Tensor::cat(
                    (0..3)
                        .map(|vx| -> Tensor<B, 4> {
                            let source_vx = 2 - vx;

                            windows
                                .clone()
                                .slice(s![.., .., vy, vx, source_vy, source_vx])
                                .squeeze_dims::<4>(&[-2, -1])
                        })
                        .collect(),
                    3,
                )
            })
            .collect(),
        2,
    );

    #[cfg(debug_assertions)]
    crate::contracts::assert_shape_contract_periodically!(
        ["H" - 2, "W" - 2, "VY", "VX"],
        result.shape().as_slice(),
        &[("H", h), ("W", w), ("VY", 3), ("VX", 3)]
    );

    result
}

/// Applies the streaming update step to a population.
///
/// This combines [`stream_interior_windows`] with edge outflow-streaming.
///
/// All edge outflow is clipped to the boundary. Closed universe sims should
/// have zero edge outflow before calling this operation.
///
/// # Arguments
///
/// - `dist`: a `[H, W, VY=3, VX=3]` population distribution.
///
/// # Returns
/// - The updated `[H, W, VY=3, VX=3]` distribution.
pub fn outflow_clipping_stream<B: Backend>(thermal_dist: Tensor<B, 4>) -> Tensor<B, 4> {
    thermal_dist
        .zeros_like()
        .cast(thermal_dist.dtype()) // TODO: remove when zeros_like() supports dtype.
        // stream top edges.
        // Top [1, 1]
        .slice_assign(s![0, .., 1, 1], thermal_dist.clone().slice(s![0, .., 1, 1]))
        // e[-1, -1]; v[0, 0]
        .slice_assign(
            s![0, ..-1, 0, 0],
            thermal_dist.clone().slice(s![1, 1.., 0, 0]),
        )
        // e[-1, 0]; v[0, 1]
        .slice_assign(s![0, .., 0, 1], thermal_dist.clone().slice(s![1, .., 0, 1]))
        // e[-1, 1]; v[0, 2]
        .slice_assign(
            s![0, 1.., 0, 2],
            thermal_dist.clone().slice(s![1, ..-1, 0, 2]),
        )
        // stream bottom edges.
        // Bottom [1, 1]
        .slice_assign(
            s![-1, .., 1, 1],
            thermal_dist.clone().slice(s![-1, .., 1, 1]),
        )
        // e[1, -1]; v[2, 0]
        .slice_assign(
            s![-1, ..-1, 2, 0],
            thermal_dist.clone().slice(s![-2, 1.., 2, 0]),
        )
        // e[1, 0]; v[2, 1]
        .slice_assign(
            s![-1, .., 2, 1],
            thermal_dist.clone().slice(s![-2, .., 2, 1]),
        )
        // e[1, 1]; v[2, 2]
        .slice_assign(
            s![-1, 1.., 2, 2],
            thermal_dist.clone().slice(s![-2, ..-1, 2, 2]),
        )
        // stream left edges.
        // Left [1, 1]
        .slice_assign(s![.., 0, 1, 1], thermal_dist.clone().slice(s![.., 0, 1, 1]))
        // e[-1, -1]; v[0, 0]
        .slice_assign(
            s![..-1, 0, 0, 0],
            thermal_dist.clone().slice(s![1.., 1, 0, 0]),
        )
        // e[0, -1]; v[1, 0]
        .slice_assign(s![.., 0, 1, 0], thermal_dist.clone().slice(s![.., 1, 1, 0]))
        // e[1, -1]; v[2, 0]
        .slice_assign(
            s![1.., 0, 2, 0],
            thermal_dist.clone().slice(s![..-1, 1, 2, 0]),
        )
        // stream right edges.
        // Right [1, 1]
        .slice_assign(
            s![.., -1, 1, 1],
            thermal_dist.clone().slice(s![.., -1, 1, 1]),
        )
        // e[-1, 1]; v[0, 2]
        .slice_assign(
            s![..-1, -1, 0, 2],
            thermal_dist.clone().slice(s![1.., -2, 0, 2]),
        )
        // e[0, 1]; v[1, 2]
        .slice_assign(
            s![.., -1, 1, 2],
            thermal_dist.clone().slice(s![.., -2, 1, 2]),
        )
        // e[1, 1]; v[2, 2]
        .slice_assign(
            s![1.., -1, 2, 2],
            thermal_dist.clone().slice(s![..-1, -2, 2, 2]),
        )
        // interior
        .slice_assign(s![1..-1, 1..-1], stream_interior_windows(thermal_dist))
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::support::testing::PerformanceBackend;

    #[test]
    #[serial]
    #[rustfmt::skip]
    fn test_stream_interior_windows() {
        type B = PerformanceBackend;
        let device = Default::default();

        let state: Tensor<B, 4> = Tensor::from_data([
            [
                [
                    [0., 1., 2.],
                    [3., 4., 5.],
                    [6., 7., 8.]
                ],
                [
                    [9., 10., 11.],
                    [12., 13., 14.],
                    [15., 16., 17.]
                ],
                [
                    [18., 19., 20.],
                    [21., 22., 23.],
                    [24., 25., 26.]
                ],
            ],
            [
                [
                    [27., 28., 29.],
                    [30., 31., 32.],
                    [33., 34., 35.]
                ],
                [
                    [36., 37., 38.],
                    [39., 40., 41.],
                    [42., 43., 44.]
                ],
                [
                    [45., 46., 47.],
                    [48., 49., 50.],
                    [51., 52., 53.]
                ],
            ],
            [
                [
                    [54., 55., 56.],
                    [57., 58., 59.],
                    [60., 61., 62.]
                ],
                [
                    [63., 64., 65.],
                    [66., 67., 68.],
                    [69., 70., 71.]
                ],
                [
                    [72., 73., 74.],
                    [75., 76., 77.],
                    [78., 79., 80.]
                ],
            ],
        ], &device);

        let result = stream_interior_windows(state.clone());

        assert_eq!(result.dims(), [1, 1, 3, 3]);

        let expected: Tensor<B, 4> = Tensor::from_data([[[
            [72., 64., 56.],
            [48., 40., 32.],
            [24., 16., 8.],
        ]]], &device);

        result.to_data().assert_eq(&expected.to_data(), false);
    }
}
