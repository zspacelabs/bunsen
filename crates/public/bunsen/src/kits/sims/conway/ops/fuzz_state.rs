//! Boolean Board State Fuzzing Operations

use burn::{
    Tensor,
    prelude::Bool,
    tensor::Distribution,
};

/// Fuzzes the state.
///
/// Flips bits with probability `density`.
///
/// # Arguments
///
/// - `state`: the `[...]` input state.
/// - `density`: the probability of flipping a given bit.
///
/// # Returns
/// - a fuzzed `[...]` state.
pub fn fuzz_state<const R: usize>(
    state: Tensor<R, Bool>,
    density: f64,
) -> Tensor<R, Bool> {
    if density == 0.0 {
        return state;
    }

    let noise = Tensor::<R>::random(
        state.shape(),
        Distribution::Bernoulli(density),
        &state.device(),
    )
    .bool();

    state.bool_xor(noise)
}

/// Fuzzes the state.
///
/// Flips bits with probability `density`.
///
/// # Arguments
///
/// - `state`: the `[H, W]` input state.
/// - `density`: the probability of flipping a given bit.
///
/// # Returns
/// - the fuzzed `[H, W]` state.
pub fn fuzz_state_2d(
    state: Tensor<2, Bool>,
    density: f64,
) -> Tensor<2, Bool> {
    fuzz_state(state, density)
}

/// Fuzzes the state.
///
/// Flips bits with probability `density`.
///
/// # Arguments
///
/// - `state`: the `[H, W, Z]` input state.
/// - `density`: the probability of flipping a given bit.
///
/// # Returns
/// - the fuzzed `[H, W, Z]` state.
pub fn fuzz_state_3d(
    state: Tensor<3, Bool>,
    density: f64,
) -> Tensor<3, Bool> {
    fuzz_state(state, density)
}
