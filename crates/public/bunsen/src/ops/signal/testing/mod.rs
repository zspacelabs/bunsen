//! Testing utilities for signal operations.

use burn::{
    prelude::{
        Backend,
        TensorData,
    },
    tensor::{
        DType,
        TensorCreationOptions,
        Tolerance,
    },
};
use tracing::debug;

use crate::{
    ops::signal::SamplingWindowBuilder,
    support::testing::assert_close_to_vec,
};

/// Asserts that both of a [`SamplingWindowBuilder`]'s materializations match
/// `expected`.
///
/// [`SamplingWindowBuilder::to_tensor_window`] has a default implementation in
/// terms of [`SamplingWindowBuilder::to_vec_window`], but an implementor may
/// override it; checking both against the same reference catches the pair
/// drifting apart, and reports which side is wrong.
///
/// The window width is taken from `expected.len()`.
///
/// # Panics
///
/// Panics if either materialization differs from `expected`.
pub fn assert_sampling_window_builder_implementation<B: Backend>(
    builder: &impl SamplingWindowBuilder,
    expected: &[f64],
    options: impl Into<TensorCreationOptions<B>>,
) {
    let size = expected.len();
    let options = options.into();

    debug!("checking to_vec_window");
    let vec_win = builder.to_vec_window(size);
    assert_close_to_vec(&vec_win, expected, 0.0001);

    debug!("checking to_tensor_window");
    let ten_win = builder.to_tensor_window(size, options);
    ten_win
        .cast(DType::F64)
        .to_data()
        .assert_approx_eq::<f64>(&TensorData::from(expected), Tolerance::default());
}
