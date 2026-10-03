//! Windowing operations for Swin Transformer v2

use burn::{
    prelude::{
        Backend,
        Tensor,
    },
    tensor::BasicOps,
};

use crate::contracts::unpack_shape_contract;

/// Window Partition
///
/// Cuts a grid into `window_size x window_size` windows, stacked on the
/// batch axis in row-major window order. [`window_reverse`] undoes it.
///
/// # Arguments
///
/// - `tensor`: a `[batch, h_wins*window_size, w_wins*window_size, channels]`
///   input tensor.
/// - `window_size`: Window size.
///
/// # Returns
///
/// A `[batch*h_wins*w_wins, window_size, window_size, channels]` tensor of
/// windows.
///
/// # Panics
///
/// On shape contract failure: when the height or the width is not a
/// multiple of `window_size`.
#[inline]
#[must_use]
pub fn window_partition<B: Backend, K>(
    tensor: Tensor<B, 4, K>,
    window_size: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    let [b, h_wins, w_wins, c] = unpack_shape_contract!(
        [
            "batch",
            "h_wins" * "window_size",
            "w_wins" * "window_size",
            "channels"
        ],
        &tensor.dims(),
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)]
    );

    tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c])
}

/// Window Reverse
///
/// Reassembles the windows [`window_partition`] cut, where `h_wins` is
/// `height / window_size` and `w_wins` is `width / window_size`.
///
/// # Arguments
///
/// - `windows`: a `[batch*h_wins*w_wins, window_size, window_size, channels]`
///   input tensor.
/// - `window_size`: Window size.
/// - `height`: Height of the original image.
/// - `width`: Width of the original image.
///
/// # Returns
///
/// A `[batch, height, width, channels]` output tensor.
///
/// # Panics
///
/// On shape contract failure.
#[inline]
#[must_use]
pub fn window_reverse<B: Backend, K>(
    windows: Tensor<B, 4, K>,
    window_size: usize,
    height: usize,
    width: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    let h_wins = height / window_size;
    let w_wins = width / window_size;

    let [b, c] = unpack_shape_contract!(
        [
            "batch" * "h_wins" * "w_wins",
            "window_size",
            "window_size",
            "channels"
        ],
        &windows.dims(),
        &["batch", "channels"],
        &[
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("window_size", window_size),
        ],
    );

    windows
        .reshape([b, h_wins, w_wins, window_size, window_size, c])
        .swap_dims(2, 3)
        .reshape([b, height, width, c])
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Tensor,
        tensor::{
            Distribution,
            Tolerance,
        },
    };
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        PerformanceBackend,
        default_device,
    };

    #[test]
    #[serial]
    fn test_window_partition() {
        type B = PerformanceBackend;
        let device = default_device();
        let _memory = DeviceMemoryGuard::<B>::new(&device);

        let b = 3;
        let window_size = 4;
        let channels = 3;

        let h_wins = 2;
        let w_wins = 3;
        let h = h_wins * window_size;
        let w = w_wins * window_size;

        let distribution = Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<B, 4>::random([b, h, w, channels], distribution, &device);

        let windows = window_partition(input.clone(), window_size);

        assert_eq!(
            &windows.dims(),
            &[b * h_wins * w_wins, window_size, window_size, channels]
        );

        let reverse = window_reverse(windows, window_size, h, w);
        assert_eq!(&reverse.dims(), &[b, h, w, channels]);

        reverse
            .to_data()
            .assert_approx_eq(&input.to_data(), Tolerance::<f64>::default());
    }
}
