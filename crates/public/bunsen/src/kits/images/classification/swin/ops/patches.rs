//! Patch collation, and its inverse.

use burn::{
    prelude::Tensor,
    tensor::kind::Basic,
};

use super::{
    window_partition,
    window_reverse,
};
use crate::contracts::unpack_shape_contract;

/// Collates patches from a tensor.
///
/// This splits the input into 4 interleaved patches, each `[height/2,
/// width/2]`; and then concatenates them along the last dimension.
///
/// # Arguments
///
/// * `x` - `[batch, height * width, channels]` input tensor.
/// * `h` - Height of the input tensor.
/// * `w` - Width of the input tensor.
///
/// # Returns
///
/// * A `[batch, height/2 * width/2, 4 * channels]` tensor.
///
/// # Panics
///
/// On shape contract failure.
pub fn collate_patches<K>(
    x: Tensor<3, K>,
    h: usize,
    w: usize,
) -> Tensor<3, K>
where
    K: Basic,
{
    let [b, h, w, c] = unpack_shape_contract!(
        ["batch", "flat" = "height" * "width", "channels"],
        &x.dims(),
        &["batch", "height", "width", "channels"],
        &[("height", h), ("width", w)]
    );

    let h2 = h / 2;
    let w2 = w / 2;
    let h2w2 = h2 * w2;

    // TODO(crutcher): re-using `window_partition` requires us to
    // double-reshape. Is it worth writing a trait to permit windowing on 3D
    // and 4D tensors? In SWIN Source; window_partition is *always*
    // immediately resized.
    let x = x.reshape([b, h, w, c]);
    let x = window_partition(x, 2);

    x.reshape([b, h2w2, 4 * c])
}

/// Decollates patches from a tensor.
///
/// This splits the input into 4 patches, each `[height/2, width/2]`;
/// and interleaves them along the height and width dimensions.
///
/// The inverse operation of `collate_patches`.
///
/// # Arguments
///
/// * `x` - `[batch, height/2 * width/2, 4 * channels]` input tensor.
/// * `height` - Height of the input tensor.
/// * `width` - Width of the input tensor.
///
/// # Returns
///
/// * A `[batch, height * width, channels]` tensor.
///
/// # Panics
///
/// On shape contract failure.
pub fn decollate_patches<K>(
    x: Tensor<3, K>,
    height: usize,
    width: usize,
) -> Tensor<3, K>
where
    K: Basic,
{
    let h2 = height / 2;
    let w2 = width / 2;

    let [b, c] = unpack_shape_contract!(
        ["batch", "half_height" * "half_width", "channels"],
        &x.dims(),
        &["batch", "channels"],
        &[("half_height", h2), ("half_width", w2)]
    );

    let c = c / 4;

    let x = x.reshape([b * h2 * w2, 2, 2, c]);
    let x = window_reverse(x, 2, height, width);

    x.reshape([b, height * width, c])
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        performance_device,
    };

    #[test]
    #[serial]
    fn test_collate_patches() {
        let b = 2;
        let h = 4;
        let w = 6;
        let c = 5;

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let distribution = Distribution::Normal(0., 1.);
        let x = Tensor::<3>::random([b, h * w, c], distribution, &device);

        let y = collate_patches(x.clone(), h, w);
        assert_eq!(&y.dims(), &[b, (h / 2) * (w / 2), 4 * c]);

        decollate_patches(y.clone(), h, w)
            .into_data()
            .assert_eq(&x.into_data(), true);
    }
}
