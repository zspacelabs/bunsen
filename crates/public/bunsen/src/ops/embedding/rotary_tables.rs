//! Rotary-embedding frequency tables.

use burn::{
    Tensor,
    tensor::{
        Device,
        linalg,
    },
};

/// Computes the rotary embedding inverse frequency table.
///
/// One frequency per pair of head dimensions, falling geometrically from 1 to
/// about `1 / base`.
///
/// # Arguments
/// - `base`: the base.
/// - `head_dim`: the number of head dimensions.
/// - `device`: the target device.
///
/// # Returns
/// - a `[head_dim / 2]` tensor, holding, for each even dimension index `d`, the
///   inverse frequency `1.0 / base**(d / head_dim)`.
pub fn inverse_frequency_table(
    base: usize,
    head_dim: usize,
    device: &Device,
) -> Tensor<1> {
    Tensor::from_data([base as f32], device).powf(
        -Tensor::arange_step(0..head_dim as i64, 2, device)
            .float()
            .div_scalar(head_dim as f32),
    )
}

/// Computes the positionally shifted frequency table.
///
/// The outer product of the positions `0..seq_len` and
/// [`inverse_frequency_table`]: the rotation angle for each position and
/// dimension pair. [`RotaryEmbedding`] caches its `cos` and `sin`.
///
/// [`RotaryEmbedding`]: crate::blocks::transformers::embedding::RotaryEmbedding
///
/// # Arguments
/// - `seq_len`: the sequence length.
/// - `base`: the base.
/// - `head_dim`: the number of head dimensions.
/// - `device`: the target device.
///
/// # Returns
/// - `[T, F=D/2]` sequence x inverse frequency table.
pub fn positional_frequency_table(
    seq_len: usize,
    base: usize,
    head_dim: usize,
    device: &Device,
) -> Tensor<2> {
    let inv_freq = inverse_frequency_table(base, head_dim, device);

    let t: Tensor<1> = Tensor::arange(0..seq_len as i64, device).float();

    linalg::outer::<1, 2, _>(t, inv_freq)
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        performance_device,
    };

    #[test]
    #[serial]
    fn test_inverse_frequency_table() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let base = 10000;
        let head_dim = 4;

        let base_f = base as f32;
        let head_dim_f = head_dim as f32;

        inverse_frequency_table(base, head_dim, &device)
            .to_data()
            .assert_approx_eq(
                &Tensor::<1>::from_data(
                    [
                        1.0 / base_f.powf(0.0 / head_dim_f),
                        1.0 / base_f.powf(2.0 / head_dim_f),
                    ],
                    &device,
                )
                .to_data(),
                Tolerance::<f32>::default(),
            );
    }

    #[test]
    #[serial]
    fn test_frequency_matrix() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let base = 10000;
        let head_dim = 4;

        let base_f = base as f32;
        let head_dim_f = head_dim as f32;

        positional_frequency_table(3, base, head_dim, &device)
            .to_data()
            .assert_approx_eq(
                &Tensor::<2>::from_data(
                    [
                        [0.0, 0.0],
                        [
                            1.0 / base_f.powf(0.0 / head_dim_f),
                            1.0 / base_f.powf(2.0 / head_dim_f),
                        ],
                        [
                            2.0 / base_f.powf(0.0 / head_dim_f),
                            2.0 / base_f.powf(2.0 / head_dim_f),
                        ],
                    ],
                    &device,
                )
                .to_data(),
                Tolerance::<f32>::default(),
            );
    }
}
