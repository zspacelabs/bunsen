//! Rotary-embedding frequency tables.

use alloc::vec::Vec;

use burn::{
    Tensor,
    tensor::{
        DType,
        Device,
        FloatDType,
        TensorData,
        linalg,
    },
};

/// The dtypes rotary tables are made in, on a device: the dtype their
/// values are rounded to, and the dtype they are kept in.
///
/// - No dtype asked for (or `F32`): exact, in f32.
/// - A narrower dtype (bf16, f16): rounded to it, and kept in it where the
///   device has it, else in f32, which holds each rounded value exactly.
/// - `F64`: in f64 where the device has it, else exact in f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RotaryTableDTypePolicy {
    /// The dtype the tables' values are rounded to, when narrower than f32.
    pub round_to: Option<FloatDType>,
    /// The dtype the tables are kept in.
    pub keep_in: FloatDType,
}

impl RotaryTableDTypePolicy {
    /// The policy for tables wanted in `table_dtype`, on a device that
    /// supports the dtypes `supports` accepts.
    pub fn init(
        table_dtype: Option<FloatDType>,
        supports: impl Fn(DType) -> bool,
    ) -> Self {
        let (round_to, keep_in) = match table_dtype {
            Some(d @ (FloatDType::BF16 | FloatDType::F16)) => (
                Some(d),
                if supports(d.into()) {
                    d
                } else {
                    FloatDType::F32
                },
            ),
            Some(FloatDType::F64) if supports(DType::F64) => (None, FloatDType::F64),
            _ => (None, FloatDType::F32),
        };
        Self { round_to, keep_in }
    }
}

/// `values` as an f32 tensor on `device`, whatever the device's default
/// float: an integer's `float()` is the default, and in bf16 the integers
/// past 256 round.
fn f32_tensor(
    values: Vec<f32>,
    device: &Device,
) -> Tensor<1> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, [n]), (device, DType::F32))
}

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
/// - a `[head_dim / 2]` f32 tensor, holding, for each even dimension index `d`,
///   the inverse frequency `1.0 / base**(d / head_dim)`; f32 whatever the
///   device's default float.
pub fn inverse_frequency_table(
    base: usize,
    head_dim: usize,
    device: &Device,
) -> Tensor<1> {
    let exponents = (0..head_dim)
        .step_by(2)
        .map(|d| -(d as f32) / head_dim as f32)
        .collect();
    f32_tensor(vec![base as f32], device).powf(f32_tensor(exponents, device))
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
/// - `[T, F=D/2]` sequence x inverse frequency table, in f32 whatever the
///   device's default float.
pub fn positional_frequency_table(
    seq_len: usize,
    base: usize,
    head_dim: usize,
    device: &Device,
) -> Tensor<2> {
    frequency_table(seq_len, inverse_frequency_table(base, head_dim, device))
}

/// The rotation angles of the positions `0..seq_len` at `inv_freq`'s
/// frequencies: their outer product.
///
/// # Arguments
/// - `seq_len`: the sequence length.
/// - `inv_freq`: `[F]` inverse frequencies, f32 (an
///   [`inverse_frequency_table`], perhaps scaled).
///
/// # Returns
/// - `[T, F]`, in f32.
pub fn frequency_table(
    seq_len: usize,
    inv_freq: Tensor<1>,
) -> Tensor<2> {
    let t = f32_tensor((0..seq_len).map(|p| p as f32).collect(), &inv_freq.device());
    linalg::outer::<1, 2, _>(t, inv_freq)
}

#[cfg(test)]
mod tests {
    /// A narrower dtype is kept where the device has it, else its rounded
    /// values in f32; f64 only where the device has it.
    #[test]
    fn test_rotary_table_dtype_policy() {
        let all = |_: DType| true;
        let no_half_no_f64 = |d: DType| !matches!(d, DType::BF16 | DType::F16 | DType::F64);
        let policy = |table_dtype, supports: &dyn Fn(DType) -> bool| {
            let p = RotaryTableDTypePolicy::init(table_dtype, supports);
            (p.round_to, p.keep_in)
        };
        assert_eq!(policy(None, &all), (None, FloatDType::F32));
        assert_eq!(
            policy(Some(FloatDType::BF16), &all),
            (Some(FloatDType::BF16), FloatDType::BF16)
        );
        assert_eq!(
            policy(Some(FloatDType::BF16), &no_half_no_f64),
            (Some(FloatDType::BF16), FloatDType::F32)
        );
        assert_eq!(
            policy(Some(FloatDType::F16), &no_half_no_f64),
            (Some(FloatDType::F16), FloatDType::F32)
        );
        assert_eq!(policy(Some(FloatDType::F64), &all), (None, FloatDType::F64));
        assert_eq!(
            policy(Some(FloatDType::F64), &no_half_no_f64),
            (None, FloatDType::F32)
        );
    }

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
