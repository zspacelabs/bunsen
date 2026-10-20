//! Rounding to `bf16` precision, kept in `f32`.
use burn::{
    Tensor,
    prelude::TensorData,
};

/// `x` rounded to the nearest `bf16` value (8 significant bits), ties to
/// even, as an `f32`. Infinities and NaN are left as they are.
pub fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let round = 0x7FFF + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round) & 0xFFFF_0000)
}

/// Each value of `x` rounded to the nearest `bf16` ([`bf16_round`]), kept in
/// `f32`, on the host: `bf16` itself need not exist on the device.
///
/// What a table stored in `bf16` and computed with in `f32` holds, such as
/// a checkpoint's analysis window.
pub fn round_to_bf16<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let device = x.device();
    let shape = x.dims();
    let values: Vec<f32> = x
        .into_data()
        .try_into_vec_as::<f32>()
        .expect("read back f32 values")
        .into_iter()
        .map(bf16_round)
        .collect();
    Tensor::from_data(TensorData::new(values, shape), &device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// Rounding to `bf16` keeps 8 significant bits, ties to even, and leaves
    /// a value that is already `bf16` alone.
    #[test]
    fn test_bf16_round() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(0.0), 0.0);
        // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: to even, 1.
        assert_eq!(bf16_round(1.0 + 2f32.powi(-8)), 1.0);
        // 1 + 3 * 2^-8 is halfway between 1 + 2^-7 and 1 + 2^-6: to even.
        assert_eq!(bf16_round(1.0 + 3.0 * 2f32.powi(-8)), 1.0 + 2f32.powi(-6));
        assert_eq!(bf16_round(1.0 + 2f32.powi(-7)), 1.0 + 2f32.powi(-7));
        let x = 0.123_456_7f32;
        assert_eq!(bf16_round(bf16_round(x)), bf16_round(x));
        assert!((bf16_round(x) - x).abs() <= x * 2f32.powi(-8));
        assert_eq!(bf16_round(f32::INFINITY), f32::INFINITY);
        assert!(bf16_round(f32::NAN).is_nan());
    }

    /// The tensor form rounds each value, keeping the shape and the device.
    #[test]
    fn test_round_to_bf16() {
        let device = cpu_device();
        let values = vec![1.0f32 + 2f32.powi(-8), 0.123_456_7, -3.0, 1e-3];
        let x = Tensor::<2>::from_data(TensorData::new(values.clone(), [2, 2]), &device);
        let y = round_to_bf16(x);
        assert_eq!(y.dims(), [2, 2]);
        let got: Vec<f32> = y.into_data().try_into_vec_as().unwrap();
        let want: Vec<f32> = values.into_iter().map(bf16_round).collect();
        assert_eq!(got, want);
    }
}
