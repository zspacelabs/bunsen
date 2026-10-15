//! Compact fixtures of reference tensors.
use burn::Tensor;
use serde::Deserialize;

/// A reference tensor, recorded compactly: its shape, two sums, and every
/// `stride`-th element of its row-major flattening.
///
/// The fixture form for a reference implementation's output too large to
/// commit whole: a model's activations recorded by a Python reference, say.
/// The samples catch a wrong value anywhere along the flattening; the sums
/// catch a systematic drift the samples straddle.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TensorRecord {
    /// The shape.
    pub shape: Vec<usize>,

    /// The sum of every element.
    pub sum: f64,

    /// The sum of every element's magnitude.
    pub abs_sum: f64,

    /// The sampling stride.
    pub stride: usize,

    /// Elements `0, stride, 2 * stride, ...`.
    pub samples: Vec<f64>,
}

impl TensorRecord {
    /// The element count the shape implies.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The sample count the shape and stride imply.
    pub fn expected_samples(&self) -> usize {
        self.numel().div_ceil(self.stride)
    }

    /// Asserts `actual` is the recorded tensor, within `atol + rtol * |x|` per
    /// sample, and within `rtol * abs_sum + atol` on each sum.
    ///
    /// # Panics
    /// On a different shape, or any value out of tolerance; the message names
    /// `what`, the worst sample, and its index.
    pub fn assert_matches<const D: usize>(
        &self,
        what: &str,
        actual: Tensor<D>,
        rtol: f64,
        atol: f64,
    ) {
        assert_eq!(actual.dims().to_vec(), self.shape, "{what}: shape");
        let values: Vec<f32> = actual
            .into_data()
            .try_into_vec_as::<f32>()
            .expect("read back f32 values");

        let mut worst = (0.0f64, 0usize);
        for (k, expected) in self.samples.iter().enumerate() {
            let got = values[k * self.stride] as f64;
            let excess = (got - expected).abs() - (atol + rtol * expected.abs());
            if excess > worst.0 {
                worst = (excess, k);
            }
        }
        let (excess, k) = worst;
        assert!(
            excess <= 0.0,
            "{what}: sample {k} (element {}) is {} where the reference has {}",
            k * self.stride,
            values[k * self.stride],
            self.samples[k],
        );

        let sum: f64 = values.iter().map(|&v| v as f64).sum();
        let abs_sum: f64 = values.iter().map(|&v| (v as f64).abs()).sum();
        for (name, got, expected) in [("sum", sum, self.sum), ("abs_sum", abs_sum, self.abs_sum)] {
            assert!(
                (got - expected).abs() <= rtol * self.abs_sum + atol,
                "{what}: {name} is {got} where the reference has {expected}",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_matches_and_refuses() {
        let record = TensorRecord {
            shape: vec![2, 3],
            sum: 15.0,
            abs_sum: 15.0,
            stride: 2,
            samples: vec![0.0, 2.0, 4.0],
        };
        assert_eq!(record.expected_samples(), 3);
        let tensor = || Tensor::<2>::from_data([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &cpu_device());
        record.assert_matches("t", tensor(), 1e-6, 1e-6);
        let off =
            std::panic::catch_unwind(|| record.assert_matches("t", tensor() * 1.01, 1e-6, 1e-6));
        assert!(off.is_err());
    }
}
