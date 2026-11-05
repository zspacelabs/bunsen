//! Operations over optional tensors.

use burn::Tensor;

/// `a` and `b` concatenated along `dim`, either one absent: both, or
/// whichever is present, or `None` when neither is.
///
/// What a buffer that may not have started yet appends with.
pub fn cat_optional<const D: usize>(
    a: Option<Tensor<D>>,
    b: Option<Tensor<D>>,
    dim: usize,
) -> Option<Tensor<D>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(Tensor::cat(vec![a, b], dim)),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// Both present concatenate; one present is itself; neither is `None`.
    #[test]
    fn test_cat_optional() {
        let device = cpu_device();
        let a = Tensor::<2>::from_floats([[1.0, 2.0]], &device);
        let b = Tensor::<2>::from_floats([[3.0, 4.0]], &device);
        let both = cat_optional(Some(a.clone()), Some(b.clone()), 0).unwrap();
        assert_eq!(both.dims(), [2, 2]);
        let one = cat_optional(None, Some(b.clone()), 0).unwrap();
        one.into_data().assert_eq(&b.into_data(), true);
        assert!(cat_optional::<2>(None, None, 0).is_none());
    }
}
