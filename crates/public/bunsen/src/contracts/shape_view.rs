//! The shape argument of contract checks.

use alloc::vec::Vec;

use burn::{
    Tensor,
    prelude::{
        Backend,
        Shape,
    },
    tensor::BasicOps,
};

/// A shape as a `[usize]` slice: the `shape` argument of every
/// [`ShapeContract`](crate::contracts::ShapeContract) check.
///
/// `ShapeView` is a struct, not a trait. The checks take
/// `S: Into<ShapeView>`, so a caller passes a shape in whatever form it has;
/// the `From` impls below list every accepted form. A view either borrows the
/// caller's `usize`s or owns a converted copy:
///
/// - borrowed, no allocation: `&[usize]`, `&[usize; D]`, `&Vec<usize>`, and
///   `&Shape`;
/// - moved, no allocation: `Vec<usize>`;
/// - converted into a new `Vec<usize>` on every call: `&[u32]`, `&[u32; D]`,
///   `&[i32]`, `&[i32; D]`, `Vec<u32>`, `Vec<i32>`, an owned `Shape`, and
///   `&Tensor`.
///
/// For a tensor, pass `&x.dims()`, a borrowed `[usize; D]`, rather than `&x`,
/// which goes through [`Shape`] and copies it into a `Vec`.
///
/// `u32` and `i32` sizes are cast with `as usize`. A negative `i32` wraps to a
/// huge size instead of being rejected, and the check then runs against that
/// size.
pub struct ShapeView<'a> {
    slice: Option<&'a [usize]>,
    vec: Option<Vec<usize>>,
}

impl<'a> ShapeView<'a> {
    /// Builds an adaptor from a slice reference.
    pub fn from_slice(slice: &'a [usize]) -> Self {
        Self {
            slice: Some(slice),
            vec: None,
        }
    }

    /// Builds an adaptor from a vector.
    pub fn from_vec(shape: Vec<usize>) -> Self {
        Self {
            slice: None,
            vec: Some(shape),
        }
    }
}

impl<'a> AsRef<[usize]> for ShapeView<'a> {
    fn as_ref(&self) -> &[usize] {
        match self.slice {
            Some(slice) => slice,
            None => self.vec.as_ref().unwrap(),
        }
    }
}

impl<'a> From<&'a [usize]> for ShapeView<'a> {
    fn from(slice: &'a [usize]) -> Self {
        Self::from_slice(slice)
    }
}

impl<'a, const D: usize> From<&'a [usize; D]> for ShapeView<'a> {
    fn from(slice: &'a [usize; D]) -> Self {
        Self::from_slice(slice)
    }
}

impl<'a, const D: usize> From<&'a [u32; D]> for ShapeView<'a> {
    fn from(slice: &'a [u32; D]) -> Self {
        slice.as_slice().into()
    }
}

impl<'a, const D: usize> From<&'a [i32; D]> for ShapeView<'a> {
    fn from(slice: &'a [i32; D]) -> Self {
        slice.as_slice().into()
    }
}

impl<'a> From<&'a [u32]> for ShapeView<'a> {
    fn from(slice: &'a [u32]) -> Self {
        Self::from_vec(slice.iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

impl<'a> From<&'a [i32]> for ShapeView<'a> {
    fn from(slice: &'a [i32]) -> Self {
        Self::from_vec(slice.iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

impl<'a> From<&'a Vec<usize>> for ShapeView<'a> {
    fn from(vec: &'a Vec<usize>) -> Self {
        Self::from_slice(vec.as_slice())
    }
}

impl<'a> From<Vec<usize>> for ShapeView<'a> {
    fn from(vec: Vec<usize>) -> Self {
        Self::from_vec(vec)
    }
}

impl<'a> From<Vec<u32>> for ShapeView<'a> {
    fn from(vec: Vec<u32>) -> Self {
        Self::from_vec(vec.iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

impl<'a> From<Vec<i32>> for ShapeView<'a> {
    fn from(vec: Vec<i32>) -> Self {
        Self::from_vec(vec.iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

impl<'a> From<&'a Shape> for ShapeView<'a> {
    fn from(shape: &'a Shape) -> Self {
        shape.as_slice().into()
    }
}

impl<'a> From<Shape> for ShapeView<'a> {
    fn from(shape: Shape) -> Self {
        shape.to_vec().into()
    }
}

impl<'a, B, const R: usize, K> From<&'a Tensor<B, R, K>> for ShapeView<'a>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn from(tensor: &'a Tensor<B, R, K>) -> Self {
        tensor.shape().into()
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        vec,
        vec::Vec,
    };

    use super::*;
    use crate::support::testing::CpuBackend;

    #[test]
    fn test_shape_views() {
        let expected = vec![2, 3, 4];

        {
            let arr: [usize; 3] = [2, 3, 4];
            let sv: ShapeView = (&arr).into();
            assert_eq!(sv.as_ref(), &expected);

            let arr_ref: &[usize] = &arr;
            let sv: ShapeView = arr_ref.into();
            assert_eq!(sv.as_ref(), &expected);
        }

        {
            let arr: [u32; 3] = [2, 3, 4];
            let sv: ShapeView = (&arr).into();
            assert_eq!(sv.as_ref(), &expected);

            let arr_ref: &[u32] = &arr;
            let sv: ShapeView = arr_ref.into();
            assert_eq!(sv.as_ref(), &expected);
        }

        {
            let arr: [i32; 3] = [2, 3, 4];
            let sv: ShapeView = (&arr).into();
            assert_eq!(sv.as_ref(), &expected);

            let arr_ref: &[i32] = &arr;
            let sv: ShapeView = arr_ref.into();
            assert_eq!(sv.as_ref(), &expected);
        }

        {
            let vec: Vec<usize> = vec![2, 3, 4];
            let sv: ShapeView = vec.clone().into();
            assert_eq!(sv.as_ref(), &expected);

            let arr_ref: &[usize] = &vec;
            let sv: ShapeView = arr_ref.into();
            assert_eq!(sv.as_ref(), &expected);
        }

        {
            let vec: Vec<u32> = vec![2, 3, 4];
            let sv: ShapeView = vec.clone().into();
            assert_eq!(sv.as_ref(), &expected);

            let arr_ref: &[u32] = &vec;
            let sv: ShapeView = arr_ref.into();
            assert_eq!(sv.as_ref(), &expected);
        }

        {
            let vec: Vec<i32> = vec![2, 3, 4];
            let sv: ShapeView = vec.clone().into();
            assert_eq!(sv.as_ref(), &expected);

            let arr_ref: &[i32] = &vec;
            let sv: ShapeView = arr_ref.into();
            assert_eq!(sv.as_ref(), &expected);
        }
    }

    #[test]
    #[allow(unused)]
    fn test_burn_shape_views() {
        type B = CpuBackend;
        let expected = vec![2, 3, 4];

        let shape = Shape::from([2, 3, 4]);
        let sv: ShapeView = shape.clone().into();
        assert_eq!(sv.as_ref(), &expected);

        let shape_ref: &Shape = &shape;
        let sv: ShapeView = shape_ref.into();
        assert_eq!(shape_ref.as_ref(), &expected);

        let tensor: Tensor<B, 2> = Tensor::zeros([2, 2], &Default::default());
        let tensor_ref = &tensor;
        let sv: ShapeView = tensor_ref.into();
        assert_eq!(sv.as_ref(), &[2, 2]);
    }
}
