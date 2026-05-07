//! # Utility crate for [`ShapeArgument`] for passing shapes in a type-safe manner.

use alloc::vec::Vec;

/// Adaptor to view sources as a &[usize].
pub struct ShapeView<'a> {
    slice: Option<&'a [usize]>,
    vec: Option<Vec<usize>>,
}

impl<'a> ShapeView<'a> {
    /// Build an adaptor from a slice reference.
    pub fn from_slice(slice: &'a [usize]) -> Self {
        Self {
            slice: Some(slice),
            vec: None,
        }
    }

    /// Build an adaptor from a vector.
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

#[cfg(test)]
mod tests {
    use alloc::{
        vec,
        vec::Vec,
    };

    use super::*;

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
}
