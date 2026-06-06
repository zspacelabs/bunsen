//! # `TensorData` View Wrappers

use std::ops::{
    Deref,
    DerefMut,
    Index,
    IndexMut,
};

use burn::{
    prelude::TensorData,
    tensor::{
        AsIndex,
        Element,
    },
};

use crate::zspace::ravel_dims;

/// [Index] view wrapper for a [`TensorData`].
///
/// # Example
/// ```rust,no_run
/// use bunsen::burner::tensor::*;
/// use burn::prelude::*;
///
/// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
/// let shape = data.shape.clone();
/// let view: TensorDataIndexView<f64> = TensorDataIndexView::view(&data);
///
/// // Deref
/// assert_eq!(&view.shape, &shape);
///
/// assert_eq!(view[&[0, 0]], 1.0);
/// assert_eq!(view[&[0, 1]], 2.0);
/// assert_eq!(view[&[1, 0]], 3.0);
/// assert_eq!(view[&[1, 1]], 4.0);
/// ```
#[derive(Debug)]
pub struct TensorDataIndexView<'a, E: Element> {
    data: &'a TensorData,
    _phantom: std::marker::PhantomData<&'a E>,
}

impl<'a, E: Element> Deref for TensorDataIndexView<'a, E> {
    type Target = TensorData;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, E: Element> TensorDataIndexView<'a, E> {
    /// Returns an indexed view of the data.
    pub fn view(data: &'a TensorData) -> TensorDataIndexView<'a, E> {
        TensorDataIndexView {
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Ravels the dims via [`ravel_dims`] and the view's shape.
    pub fn ravel_dims<I: AsIndex>(
        &self,
        dims: &[I],
    ) -> usize {
        ravel_dims(&self.data.shape, dims)
    }
}

impl<'a, I: AsIndex, E: Element> Index<&[I]> for TensorDataIndexView<'a, E> {
    type Output = E;

    fn index(
        &self,
        index: &[I],
    ) -> &Self::Output {
        let o = self.ravel_dims(index);
        &self.data.as_slice::<E>().unwrap()[o]
    }
}

/// Mutable [`IndexMut`] view wrapper for a [`TensorData`].
///
/// # Example
/// ```rust,no_run
/// use bunsen::burner::tensor::*;
/// use burn::prelude::*;
///
/// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
/// let shape = data.shape.clone();
/// let mut view: TensorDataIndexMutView<f64> =
///     TensorDataIndexMutView::view(&mut data);
///
/// // Deref
/// assert_eq!(&view.shape, &shape);
///
/// assert_eq!(view[&[0, 0]], 1.0);
/// assert_eq!(view[&[0, 1]], 2.0);
/// assert_eq!(view[&[1, 0]], 3.0);
/// assert_eq!(view[&[1, 1]], 4.0);
///
/// view[&[0, 0]] = 10.0;
/// assert_eq!(view[&[0, 0]], 10.0);
/// ```
#[derive(Debug)]
pub struct TensorDataIndexMutView<'a, E: Element> {
    data: &'a mut TensorData,
    _phantom: std::marker::PhantomData<&'a E>,
}

impl<'a, E: Element> Deref for TensorDataIndexMutView<'a, E> {
    type Target = TensorData;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, E: Element> DerefMut for TensorDataIndexMutView<'a, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl<'a, E: Element> TensorDataIndexMutView<'a, E> {
    /// Returns an indexed view of the data.
    pub fn view(data: &'a mut TensorData) -> TensorDataIndexMutView<'a, E> {
        TensorDataIndexMutView {
            data,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Ravels the dims via [`ravel_dims`] and the view's shape.
    pub fn ravel_dims<I: AsIndex>(
        &self,
        dims: &[I],
    ) -> usize {
        ravel_dims(&self.data.shape, dims)
    }
}

impl<'a, I: AsIndex, E: Element> Index<&[I]> for TensorDataIndexMutView<'a, E> {
    type Output = E;

    fn index(
        &self,
        index: &[I],
    ) -> &Self::Output {
        let o = self.ravel_dims(index);
        &self.data.as_slice::<E>().unwrap()[o]
    }
}

impl<'a, I: AsIndex, E: Element> IndexMut<&[I]> for TensorDataIndexMutView<'a, E> {
    fn index_mut(
        &mut self,
        index: &[I],
    ) -> &mut Self::Output {
        let o = self.ravel_dims(index);
        &mut self.data.as_mut_slice::<E>().unwrap()[o]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_index_view() {
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let view: TensorDataIndexView<f64> = TensorDataIndexView::view(&data);

        // Deref
        assert_eq!(&data.shape, &view.shape);

        assert_eq!(view[&[0, 0]], 1.0);
        assert_eq!(view[&[0, 1]], 2.0);
        assert_eq!(view[&[1, 0]], 3.0);
        assert_eq!(view[&[1, 1]], 4.0);
    }

    #[test]
    fn test_tensor_data_index_mut_view() {
        let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let shape = data.shape.clone();

        let mut view: TensorDataIndexMutView<f64> = TensorDataIndexMutView::view(&mut data);

        // Deref
        assert_eq!(&view.shape, &shape);

        assert_eq!(view[&[0, 0]], 1.0);
        assert_eq!(view[&[0, 1]], 2.0);
        assert_eq!(view[&[1, 0]], 3.0);
        assert_eq!(view[&[1, 1]], 4.0);

        view[&[0, 0]] = 10.0;
        assert_eq!(view[&[0, 0]], 10.0);
    }
}
