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
        BoolStore,
        DType,
        DataError,
        Element,
    },
};

use crate::zspace::ravel_dims;

/// Extension trait for [`TensorData`] which provides index view builders.
pub trait TensorDataViewExt {
    /// Returns an [`Index`] view wrapper of the [`TensorData`].
    ///
    /// The view implements [`Deref<Target=TensorData>`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use bunsen::burner::tensor::*;
    /// use burn::prelude::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let view: TensorDataView<f64> = data.try_view().unwrap();
    ///
    /// // Deref
    /// assert_eq!(&view.shape, &shape);
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Returns
    /// `Ok(view)` on success; `Err(DataError::TypeError)` on view [`DType`]
    /// missmatch.
    fn try_view<E: Element>(&self) -> Result<TensorDataView<'_, E>, DataError>;

    /// Returns a [`TensorDataView<'a, E>`] of the [`TensorData`].
    ///
    /// The view implements [`Deref<Target=TensorData>`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use bunsen::burner::tensor::*;
    /// use burn::prelude::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let view: TensorDataView<f64> = data.expect_view();
    ///
    /// // Deref
    /// assert_eq!(&view.shape, &shape);
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Returns
    /// The view.
    ///
    /// # Panics
    /// If the view [`DType`] is not compatible with the [`TensorData`].
    fn expect_view<E: Element>(&self) -> TensorDataView<'_, E> {
        self.try_view().unwrap()
    }

    /// Returns a [`TensorDataViewMut<'a, E>`] of the [`TensorData`].
    ///
    /// The view implements [`DerefMut<Target=TensorData>`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use bunsen::burner::tensor::*;
    /// use burn::prelude::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let mut view: TensorDataViewMut<f64> = data.try_mut_view().unwrap();
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
    ///
    /// # Returns
    /// `Ok(mut view)` on success; `Err(DataError::TypeError)` on view [`DType`]
    /// missmatch.
    fn try_mut_view<E: Element>(&mut self) -> Result<TensorDataViewMut<'_, E>, DataError>;

    /// Returns a [`TensorDataViewMut<'a, E>`] of the [`TensorData`].
    ///
    /// The view implements [`DerefMut<Target=TensorData>`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use bunsen::burner::tensor::*;
    /// use burn::prelude::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let mut view: TensorDataViewMut<f64> = data.expect_mut_view();
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
    ///
    /// # Returns
    /// The mut view.
    ///
    /// # Panics
    /// If the view [`DType`] is not compatible with the [`TensorData`].
    fn expect_mut_view<E: Element>(&mut self) -> TensorDataViewMut<'_, E> {
        self.try_mut_view().unwrap()
    }
}

/// This is [`TensorData`]'s method, but should be public.
/// TODO: expose [`TensorData::matches_target_dtype`] upstream.
fn matches_target_dtype<E: Element>(data: &TensorData) -> bool {
    let target_dtype = E::dtype();
    match data.dtype {
        DType::Bool(BoolStore::U8) => {
            matches!(target_dtype, DType::U8 | DType::Bool(BoolStore::U8))
        }
        DType::Bool(BoolStore::U32) => {
            matches!(target_dtype, DType::U32 | DType::Bool(BoolStore::U32))
        }
        dtype => dtype == target_dtype,
    }
}

impl TensorDataViewExt for TensorData {
    fn try_view<E: Element>(&self) -> Result<TensorDataView<'_, E>, DataError> {
        TensorDataView::try_view(self)
    }

    fn try_mut_view<E: Element>(&mut self) -> Result<TensorDataViewMut<'_, E>, DataError> {
        TensorDataViewMut::try_mut_view(self)
    }
}

/// [Index] view wrapper for a [`TensorData`].
///
/// The view implements [`Deref<Target=TensorData>`].
///
/// # Example
/// ```rust,no_run
/// use bunsen::burner::tensor::*;
/// use burn::prelude::*;
///
/// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
/// let view: TensorDataView<f64> = data.expect_view();
///
/// // Deref
/// assert_eq!(&view.shape, &data.shape);
///
/// assert_eq!(view[&[0, 0]], 1.0);
/// assert_eq!(view[&[0, 1]], 2.0);
/// assert_eq!(view[&[1, 0]], 3.0);
/// assert_eq!(view[&[1, 1]], 4.0);
/// ```
#[derive(Debug)]
pub struct TensorDataView<'a, E: Element> {
    data: &'a TensorData,
    _phantom: std::marker::PhantomData<&'a E>,
}

impl<'a, E: Element> Deref for TensorDataView<'a, E> {
    type Target = TensorData;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, E: Element> TensorDataView<'a, E> {
    /// Returns an [`Index`] view wrapper of the [`TensorData`].
    ///
    /// The view implements [`Deref<Target=TensorData>`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use bunsen::burner::tensor::*;
    /// use burn::prelude::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let view: TensorDataView<f64> = data.try_view().unwrap();
    ///
    /// // Deref.
    /// assert_eq!(&view.shape, &data.shape);
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Returns
    /// `Ok(view)` on success; `Err(DataError::TypeError)` on view [`DType`]
    /// missmatch.
    pub fn try_view(data: &'a TensorData) -> Result<TensorDataView<'a, E>, DataError> {
        if !matches_target_dtype::<E>(data) {
            Err(DataError::TypeMismatch(format!(
                "Cannot view TensorData DType \"{:?}\" as \"{:?}\"",
                data.dtype,
                E::dtype()
            )))
        } else {
            Ok(TensorDataView {
                data,
                _phantom: std::marker::PhantomData,
            })
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

impl<'a, I: AsIndex, E: Element> Index<&[I]> for TensorDataView<'a, E> {
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
/// The view implements [`DerefMut<Target=TensorData>`].
///
/// # Example
/// ```rust,no_run
/// use bunsen::burner::tensor::*;
/// use burn::prelude::*;
///
/// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
/// let shape = data.shape.clone();
/// let mut view: TensorDataViewMut<f64> =
///     TensorDataViewMut::try_mut_view(&mut data).unwrap();
///
/// // The view implements [`Deref`].
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
pub struct TensorDataViewMut<'a, E: Element> {
    data: &'a mut TensorData,
    _phantom: std::marker::PhantomData<&'a E>,
}

impl<'a, E: Element> Deref for TensorDataViewMut<'a, E> {
    type Target = TensorData;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, E: Element> DerefMut for TensorDataViewMut<'a, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl<'a, E: Element> TensorDataViewMut<'a, E> {
    /// Returns an indexed view of the data.
    ///
    /// The view implements [`DerefMut<Target=TensorData>`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use bunsen::burner::tensor::*;
    /// use burn::prelude::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let mut view: TensorDataViewMut<f64> =
    ///     TensorDataViewMut::try_mut_view(&mut data).unwrap();
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
    ///
    /// # Returns
    /// `Ok(mut view)` on success; `Err(DataError::TypeError)` on view [`DType`]
    /// missmatch.
    pub fn try_mut_view(data: &'a mut TensorData) -> Result<TensorDataViewMut<'a, E>, DataError> {
        if !matches_target_dtype::<E>(data) {
            Err(DataError::TypeMismatch(format!(
                "Cannot view TensorData DType \"{:?}\" as \"{:?}\"",
                data.dtype,
                E::dtype()
            )))
        } else {
            Ok(TensorDataViewMut {
                data,
                _phantom: std::marker::PhantomData,
            })
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

impl<'a, I: AsIndex, E: Element> Index<&[I]> for TensorDataViewMut<'a, E> {
    type Output = E;

    fn index(
        &self,
        index: &[I],
    ) -> &Self::Output {
        let o = self.ravel_dims(index);
        &self.data.as_slice::<E>().unwrap()[o]
    }
}

impl<'a, I: AsIndex, E: Element> IndexMut<&[I]> for TensorDataViewMut<'a, E> {
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
    fn test_tensor_data_try_view_dtype_mismatch() {
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = data.try_view::<i32>();
        assert!(matches!(result, Err(DataError::TypeMismatch(_))));
    }

    #[test]
    #[should_panic(expected = "Cannot view TensorData DType")]
    fn test_tensor_data_expect_view_dtype_mismatch() {
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let _view = data.expect_view::<i32>();
    }

    #[test]
    fn test_tensor_data_try_mut_view_dtype_mismatch() {
        let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = data.try_mut_view::<i32>();
        assert!(matches!(result, Err(DataError::TypeMismatch(_))));
    }

    #[test]
    #[should_panic(expected = "Cannot view TensorData DType")]
    fn test_tensor_data_expect_mut_view_dtype_mismatch() {
        let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let _view = data.expect_mut_view::<i32>();
    }

    #[test]
    fn test_tensor_data_index_view() {
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let view = data.expect_view::<f64>();

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

        let mut view = data.expect_mut_view::<f64>();

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
