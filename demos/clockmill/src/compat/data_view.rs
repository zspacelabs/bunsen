//! # `TensorData` View Wrappers

use std::ops::Index;

use burn::{
    prelude::TensorData,
    tensor::{
        AsIndex,
        Element,
    },
};

use crate::compat::shape::ravel_dims;

/// Ravel Index View for a `TensorData`.
#[derive(Debug)]
pub struct TensorDataIndexView<'a, E: Element> {
    data: &'a TensorData,
    _phantom: std::marker::PhantomData<&'a E>,
}

impl<'a, E: Element> TensorDataIndexView<'a, E> {
    /// Get an indexed view of the data.
    pub fn view(data: &'a TensorData) -> TensorDataIndexView<'a, E> {
        TensorDataIndexView {
            data,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, I: AsIndex, E: Element> Index<&[I]> for TensorDataIndexView<'a, E> {
    type Output = E;

    fn index(
        &self,
        index: &[I],
    ) -> &Self::Output {
        &self.data.as_slice::<E>().unwrap()[ravel_dims(&self.data.shape, index)]
    }
}
