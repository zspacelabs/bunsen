use burn::{
    Tensor,
    prelude::{
        Backend,
        TensorData,
    },
    tensor::{
        BasicOps,
        Element,
    },
};

use crate::{
    burner::testing::data_stream::tensor_data_assert_eq,
    errors::BunsenResult,
    prelude::TensorElemOpExt,
};

/// Events for [`TensorDataTestStream`].
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Event for [`TensorDataTestStream::assert_eq`].
    AssertEq {
        /// Event Label.
        label: String,

        /// Event Data.
        data: TensorData,

        /// Strict dtype comparison.
        strict: bool,
    },
}

/// TensorData Test Stream
pub trait TensorDataTestStream {
    /// [`TensorData`] equality.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    fn assert_eq(
        &mut self,
        label: &str,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()>;
}

impl<T: ?Sized + TensorDataTestStream> TensorDataTestStreamExt for T {}

/// TensorData Test Stream Extension
pub trait TensorDataTestStreamExt: TensorDataTestStream {
    /// [`Tensor`] equality; run over [`assert_eq`](`Self::assert_eq`).
    ///
    /// Data is used as `tensor.to_data_as::<E>()`.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    fn assert_tensor_eq_as<B, const R: usize, K, E>(
        &mut self,
        label: &str,
        tensor: &Tensor<B, R, K>,
        strict: bool,
    ) -> BunsenResult<()>
    where
        E: Element,
        B: Backend,
        K: BasicOps<B>,
    {
        let data = tensor.to_data_as::<E>();
        self.assert_eq(label, &data, strict)
    }
}

/// Trait for verifying events in a [`TensorDataTestStream`].
pub trait TensorDataTestStreamVerifier: TensorDataTestStream {
    /// Pop the next event from the stream.
    fn pop(&mut self) -> BunsenResult<&StreamEvent>;
}

impl<T: TensorDataTestStreamVerifier> TensorDataTestStream for T {
    fn assert_eq(
        &mut self,
        label: &str,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        let label = label.to_string();
        match self.pop()? {
            StreamEvent::AssertEq {
                label: e_label,
                data: e_data,
                strict: e_strict,
            } => {
                assert_eq!(&label, e_label);
                assert_eq!(strict, *e_strict);
                tensor_data_assert_eq(&e_data, data, strict)
            } // _ => Err(BunsenError::Invalid("Expected assert_eq event".to_string())),
        }
    }
}
