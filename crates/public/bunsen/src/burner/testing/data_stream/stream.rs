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
    errors::{
        BunsenError,
        BunsenResult,
    },
    prelude::TensorElemOpExt,
};

/// Events for [`TensorDataTestStream`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamEventParams {
    /// Event for [`TensorDataTestStreamExt::assert_eq`].
    AssertEq {
        /// Event Label.
        label: String,

        /// Strict dtype comparison.
        strict: bool,
    },
}

/// Events for [`TensorDataTestStream`].
#[derive(Debug, Clone)]
pub struct StreamEventFrame {
    /// Event Parameters.
    pub params: StreamEventParams,

    /// Event Data.
    pub data: Vec<TensorData>,
}

impl StreamEventFrame {
    /// Compare this event (the expected) with actual parameters and data.
    pub fn compare(
        &self,
        actual_params: &StreamEventParams,
        actual_data: &[TensorData],
    ) -> BunsenResult<()> {
        if self.params != *actual_params {
            return Err(BunsenError::InvalidArgument {
                msg: format!(
                    "StreamEvent params {:?} != expected {:?}",
                    self.params, actual_params
                ),
            });
        }
        if self.data.len() != actual_data.len() {
            return Err(BunsenError::InvalidArgument {
                msg: format!(
                    "StreamEvent data count ({:?}) != expected ({:?})",
                    self.data.len(),
                    actual_data.len()
                ),
            });
        }
        if !actual_data.iter().all(|d| d.shape == self.data[0].shape) {
            let actual_shapes = actual_data
                .iter()
                .map(|d| d.shape.clone())
                .collect::<Vec<_>>();
            let expected_shapes = self
                .data
                .iter()
                .map(|d| d.shape.clone())
                .collect::<Vec<_>>();
            return Err(BunsenError::InvalidArgument {
                msg: format!(
                    "event data shapes do not match:\nactual: {:?}\nexpect: {:?}",
                    actual_shapes, expected_shapes
                ),
            });
        }

        match &self.params {
            StreamEventParams::AssertEq { strict, .. } => {
                assert_eq!(actual_data.len(), 1);
                let actual = &actual_data[0];
                let expected = &self.data[0];
                tensor_data_assert_eq(expected, actual, *strict)
            }
        }
    }
}

/// `TensorData` Test Stream.
///
/// Implementors take one of two roles:
/// * a *recorder* appends each event to the stream;
/// * a *verifier* compares each event against the next expected event.
///
/// The role is the implementation of [`handle_event`](`Self::handle_event`);
/// the event constructors live on [`TensorDataTestStreamExt`], and are shared
/// by both roles.
///
/// A type in a role implements [`TensorDataTestStreamRecorder`] or
/// [`TensorDataTestStreamVerifier`], and forwards
/// [`handle_event`](`Self::handle_event`) to it.
pub trait TensorDataTestStream {
    /// Handle a stream event.
    ///
    /// # Arguments
    /// * `params` - the [`StreamEventParams`] to handle.
    /// * `data` - the [`TensorData`] to compare.
    ///
    /// # Panics and/or Err Returns
    /// If the event does not match the expected event under verification.
    fn handle_event(
        &mut self,
        params: &StreamEventParams,
        data: &[TensorData],
    ) -> BunsenResult<()>;
}

impl<T: ?Sized + TensorDataTestStream> TensorDataTestStreamExt for T {}

/// `TensorData` Test Stream Extension
pub trait TensorDataTestStreamExt: TensorDataTestStream {
    /// [`TensorData`] equality; run over
    /// [`handle_event`](`TensorDataTestStream::handle_event`).
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
    ) -> BunsenResult<()> {
        self.handle_event(
            &StreamEventParams::AssertEq {
                label: label.to_string(),
                strict,
            },
            std::slice::from_ref(data),
        )
    }

    /// [`Tensor`] equality; run over
    /// [`assert_eq`](`TensorDataTestStreamExt::assert_eq`).
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

/// Trait for recording events in a [`TensorDataTestStream`].
///
/// No type may implement both this and [`TensorDataTestStreamVerifier`].
pub trait TensorDataTestStreamRecorder {
    /// Record an event in the stream.
    fn write(
        &mut self,
        event: StreamEventFrame,
    ) -> BunsenResult<()>;
}

/// Trait for verifying events in a [`TensorDataTestStream`].
///
/// No type may implement both this and [`TensorDataTestStreamRecorder`].
pub trait TensorDataTestStreamVerifier {
    /// Pop the next expected event from the stream.
    fn read(&mut self) -> BunsenResult<&StreamEventFrame>;
}
