use std::time::Instant;

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

/// Common metadata for all events.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventMeta {
    /// Event Label.
    label: String,

    /// The time the event happened.
    ts: Instant,
}

impl EventMeta {
    /// Create a new event meta.
    pub fn new(
        label: String,
        ts: Option<Instant>,
    ) -> Self {
        Self {
            label,
            ts: ts.unwrap_or_else(Instant::now),
        }
    }

    /// Compare this event (the expected) with actual parameters and data.
    pub fn compare(
        &self,
        other: &EventMeta,
    ) -> BunsenResult<()> {
        if self.label != other.label {
            return Err(BunsenError::InvalidArgument {
                msg: format!("Event metadata ({self:?}) != {other:?}"),
            });
        }
        Ok(())
    }
}

/// Events for [`TensorDataTestStream`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamEventParams {
    /// Event for [`TensorDataTestStreamExt::assert_eq`].
    AssertEq {
        /// Strict dtype comparison.
        strict: bool,
    },
}

/// Common access trait for [`StreamEventFrame`].
pub trait StreamEventFrameMeta {
    /// Common event metadata.
    fn meta(&self) -> &EventMeta;

    /// Event Parameters.
    fn params(&self) -> &StreamEventParams;

    /// Event Data.
    fn data(&self) -> &[TensorData];

    /// Copy the data into an owned [`StreamEventFrame`].
    fn to_owned(&self) -> StreamEventFrame {
        StreamEventFrame {
            meta: self.meta().clone(),
            params: self.params().clone(),
            data: self.data().to_vec(),
        }
    }

    /// Compare this event (the expected) with actual parameters and data.
    fn try_match(
        &self,
        actual: &impl StreamEventFrameMeta,
    ) -> BunsenResult<()> {
        self.meta().compare(actual.meta())?;

        if self.params() != actual.params() {
            return Err(BunsenError::InvalidArgument {
                msg: format!(
                    "StreamEvent params {:?} != expected {:?}",
                    self.params(),
                    actual.params()
                ),
            });
        }
        if self.data().len() != actual.data().len() {
            return Err(BunsenError::InvalidArgument {
                msg: format!(
                    "StreamEvent data count ({:?}) != expected ({:?})",
                    self.data().len(),
                    actual.data().len()
                ),
            });
        }

        let expected_data: &[TensorData] = self.data().as_ref();
        if !actual
            .data()
            .iter()
            .zip(expected_data.iter())
            .all(|(a, e)| a.shape == e.shape)
        {
            return Err(BunsenError::InvalidArgument {
                msg: format!(
                    "event data shapes do not match:\nactual: {:?}\nexpect: {:?}",
                    actual
                        .data()
                        .iter()
                        .map(|d| d.shape.clone())
                        .collect::<Vec<_>>(),
                    expected_data
                        .iter()
                        .map(|d| d.shape.clone())
                        .collect::<Vec<_>>()
                ),
            });
        }

        match self.params() {
            StreamEventParams::AssertEq { strict, .. } => {
                assert_eq!(actual.data().len(), 1);
                let actual = &actual.data()[0];
                let expected = &expected_data[0];
                tensor_data_assert_eq(expected, actual, *strict)
            }
        }
    }
}

/// Events for [`TensorDataTestStream`].
#[derive(Debug, Clone)]
pub struct StreamEventFrame {
    /// Common event metadata.
    pub meta: EventMeta,

    /// Event Parameters.
    pub params: StreamEventParams,

    /// Event Data.
    pub data: Vec<TensorData>,
}

impl StreamEventFrameMeta for StreamEventFrame {
    fn meta(&self) -> &EventMeta {
        &self.meta
    }

    fn params(&self) -> &StreamEventParams {
        &self.params
    }

    fn data(&self) -> &[TensorData] {
        &self.data
    }
}

/// A [`StreamEventFrame`] stub.
pub struct StreamEventFrameStub<'a> {
    /// Common event metadata.
    pub meta: &'a EventMeta,

    /// Event Parameters.
    pub params: &'a StreamEventParams,

    /// Event Data.
    pub data: &'a [TensorData],
}

impl StreamEventFrameMeta for StreamEventFrameStub<'_> {
    fn meta(&self) -> &EventMeta {
        self.meta
    }

    fn params(&self) -> &StreamEventParams {
        self.params
    }

    fn data(&self) -> &[TensorData] {
        self.data
    }
}

/// [`StreamEventFrame`] handling trait.
pub trait OnStreamEvent {
    /// Handle a stream event.
    fn on_stream_event(
        &mut self,
        event: impl StreamEventFrameMeta,
    ) -> BunsenResult<()>;
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
/// A type in a role implements [`StreamEventSink`] or
/// [`StreamEventSource`], and forwards
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
        meta: &EventMeta,
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
    #[track_caller]
    fn assert_eq(
        &mut self,
        label: &str,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        self.handle_event(
            &EventMeta::new(label.to_string(), None),
            &StreamEventParams::AssertEq { strict },
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
    #[track_caller]
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
/// No type may implement both this and [`StreamEventSource`].
pub trait StreamEventSink {
    /// Record an event in the stream.
    fn write(
        &mut self,
        event: StreamEventFrame,
    ) -> BunsenResult<()>;
}

/// Trait for verifying events in a [`TensorDataTestStream`].
///
/// No type may implement both this and [`StreamEventSink`].
pub trait StreamEventSource {
    /// Pop the next expected event from the stream.
    fn read(&mut self) -> BunsenResult<&StreamEventFrame>;
}
