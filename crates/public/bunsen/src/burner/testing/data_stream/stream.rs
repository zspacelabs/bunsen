use std::time::Instant;

use burn::prelude::TensorData;

use crate::{
    burner::testing::data_stream::try_match_stream_events,
    errors::{
        BunsenError,
        BunsenResult,
    },
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

    /// Compare two stream events.
    fn try_match<T: StreamEventFrameMeta + ?Sized>(
        &self,
        expected: &T,
    ) -> BunsenResult<()> {
        try_match_stream_events(self, expected)
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
        event: &impl StreamEventFrameMeta,
    ) -> BunsenResult<()>;
}
