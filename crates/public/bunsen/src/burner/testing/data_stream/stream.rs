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
pub struct EventCommon {
    /// Event Label.
    label: String,

    /// The time the event happened.
    ts: Instant,
}

impl EventCommon {
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
        other: &EventCommon,
    ) -> BunsenResult<()> {
        if self.label != other.label {
            return Err(BunsenError::InvalidArgument {
                msg: format!("Event metadata ({self:?}) != {other:?}"),
            });
        }
        Ok(())
    }
}

/// Event specific params for [`StreamEventFrame`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventParams {
    /// Event for [`TensorDataTestStreamExt::assert_eq`].
    AssertEq {
        /// Strict dtype comparison.
        strict: bool,
    },
}

/// Common access trait for [`StreamEventFrame`].
pub trait StreamEventMeta {
    /// Common event metadata.
    fn common(&self) -> &EventCommon;

    /// Event Parameters.
    fn params(&self) -> &EventParams;

    /// Event Data.
    fn data(&self) -> &[TensorData];

    /// Copy the data into an owned [`StreamEventFrame`].
    fn to_owned(&self) -> StreamEventFrame {
        StreamEventFrame {
            common: self.common().clone(),
            params: self.params().clone(),
            data: self.data().to_vec(),
        }
    }

    /// Compare two stream events.
    fn try_match<T: StreamEventMeta + ?Sized>(
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
    pub common: EventCommon,

    /// Event Parameters.
    pub params: EventParams,

    /// Event Data.
    pub data: Vec<TensorData>,
}

impl StreamEventMeta for StreamEventFrame {
    fn common(&self) -> &EventCommon {
        &self.common
    }

    fn params(&self) -> &EventParams {
        &self.params
    }

    fn data(&self) -> &[TensorData] {
        &self.data
    }
}

/// A [`StreamEventFrame`] stub.
pub struct StreamEventStub<'a> {
    /// Common event metadata.
    pub common: &'a EventCommon,

    /// Event Parameters.
    pub params: &'a EventParams,

    /// Event Data.
    pub data: &'a [TensorData],
}

impl StreamEventMeta for StreamEventStub<'_> {
    fn common(&self) -> &EventCommon {
        self.common
    }

    fn params(&self) -> &EventParams {
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
        event: &impl StreamEventMeta,
    ) -> BunsenResult<()>;
}
