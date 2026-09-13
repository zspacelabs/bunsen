use core::fmt::Debug;
use std::{
    collections::HashMap,
    time::SystemTime,
};

use burn::prelude::TensorData;

/// Common prefix for [`AuditLogEvent`].
#[derive(Debug, Clone, PartialEq)]
pub struct AuditLogEventPrefix {
    /// The timestamp of the event.
    ts: SystemTime,
}

impl AuditLogEventPrefix {
    /// Create a new [`AuditLogEventPrefix`].
    pub fn new(ts: Option<SystemTime>) -> Self {
        Self {
            ts: ts.unwrap_or_else(SystemTime::now),
        }
    }
}

/// Type-params for [`AuditLogEvent`].
#[derive(Debug, Clone, PartialEq)]
pub enum AuditLogEventParams {
    /// `assert_eq` event.
    AssertEq {
        /// Strict dtype comparison.
        strict: bool,
    },
}

/// [`AuditLogEvent`]-like View trait.
pub trait AuditLogEventView: Debug {
    /// Return an owned [`AuditLogEvent`].
    fn to_event(&self) -> AuditLogEvent {
        AuditLogEvent {
            prefix: self.prefix().clone(),
            params: self.params().clone(),
            data: self.to_owned_data_map(),
        }
    }

    /// Get a stub-view of this event.
    fn to_stub(&self) -> AuditLogEventStub<'_>;

    /// Common event prefix.
    fn prefix(&self) -> &AuditLogEventPrefix;

    /// Type-params for [`AuditLogEvent`].
    fn params(&self) -> &AuditLogEventParams;

    /// Iterate over the data in the audit log event.
    fn data_map_iter(&self) -> impl Iterator<Item = (&str, Vec<&TensorData>)>;

    /// Map view over the data in the audit log event.
    fn data_map_view(&self) -> HashMap<&str, Vec<&TensorData>> {
        self.data_map_iter().collect()
    }

    /// Clone the data in the audit log event.
    fn to_owned_data_map(&self) -> HashMap<String, Vec<TensorData>> {
        self.data_map_iter()
            .map(|(k, v)| {
                (
                    k.to_owned(),
                    v.iter().map(|&d| d.clone()).collect::<Vec<TensorData>>(),
                )
            })
            .collect()
    }
}

/// (TODO) Serializable [`AuditProbe`] event.
#[derive(Debug, Clone)]
pub struct AuditLogEvent {
    /// Common event prefix.
    pub prefix: AuditLogEventPrefix,

    /// Type-params for [`AuditLogEvent`].
    pub params: AuditLogEventParams,

    /// Attached event data.
    pub data: HashMap<String, Vec<TensorData>>,
}

impl AuditLogEventView for AuditLogEvent {
    fn prefix(&self) -> &AuditLogEventPrefix {
        &self.prefix
    }

    fn params(&self) -> &AuditLogEventParams {
        &self.params
    }

    fn data_map_iter(&self) -> impl Iterator<Item = (&str, Vec<&TensorData>)> {
        self.data
            .iter()
            .map(|(k, v)| (k.as_ref(), v.iter().collect::<Vec<_>>()))
    }

    fn to_owned_data_map(&self) -> HashMap<String, Vec<TensorData>> {
        self.data.clone()
    }

    fn to_stub(&self) -> AuditLogEventStub<'_> {
        AuditLogEventStub {
            prefix: self.prefix.clone(),
            params: self.params.clone(),
            data: self
                .data
                .iter()
                .map(|(k, v)| (k.clone(), v.iter().collect::<Vec<_>>()))
                .collect(),
        }
    }
}

/// [`AuditLogEvent`]-like stub, doesn't own the [`TensorData`].
#[derive(Debug, Clone)]
pub struct AuditLogEventStub<'a> {
    /// Common event prefix.
    pub prefix: AuditLogEventPrefix,

    /// Type-params for [`AuditLogEvent`].
    pub params: AuditLogEventParams,

    /// Stub-data for [`AuditLogEvent`].
    pub data: HashMap<String, Vec<&'a TensorData>>,
}

impl<'a> AuditLogEventView for AuditLogEventStub<'a> {
    fn prefix(&self) -> &AuditLogEventPrefix {
        &self.prefix
    }

    fn params(&self) -> &AuditLogEventParams {
        &self.params
    }

    fn data_map_iter(&self) -> impl Iterator<Item = (&str, Vec<&TensorData>)> {
        self.data.iter().map(|(k, v)| (k.as_ref(), v.clone()))
    }

    fn to_stub(&self) -> AuditLogEventStub<'a> {
        self.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_event_stub() {
        let prefix = AuditLogEventPrefix::new(None);
        let params = AuditLogEventParams::AssertEq { strict: true };

        let a = TensorData::from([1, 2, 3]);
        let b = TensorData::from([[2.0, 3.0], [4.0, 5.0]]);

        let event = AuditLogEvent {
            prefix: prefix.clone(),
            params: params.clone(),
            data: HashMap::from([
                ("x".to_string(), vec![a.clone()]),
                ("y".to_string(), vec![a.clone(), b.clone()]),
            ]),
        };

        let stub = AuditLogEventStub {
            prefix: prefix.clone(),
            params: params.clone(),
            data: HashMap::from([("x".to_string(), vec![&a]), ("y".to_string(), vec![&a, &b])]),
        };

        assert_eq!(event.data_map_view(), stub.data_map_view(),);
    }
}
