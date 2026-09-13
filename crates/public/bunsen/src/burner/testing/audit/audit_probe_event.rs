use core::fmt::Debug;
use std::{
    collections::HashMap,
    time::SystemTime,
};

use burn::prelude::TensorData;

/// Common prefix for [`AuditProbeEvent`].
#[derive(Debug, Clone, PartialEq)]
pub struct AuditProbeEventPrefix {
    /// The timestamp of the event.
    ts: SystemTime,
}

impl AuditProbeEventPrefix {
    /// Create a new [`AuditProbeEventPrefix`].
    pub fn new(ts: Option<SystemTime>) -> Self {
        Self {
            ts: ts.unwrap_or_else(SystemTime::now),
        }
    }
}

/// Type-params for [`AuditProbeEvent`].
#[derive(Debug, Clone, PartialEq)]
pub enum AuditProbeEventParams {
    /// `assert_eq` event.
    AssertEq {
        /// Strict dtype comparison.
        strict: bool,
    },
}

/// [`AuditProbeEvent`]-like View trait.
pub trait AuditProbeEventView: Debug {
    /// Return an owned [`AuditProbeEvent`].
    fn to_event(&self) -> AuditProbeEvent {
        AuditProbeEvent {
            prefix: self.prefix().clone(),
            params: self.params().clone(),
            data: self.to_owned_data_map(),
        }
    }

    /// Get a stub-view of this event.
    fn to_stub(&self) -> AuditProbeEventStub<'_>;

    /// Common event prefix.
    fn prefix(&self) -> &AuditProbeEventPrefix;

    /// Type-params for [`AuditProbeEvent`].
    fn params(&self) -> &AuditProbeEventParams;

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
pub struct AuditProbeEvent {
    /// Common event prefix.
    pub prefix: AuditProbeEventPrefix,

    /// Type-params for [`AuditProbeEvent`].
    pub params: AuditProbeEventParams,

    /// Attached event data.
    pub data: HashMap<String, Vec<TensorData>>,
}

impl AuditProbeEventView for AuditProbeEvent {
    fn prefix(&self) -> &AuditProbeEventPrefix {
        &self.prefix
    }

    fn params(&self) -> &AuditProbeEventParams {
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

    fn to_stub(&self) -> AuditProbeEventStub<'_> {
        AuditProbeEventStub {
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

/// [`AuditProbeEvent`]-like stub, doesn't own the [`TensorData`].
#[derive(Debug, Clone)]
pub struct AuditProbeEventStub<'a> {
    /// Common event prefix.
    pub prefix: AuditProbeEventPrefix,

    /// Type-params for [`AuditProbeEvent`].
    pub params: AuditProbeEventParams,

    /// Stub-data for [`AuditProbeEvent`].
    pub data: HashMap<String, Vec<&'a TensorData>>,
}

impl<'a> AuditProbeEventView for AuditProbeEventStub<'a> {
    fn prefix(&self) -> &AuditProbeEventPrefix {
        &self.prefix
    }

    fn params(&self) -> &AuditProbeEventParams {
        &self.params
    }

    fn data_map_iter(&self) -> impl Iterator<Item = (&str, Vec<&TensorData>)> {
        self.data.iter().map(|(k, v)| (k.as_ref(), v.clone()))
    }

    fn to_stub(&self) -> AuditProbeEventStub<'a> {
        self.clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_event_stub() {
        let prefix = AuditProbeEventPrefix::new(None);
        let params = AuditProbeEventParams::AssertEq { strict: true };

        let a = TensorData::from([1, 2, 3]);
        let b = TensorData::from([[2.0, 3.0], [4.0, 5.0]]);

        let event = AuditProbeEvent {
            prefix: prefix.clone(),
            params: params.clone(),
            data: HashMap::from([
                ("x".to_string(), vec![a.clone()]),
                ("y".to_string(), vec![a.clone(), b.clone()]),
            ]),
        };

        let stub = AuditProbeEventStub {
            prefix: prefix.clone(),
            params: params.clone(),
            data: HashMap::from([("x".to_string(), vec![&a]), ("y".to_string(), vec![&a, &b])]),
        };

        assert_eq!(event.data_map_view(), stub.data_map_view(),);
    }
}
