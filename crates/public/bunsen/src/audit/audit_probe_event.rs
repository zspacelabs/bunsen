use alloc::vec::Vec;
use core::fmt::Debug;
use std::{
    collections::HashMap,
    time::SystemTime,
};

use burn::prelude::{
    Shape,
    TensorData,
};

use crate::{
    audit::audit_probe::AuditProbeEventParams,
    errors::{
        BunsenError,
        BunsenResult,
    },
    rust_ext::reflection::LocationDesc,
};

/// Receives the events an [`AuditProbe`] emits.
///
/// A probe calls [`on_event`](Self::on_event) on each of its handlers, in
/// order, for every checkpoint; an `Err` stops the event and is returned from
/// the checkpoint. Recorders ([`AuditStreamRecorder`],
/// [`AuditProbeVecRecorder`]) keep [`to_event`](AuditProbeEventView::to_event)
/// copies; verifiers ([`AuditStreamVerifier`], [`AuditProbeVecVerifier`])
/// compare each stub with an expected event using [`try_match_events`].
///
/// [`AuditProbe`]: crate::audit::AuditProbe
/// [`AuditStreamRecorder`]: crate::audit::AuditStreamRecorder
/// [`AuditProbeVecRecorder`]: crate::audit::AuditProbeVecRecorder
/// [`AuditStreamVerifier`]: crate::audit::AuditStreamVerifier
/// [`AuditProbeVecVerifier`]: crate::audit::AuditProbeVecVerifier
/// [`try_match_events`]: crate::audit::try_match_events
pub trait AuditProbeEventHandler: Debug {
    /// Handler name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Handle an audit log event.
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()>;
}

/// The identity of an audit event: label, source location and time.
///
/// Shared by [`AuditProbeEvent`] and [`AuditProbeEventStub`]. The
/// [`AuditProbe`] checkpoint methods fill it from their label and their
/// `#[track_caller]` location. The header is diagnostic only:
/// [`try_match_events`] does not compare it, but puts it in the error message,
/// so a mismatch names the checkpoint that failed.
///
/// [`AuditProbe`]: crate::audit::AuditProbe
/// [`try_match_events`]: crate::audit::try_match_events
#[derive(Debug, Clone, PartialEq)]
pub struct AuditProbeEventHeader {
    /// Label of the event.
    pub label: Option<String>,

    /// Location of the event.
    pub loc: Option<LocationDesc>,

    /// The timestamp of the event.
    pub ts: SystemTime,
}

impl AuditProbeEventHeader {
    /// Create a new [`AuditProbeEventHeader`].
    pub fn new(
        label: Option<String>,
        loc: Option<LocationDesc>,
        ts: Option<SystemTime>,
    ) -> Self {
        Self {
            label,
            loc,
            ts: ts.unwrap_or_else(SystemTime::now),
        }
    }
}

/// Read access common to owned and borrowed audit events.
///
/// Implemented by [`AuditProbeEvent`] (owned, as recorded) and
/// [`AuditProbeEventStub`] (borrowed, as emitted), so [`try_match_events`] and
/// [`unpack_audit_probe_event_data!`] can compare an incoming stub with a
/// stored event without copying either. An event is an
/// [`AuditProbeEventHeader`], [`AuditProbeEventParams`], and a data map from
/// names to one or more [`TensorData`].
///
/// [`try_match_events`]: crate::audit::try_match_events
/// [`unpack_audit_probe_event_data!`]: crate::audit::unpack_audit_probe_event_data
pub trait AuditProbeEventView: Debug {
    /// Return an owned [`AuditProbeEvent`].
    fn to_event(&self) -> AuditProbeEvent {
        AuditProbeEvent {
            header: self.header().clone(),
            params: self.params().clone(),
            data: self.to_owned_data_map(),
        }
    }

    /// Get a stub-view of this event.
    fn to_stub(&self) -> AuditProbeEventStub<'_>;

    /// Common event header.
    fn header(&self) -> &AuditProbeEventHeader;

    /// Type-params for [`AuditProbeEvent`].
    fn params(&self) -> &AuditProbeEventParams;

    /// Iterate over the data in the audit log event.
    fn data_map_iter(&self) -> impl Iterator<Item = (&str, Vec<&TensorData>)>;

    /// Map view over the data in the audit log event.
    fn data_map_view(&self) -> HashMap<&str, Vec<&TensorData>> {
        self.data_map_iter().collect()
    }

    /// Map signature over the data in the audit log event.
    fn data_map_shape_signature(&self) -> HashMap<String, Vec<Shape>> {
        self.data_map_iter()
            .map(|(k, v)| {
                (
                    k.to_owned(),
                    v.iter().map(|&d| d.shape.clone()).collect::<Vec<Shape>>(),
                )
            })
            .collect()
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

    /// Assert that the shape signatures of the data map match the expected
    /// event.
    fn assert_shape_signatures_eq(
        &self,
        expected: &impl AuditProbeEventView,
    ) -> BunsenResult<()> {
        let actual_shape_sig = self.data_map_shape_signature();
        let expected_shape_sig = expected.data_map_shape_signature();
        if actual_shape_sig != expected_shape_sig {
            // TODO: Better error message.
            Err(BunsenError::Invalid(format!(
                "data map signatures don't match:\nactual: {:#?}\nexpect: {:#?}",
                actual_shape_sig, expected_shape_sig,
            )))
        } else {
            Ok(())
        }
    }
}

/// An owned audit event: what recorders keep, and verifiers expect.
///
/// Made from an emitted [`AuditProbeEventStub`] with
/// [`to_event`](AuditProbeEventView::to_event). An in-memory stream is a
/// `Vec<AuditProbeEvent>` ([`AuditStreamRecorder::events`]); on disk, each
/// event is an [`AuditEventRecord`] in an [`AuditStreamFile`].
///
/// [`AuditStreamRecorder::events`]: crate::audit::AuditStreamRecorder::events
/// [`AuditEventRecord`]: crate::audit::AuditEventRecord
/// [`AuditStreamFile`]: crate::audit::AuditStreamFile
#[derive(Debug, Clone)]
pub struct AuditProbeEvent {
    /// Common event header.
    pub header: AuditProbeEventHeader,

    /// Type-params for [`AuditProbeEvent`].
    pub params: AuditProbeEventParams,

    /// Attached event data.
    pub data: HashMap<String, Vec<TensorData>>,
}

impl AuditProbeEventView for AuditProbeEvent {
    fn to_stub(&self) -> AuditProbeEventStub<'_> {
        AuditProbeEventStub {
            header: self.header.clone(),
            params: self.params.clone(),
            data: self
                .data
                .iter()
                .map(|(k, v)| (k.clone(), v.iter().collect::<Vec<_>>()))
                .collect(),
        }
    }

    fn header(&self) -> &AuditProbeEventHeader {
        &self.header
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
}

/// A borrowed audit event, as an [`AuditProbe`] emits it.
///
/// Handlers receive a stub in [`AuditProbeEventHandler::on_event`]. It borrows
/// the checkpoint's [`TensorData`], so a handler that only compares (a
/// verifier) never copies the data; one that keeps it (a recorder) calls
/// [`to_event`](AuditProbeEventView::to_event) for an [`AuditProbeEvent`].
///
/// [`AuditProbe`]: crate::audit::AuditProbe
#[derive(Debug, Clone)]
pub struct AuditProbeEventStub<'a> {
    /// Common event header.
    pub header: AuditProbeEventHeader,

    /// Type-params for [`AuditProbeEvent`].
    pub params: AuditProbeEventParams,

    /// Stub-data for [`AuditProbeEvent`].
    pub data: HashMap<String, Vec<&'a TensorData>>,
}

impl<'a> AuditProbeEventView for AuditProbeEventStub<'a> {
    fn to_stub(&self) -> AuditProbeEventStub<'a> {
        self.clone()
    }

    fn header(&self) -> &AuditProbeEventHeader {
        &self.header
    }

    fn params(&self) -> &AuditProbeEventParams {
        &self.params
    }

    fn data_map_iter(&self) -> impl Iterator<Item = (&str, Vec<&TensorData>)> {
        self.data.iter().map(|(k, v)| (k.as_ref(), v.clone()))
    }
}

#[cfg(test)]
mod test {
    use std::panic::Location;

    use super::*;
    use crate::burner::descriptors::{
        ToleranceDesc,
        TolerancePolicy,
    };

    #[test]
    fn test_event_stub() -> BunsenResult<()> {
        let header = AuditProbeEventHeader::new(
            Some("example header".to_string()),
            Some(Location::caller().into()),
            None,
        );
        let params = AuditProbeEventParams::AssertTensorApproxEx {
            tolerance: ToleranceDesc::of::<f32>(TolerancePolicy::default())?,
        };

        let a = TensorData::from([1, 2, 3]);
        let b = TensorData::from([[2.0, 3.0], [4.0, 5.0]]);

        let event = AuditProbeEvent {
            header: header.clone(),
            params: params.clone(),
            data: HashMap::from([
                ("x".to_string(), vec![a.clone()]),
                ("y".to_string(), vec![a.clone(), b.clone()]),
            ]),
        };

        let stub = AuditProbeEventStub {
            header: header.clone(),
            params: params.clone(),
            data: HashMap::from([("x".to_string(), vec![&a]), ("y".to_string(), vec![&a, &b])]),
        };

        assert_eq!(event.data_map_view(), stub.data_map_view());

        Ok(())
    }
}
