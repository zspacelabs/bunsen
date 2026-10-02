use std::{
    collections::BTreeMap,
    time::SystemTime,
};

use burn::prelude::TensorData;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    audit::{
        AuditProbeEvent,
        AuditProbeEventHeader,
        AuditProbeEventParams,
    },
    burner::descriptors::ToleranceDesc,
    support::reflection::LocationDesc,
};

/// Serializable mirror of [`AuditProbeEventParams`].
///
/// The `params` of an [`AuditEventRecord`]; converts both ways with
/// [`AuditProbeEventParams`]. A separate type pins the on-disk form of an
/// [`AuditStreamFile`] to serde's `kind` tag, independent of the in-memory
/// enum.
///
/// [`AuditStreamFile`]: crate::audit::AuditStreamFile
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum AuditParamsRecord {
    /// `AuditProbeEventParams::AssertTensorEq`.
    AssertTensorEq,

    /// `AuditProbeEventParams::AssertTensorApproxEx`.
    AssertTensorApproxEq {
        /// Tolerance.
        tolerance: ToleranceDesc,
    },
}

impl From<&AuditProbeEventParams> for AuditParamsRecord {
    fn from(params: &AuditProbeEventParams) -> Self {
        match params {
            AuditProbeEventParams::AssertTensorEq => Self::AssertTensorEq,
            AuditProbeEventParams::AssertTensorApproxEx { tolerance } => {
                Self::AssertTensorApproxEq {
                    tolerance: *tolerance,
                }
            }
        }
    }
}

impl From<AuditParamsRecord> for AuditProbeEventParams {
    fn from(record: AuditParamsRecord) -> Self {
        match record {
            AuditParamsRecord::AssertTensorEq => Self::AssertTensorEq,
            AuditParamsRecord::AssertTensorApproxEq { tolerance } => {
                Self::AssertTensorApproxEx { tolerance }
            }
        }
    }
}

/// Serializable mirror of [`AuditProbeEvent`].
///
/// One entry of an [`AuditStreamFile`]'s `events`; converts both ways with
/// [`AuditProbeEvent`]. [`save_audit_stream`] and [`load_audit_stream`] do the
/// conversion, so handlers and verifiers only see [`AuditProbeEvent`]s. The
/// data map is ordered, so a stream serializes the same way every time.
///
/// [`AuditStreamFile`]: crate::audit::AuditStreamFile
/// [`save_audit_stream`]: crate::audit::save_audit_stream
/// [`load_audit_stream`]: crate::audit::load_audit_stream
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEventRecord {
    /// Event label.
    pub label: Option<String>,

    /// Source location of the checkpoint.
    pub loc: Option<LocationDesc>,

    /// When the event was recorded.
    pub ts: SystemTime,

    /// Event params.
    pub params: AuditParamsRecord,

    /// Attached data.
    pub data: BTreeMap<String, Vec<TensorData>>,
}

impl From<&AuditProbeEvent> for AuditEventRecord {
    fn from(event: &AuditProbeEvent) -> Self {
        Self {
            label: event.header.label.clone(),
            loc: event.header.loc.clone(),
            ts: event.header.ts,
            params: (&event.params).into(),
            data: event
                .data
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }
}

impl From<AuditEventRecord> for AuditProbeEvent {
    fn from(record: AuditEventRecord) -> Self {
        Self {
            header: AuditProbeEventHeader::new(record.label, record.loc, Some(record.ts)),
            params: record.params.into(),
            data: record.data.into_iter().collect(),
        }
    }
}
