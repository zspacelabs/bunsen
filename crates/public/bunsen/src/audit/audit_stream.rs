use std::{
    fs,
    io::{
        BufReader,
        BufWriter,
    },
    path::Path,
};

use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    audit::{
        AuditEventRecord,
        AuditProbeEvent,
        AuditProbeEventHandler,
        AuditProbeEventStub,
        AuditProbeEventView,
        try_match_events,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// The on-disk format tag of an audit stream file.
pub const AUDIT_STREAM_FORMAT: &str = "bunsen-audit-stream";

/// The on-disk format version of an audit stream file.
pub const AUDIT_STREAM_VERSION: u32 = 1;

/// The on-disk form of an audit stream: a CBOR document holding the events, in
/// order.
///
/// Written by [`save_audit_stream`] (through [`AuditStreamRecorder::save`]) and
/// read by [`load_audit_stream`] (through [`AuditStreamVerifier::load`]), which
/// rejects another `format` or `version`. A stored baseline
/// ([`audit_baseline`]) is a file of this form.
///
/// [`audit_baseline`]: crate::audit::audit_baseline
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditStreamFile {
    /// Always [`AUDIT_STREAM_FORMAT`].
    pub format: String,

    /// The format version; see [`AUDIT_STREAM_VERSION`].
    pub version: u32,

    /// The events, in the order they were recorded.
    pub events: Vec<AuditEventRecord>,
}

/// Writes `events` to `path` as an [`AuditStreamFile`], creating parent
/// directories.
///
/// # Errors
/// [`BunsenError::External`] on I/O or encoding failure.
pub fn save_audit_stream(
    path: &Path,
    events: &[AuditProbeEvent],
) -> BunsenResult<()> {
    let file = AuditStreamFile {
        format: AUDIT_STREAM_FORMAT.to_string(),
        version: AUDIT_STREAM_VERSION,
        events: events.iter().map(AuditEventRecord::from).collect(),
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| io_error(path, e))?;
    }
    let out = fs::File::create(path).map_err(|e| io_error(path, e))?;
    ciborium::into_writer(&file, BufWriter::new(out))
        .map_err(|e| BunsenError::External(format!("writing {}: {e}", path.display())))
}

/// Reads the events of an [`AuditStreamFile`] from `path`.
///
/// # Errors
/// [`BunsenError::ResourceNotFound`] if `path` does not exist;
/// [`BunsenError::External`] on I/O or decoding failure;
/// [`BunsenError::Invalid`] on a wrong format tag or version.
pub fn load_audit_stream(path: &Path) -> BunsenResult<Vec<AuditProbeEvent>> {
    if !path.exists() {
        return Err(BunsenError::ResourceNotFound(format!(
            "audit stream {}",
            path.display()
        )));
    }
    let input = fs::File::open(path).map_err(|e| io_error(path, e))?;
    let file: AuditStreamFile = ciborium::from_reader(BufReader::new(input))
        .map_err(|e| BunsenError::External(format!("reading {}: {e}", path.display())))?;
    if file.format != AUDIT_STREAM_FORMAT || file.version != AUDIT_STREAM_VERSION {
        return Err(BunsenError::Invalid(format!(
            "{}: format {:?} version {}, expected {AUDIT_STREAM_FORMAT:?} version \
             {AUDIT_STREAM_VERSION}",
            path.display(),
            file.format,
            file.version,
        )));
    }
    Ok(file.events.into_iter().map(AuditProbeEvent::from).collect())
}

fn io_error(
    path: &Path,
    err: std::io::Error,
) -> BunsenError {
    BunsenError::External(format!("{}: {err}", path.display()))
}

/// An [`AuditProbeEventHandler`] that records each event in memory.
///
/// Use it on the reference run. Afterwards,
/// [`into_verifier`](Self::into_verifier) turns the recording into an
/// [`AuditStreamVerifier`] for an in-memory comparison ([`audit_across`]), and
/// [`save`](Self::save) writes it as an [`AuditStreamFile`]
/// ([`audit_baseline`]). Recording never fails.
///
/// [`audit_across`]: crate::audit::audit_across
/// [`audit_baseline`]: crate::audit::audit_baseline
#[derive(Debug, Clone, Default)]
pub struct AuditStreamRecorder {
    events: Vec<AuditProbeEvent>,
}

impl AuditProbeEventHandler for AuditStreamRecorder {
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()> {
        self.events.push(stub.to_event());
        Ok(())
    }
}

impl AuditStreamRecorder {
    /// The recorded events, in order.
    pub fn events(&self) -> &[AuditProbeEvent] {
        &self.events
    }

    /// Build a verifier expecting the recorded events.
    pub fn into_verifier(self) -> AuditStreamVerifier {
        AuditStreamVerifier::new(self.events)
    }

    /// Save the recorded events to `path`; see [`save_audit_stream`].
    ///
    /// # Errors
    /// See [`save_audit_stream`].
    pub fn save(
        &self,
        path: &Path,
    ) -> BunsenResult<()> {
        save_audit_stream(path, &self.events)
    }
}

/// An [`AuditProbeEventHandler`] that verifies events against an expected
/// stream, in order.
///
/// Built from a recording ([`AuditStreamRecorder::into_verifier`]) or a saved
/// [`AuditStreamFile`] ([`load`](Self::load)). Each event is matched against
/// the next expected one with [`try_match_events`]. Unlike
/// [`AuditProbeVecVerifier`], an event past the end of the stream is an error
/// rather than a panic, and [`finish`](Self::finish) reports expected events
/// that never arrived; call it when the run ends.
///
/// [`AuditProbeVecVerifier`]: crate::audit::AuditProbeVecVerifier
#[derive(Debug, Clone)]
pub struct AuditStreamVerifier {
    events: Vec<AuditProbeEvent>,
    next: usize,
}

impl AuditProbeEventHandler for AuditStreamVerifier {
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()> {
        let Some(expected) = self.events.get(self.next) else {
            return Err(BunsenError::AssertionError(format!(
                "unexpected audit event {:?}: the expected stream has only {} events",
                stub.header.label,
                self.events.len(),
            )));
        };
        self.next += 1;
        try_match_events(stub, expected)
    }
}

impl AuditStreamVerifier {
    /// A verifier expecting `events`, in order.
    pub fn new(events: Vec<AuditProbeEvent>) -> Self {
        Self { events, next: 0 }
    }

    /// A verifier expecting the stream saved at `path`.
    ///
    /// # Errors
    /// See [`load_audit_stream`].
    pub fn load(path: &Path) -> BunsenResult<Self> {
        Ok(Self::new(load_audit_stream(path)?))
    }

    /// Checks that every expected event was seen.
    ///
    /// # Errors
    /// [`BunsenError::AssertionError`] naming the first missing event.
    pub fn finish(&self) -> BunsenResult<()> {
        match self.events.get(self.next) {
            None => Ok(()),
            Some(missing) => Err(BunsenError::AssertionError(format!(
                "audit stream ended early: saw {} of {} events; next expected {:?}",
                self.next,
                self.events.len(),
                missing.header.label,
            ))),
        }
    }
}
