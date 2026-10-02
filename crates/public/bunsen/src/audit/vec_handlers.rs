use core::fmt::Debug;

use crate::{
    audit::{
        AuditProbeEvent,
        AuditProbeEventStub,
        AuditProbeEventView,
        audit_probe::try_match_events,
        audit_probe_event::AuditProbeEventHandler,
    },
    errors::BunsenResult,
};

/// An [`AuditProbeEventHandler`] that records events in a public `Vec`.
///
/// The older in-memory recorder; [`into_verifier`](Self::into_verifier) gives
/// an [`AuditProbeVecVerifier`]. Prefer [`AuditStreamRecorder`], whose verifier
/// detects a short run, and which can save to disk.
///
/// [`AuditStreamRecorder`]: crate::audit::AuditStreamRecorder
#[derive(Debug, Clone, Default)]
pub struct AuditProbeVecRecorder {
    /// Recorded events.
    pub events: Vec<AuditProbeEvent>,
}

impl AuditProbeVecRecorder {
    /// Build a verifier.
    pub fn into_verifier(self) -> AuditProbeVecVerifier {
        AuditProbeVecVerifier::new(self.events)
    }
}

impl AuditProbeEventHandler for AuditProbeVecRecorder {
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()> {
        self.events.push(stub.to_event());
        Ok(())
    }
}

/// An [`AuditProbeEventHandler`] that verifies events against an expected
/// `Vec`, in order.
///
/// Each event is matched against the next expected one with
/// [`try_match_events`]. It panics on an event past the end, and cannot report
/// expected events that never arrived; prefer [`AuditStreamVerifier`].
///
/// [`AuditStreamVerifier`]: crate::audit::AuditStreamVerifier
#[derive(Debug, Clone, Default)]
pub struct AuditProbeVecVerifier {
    events: Vec<AuditProbeEvent>,
    next: Option<usize>,
}

impl AuditProbeVecVerifier {
    /// Create a new [`AuditProbeVecVerifier`] from a vector of
    /// [`AuditProbeEvent`]s.
    pub fn new(events: Vec<AuditProbeEvent>) -> Self {
        let next = if events.is_empty() { None } else { Some(0) };
        Self { events, next }
    }

    fn next_expected_event(&mut self) -> Option<AuditProbeEventStub<'_>> {
        match self.next {
            None => None,
            Some(idx) => {
                let event = &self.events[idx];
                if idx + 1 < self.events.len() {
                    self.next = Some(idx + 1);
                } else {
                    self.next = None;
                }
                Some(event.to_stub())
            }
        }
    }
}

impl AuditProbeEventHandler for AuditProbeVecVerifier {
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()> {
        try_match_events(stub, &self.next_expected_event().unwrap())
    }
}
