use core::fmt::Debug;

use crate::burner::testing::audit::{
    AuditProbeEvent,
    handlers::AuditProbeEventRecorder,
};

/// An [`AuditProbeEvent`] recorder.
#[derive(Debug, Clone, Default)]
pub struct AuditProbeVecRecorder {
    log: Vec<AuditProbeEvent>,
}

impl AuditProbeEventRecorder for AuditProbeVecRecorder {
    fn log_event(
        &mut self,
        event: AuditProbeEvent,
    ) {
        self.log.push(event);
    }
}
