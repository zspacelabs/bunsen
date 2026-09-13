use core::fmt::Debug;

use crate::{
    burner::testing::audit::{
        AuditProbeEvent,
        AuditProbeEventStub,
        AuditProbeEventView,
    },
    errors::BunsenResult,
};

/// Audit log event handler.
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

/// A [`AuditProbeEvent`] recorder.
pub trait AuditProbeEventRecorder: Debug {
    /// Log an [`AuditProbeEvent`].
    fn log_event(
        &mut self,
        event: AuditProbeEvent,
    );
}

impl<T: AuditProbeEventRecorder> AuditProbeEventHandler for T {
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()> {
        self.log_event(stub.to_event());
        Ok(())
    }
}
