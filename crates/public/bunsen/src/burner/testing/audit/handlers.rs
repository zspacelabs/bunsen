use core::fmt::Debug;

use crate::{
    burner::testing::audit::{
        AuditProbeEventStub,
        AuditProbeEventView,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
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

/// TODO: try match events
pub fn try_match_events(
    actual: &impl AuditProbeEventView,
    expected: &impl AuditProbeEventView,
) -> BunsenResult<()> {
    if expected.params() != actual.params() {
        return Err(BunsenError::InvalidArgument {
            msg: format!(
                "StreamEvent params {:?} != expected {:?}",
                expected.params(),
                actual.params()
            ),
        });
    }

    // TODO: BunsenError::assert_eq(a, b, msg) -> BunsenResult<()>.
    let actual_shape_sig = actual.data_map_shape_signature();
    let expected_shape_sig = expected.data_map_shape_signature();
    if actual_shape_sig != expected_shape_sig {
        // TODO: Better error message.
        return Err(BunsenError::InvalidArgument {
            msg: format!(
                "data map signatures don't match:\nactual: {:#?}\nexpect: {:#?}",
                actual_shape_sig, expected_shape_sig,
            ),
        });
    }

    /*
    if !actual
        .data()
        .iter()
        .zip(expected.data().iter())
        .all(|(a, e)| a.shape == e.shape)
    {
        return Err(BunsenError::InvalidArgument {
            msg: format!(
                "event data shapes do not match:\nactual: {:?}\nexpect: {:?}",
                actual
                    .data()
                    .iter()
                    .map(|d| d.shape.clone())
                    .collect::<Vec<_>>(),
                expected
                    .data()
                    .iter()
                    .map(|d| d.shape.clone())
                    .collect::<Vec<_>>()
            ),
        });
    }

    match expected.params() {
        EventParams::AssertEq { strict, .. } => {
            assert_eq!(actual.data().len(), 1);
            tensor_data_assert_eq(&actual.data()[0], &expected.data()[0], *strict)
        }
    }

     */

    Ok(())
}
