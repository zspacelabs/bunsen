use core::fmt::Debug;
use std::{
    collections::HashMap,
    panic::Location,
};

use burn::{
    Tensor,
    prelude::{
        Backend,
        TensorData,
    },
    tensor::{
        BasicOps,
        Element,
        Tolerance,
    },
};
use num_traits::Float;

use crate::{
    burner::testing::audit::{
        AuditProbeEventHeader,
        AuditProbeEventStub,
        AuditProbeEventView,
        audit_probe_event::AuditProbeEventHandler,
        unpack_event_data,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    prelude::TensorElemOpExt,
};

/// Probe for auditing purposes.
#[derive(Debug)]
pub struct AuditProbe<'a> {
    handlers: Vec<&'a mut dyn AuditProbeEventHandler>,
}

impl<'a> AuditProbe<'a> {
    /// Construct a new audit probe.
    pub fn new(handlers: Vec<&'a mut dyn AuditProbeEventHandler>) -> Self {
        Self { handlers }
    }

    /// Dispatch event to handlers.
    ///
    /// # Arguments
    /// * `event` - an event stub to handle.
    ///
    /// # Panics and/or Err Returns
    /// If the event does not match the expected event under verification.
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()> {
        for handler in &mut self.handlers {
            handler.on_event(stub)?;
        }
        Ok(())
    }

    /// Location forwarding impl of `assert_eq`.
    fn loc_assert_eq(
        &mut self,
        _label: &str,
        _location: &Location,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        let data: HashMap<String, Vec<&TensorData>> =
            HashMap::from([("data".to_string(), vec![data])]);

        self.on_event(&AuditProbeEventStub {
            header: AuditProbeEventHeader::new(None),
            params: AuditProbeEventParams::AssertEq { strict },
            data,
        })
    }

    fn loc_assert_approx_eq<F: Float + Element>(
        &mut self,
        _label: &str,
        _location: &Location,
        data: &TensorData,
        tolerance: ToleranceDesc,
    ) -> BunsenResult<()> {
        let data: HashMap<String, Vec<&TensorData>> =
            HashMap::from([("data".to_string(), vec![data])]);

        self.on_event(&AuditProbeEventStub {
            header: AuditProbeEventHeader::new(None),
            params: AuditProbeEventParams::AssertApproxEq { tolerance },
            data,
        })
    }

    /// [`TensorData`] equality.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    #[track_caller]
    pub fn assert_eq(
        &mut self,
        label: &str,
        data: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        self.loc_assert_eq(label, Location::caller(), data, strict)
    }

    /// Assert approximately equal.
    #[track_caller]
    pub fn assert_approx_eq<F: Float + Element>(
        &mut self,
        label: &str,
        data: &TensorData,
        tolerance: ToleranceDesc,
    ) -> BunsenResult<()> {
        self.loc_assert_approx_eq::<F>(label, Location::caller(), data, tolerance)
    }

    /// [`Tensor`] equality.
    ///
    /// Data is used as `tensor.to_data_as::<E>()`.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    #[track_caller]
    pub fn assert_tensor_eq_as<B, const R: usize, K, E>(
        &mut self,
        label: &str,
        tensor: &Tensor<B, R, K>,
        strict: bool,
    ) -> BunsenResult<()>
    where
        E: Element,
        B: Backend,
        K: BasicOps<B>,
    {
        let data = tensor.to_data_as::<E>();
        self.loc_assert_eq(label, Location::caller(), &data, strict)
    }
}

/// Description of a tolerance.
#[derive(Debug, Clone, PartialEq)]
pub enum ToleranceDesc {
    /// Default tolerance, see [`Tolerance::default`].
    Default,
}

impl ToleranceDesc {
    /// get the burn tolerance.
    pub fn tolerance<F: Float + Element>(&self) -> Tolerance<F> {
        match self {
            ToleranceDesc::Default => Tolerance::default(),
        }
    }
}

/// Type-specific event params.
#[derive(Debug, Clone, PartialEq)]
pub enum AuditProbeEventParams {
    /// `assert_eq` event.
    AssertEq {
        /// Strict dtype comparison.
        strict: bool,
    },

    /// `assert_approx_eq` event.
    AssertApproxEq {
        /// Tolerance.
        tolerance: ToleranceDesc,
    },
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

    actual.assert_shape_signatures_eq(expected)?;

    match actual.params() {
        AuditProbeEventParams::AssertEq { strict } => {
            let [actual_data, expected_data] = unpack_event_data!([actual, expected], { data })?;

            tensor_data_assert_eq(actual_data.data, expected_data.data, *strict)
        }
        AuditProbeEventParams::AssertApproxEq { tolerance } => {
            let [actual_data, expected_data] = unpack_event_data!([actual, expected], { data })?;

            actual_data
                .data
                .assert_approx_eq::<f64>(expected_data.data, tolerance.tolerance::<f64>());
            Ok(())
        }
    }
}

/// [`TensorData::assert_eq`] api, returning a [`BunsenResult`].
pub fn tensor_data_assert_eq(
    actual: &TensorData,
    expected: &TensorData,
    strict: bool,
) -> BunsenResult<()> {
    // TODO: Result-generating version of this.
    // * Expand `burn` api.
    // * Clone `burn` api, generate Results.
    // * `panic::catch_unwind` version of this.
    actual.assert_eq(expected, strict);

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]

    use burn::prelude::Int;

    use super::*;
    use crate::{
        burner::testing::audit::AuditProbeVecRecorder,
        support::testing::{
            CpuBackend,
            PerformanceBackend,
        },
    };

    #[test]
    #[serial_test::serial]
    fn test_stream() -> BunsenResult<()> {
        fn example<B: Backend>(stream: &mut AuditProbe) -> BunsenResult<()> {
            let device = Default::default();

            let a = TensorData::from([1.0, 2.0]);
            stream.assert_eq("a", &a, false)?;

            stream.assert_approx_eq::<f32>("a.2", &a, ToleranceDesc::Default)?;

            let b: Tensor<B, 1, Int> = Tensor::arange(0..4, &device);
            stream.assert_tensor_eq_as::<B, _, _, f32>("b", &b, false)?;

            Ok(())
        }

        let mut recorder = AuditProbeVecRecorder::default();
        example::<PerformanceBackend>(&mut AuditProbe::new(vec![&mut recorder]))?;

        let mut verifier = recorder.verifier();
        example::<CpuBackend>(&mut AuditProbe::new(vec![&mut verifier]))?;

        Ok(())
    }
}
