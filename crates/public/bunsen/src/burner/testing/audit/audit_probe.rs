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
    },
};
use num_traits::Float;

use crate::{
    burner::{
        descriptors::{
            ToleranceDesc,
            TolerancePolicy,
        },
        testing::audit::{
            AuditProbeEventHeader,
            AuditProbeEventStub,
            AuditProbeEventView,
            audit_probe_event::AuditProbeEventHandler,
            unpack_event_data,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    prelude::{
        TensorDataCheckExt,
        TensorElemOpExt,
    },
    support::CloneRef,
};

/// Argument conversion trait for [`Tensor`] and [`TensorData`].
pub trait DataArg<'a> {
    /// Build a default [`CloneRef`] to [`TensorData`].
    fn into_data_ref(self) -> CloneRef<'a, TensorData>;

    /// Build a [`CloneRef`] to [`TensorData`] with a specific element type.
    fn into_data_ref_as<E: Element>(self) -> CloneRef<'a, TensorData>;
}

impl<'a> DataArg<'a> for &'a TensorData {
    fn into_data_ref(self) -> CloneRef<'a, TensorData> {
        CloneRef::Ref(self)
    }

    fn into_data_ref_as<E: Element>(self) -> CloneRef<'a, TensorData> {
        CloneRef::Clone(self.clone().convert::<E>())
    }
}

impl<'a, B: Backend, const R: usize, K: BasicOps<B>> DataArg<'a> for &'a Tensor<B, R, K> {
    fn into_data_ref(self) -> CloneRef<'a, TensorData> {
        CloneRef::Clone(self.to_data())
    }

    fn into_data_ref_as<E: Element>(self) -> CloneRef<'a, TensorData> {
        CloneRef::Clone(self.to_data_as::<E>())
    }
}

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

    /// [`Tensor`] / [`TensorData`] approximate equality.
    ///
    /// Data is compared at the float type `F`.
    ///
    /// `F` is recorded in the event as part of the [`ToleranceDesc`], so a
    /// replay comparing at a different float type fails on params equality
    /// rather than silently applying a different tolerance.
    ///
    /// # Arguments
    /// * `header` - Event header.
    /// * `data` - the [`TensorData`] to compare.
    /// * `strict` - whether to use strict dtype equality.
    /// * `tolerance` - the [`TolerancePolicy`] to compare under.
    ///
    /// # Panics and/or Err Returns
    /// If `policy` is out of range, or if the data do not match under
    /// verification.
    fn try_header_tensor_approx_eq<F: Float + Element>(
        &mut self,
        header: AuditProbeEventHeader,
        data: &TensorData,
        strict: bool,
        policy: Option<TolerancePolicy>,
    ) -> BunsenResult<()> {
        let data: HashMap<String, Vec<&TensorData>> =
            HashMap::from([("data".to_string(), vec![data])]);

        let tolerance = policy.map(|p| ToleranceDesc::of::<F>(p)).transpose()?;

        self.on_event(&AuditProbeEventStub {
            header,
            params: AuditProbeEventParams::AssertTensorEq { strict, tolerance },
            data,
        })
    }

    /// [`Tensor`] / [`TensorData`] equality.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the data to compare.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Result
    /// This *may* return an error if the data or data types do not match under
    /// verification.
    #[track_caller]
    pub fn try_tensor_eq<'b>(
        &mut self,
        label: &str,
        data: impl DataArg<'b>,
        strict: bool,
    ) -> BunsenResult<()> {
        let header = AuditProbeEventHeader::new(
            Some(label.to_string()),
            Some(Location::caller().into()),
            None,
        );
        let data_cr = data.into_data_ref();

        self.try_header_tensor_approx_eq::<f32>(header, data_cr.as_ref(), strict, None)
    }

    /// [`Tensor`] / [`TensorData`] approximate equality.
    ///
    /// Data is compared at the float type `F`.
    ///
    /// `F` is recorded in the event as part of the [`ToleranceDesc`], so a
    /// replay comparing at a different float type fails on params equality
    /// rather than silently applying a different tolerance.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `tolerance` - the [`TolerancePolicy`] to compare under.
    ///
    /// # Panics and/or Err Returns
    /// If `policy` is out of range, or if the data do not match under
    /// verification.
    #[track_caller]
    pub fn try_tensor_approx_eq<'b, F: Float + Element>(
        &mut self,
        label: &str,
        data: impl DataArg<'b>,
        tolerance: TolerancePolicy,
    ) -> BunsenResult<()> {
        let header = AuditProbeEventHeader::new(
            Some(label.to_string()),
            Some(Location::caller().into()),
            None,
        );
        let data_cr = data.into_data_ref();

        self.try_header_tensor_approx_eq::<F>(header, data_cr.as_ref(), false, Some(tolerance))
    }

    /// [`Tensor`] / [`TensorData`] approximate equality.
    ///
    /// Data is converted to and compared at the float type `F`.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the [`TensorData`] to compare.
    /// * `tolerance` - the [`TolerancePolicy`] to compare under.
    ///
    /// # Panics and/or Err Returns
    /// If the data or data types do not match under verification.
    #[track_caller]
    pub fn try_tensor_approx_eq_as<'b, F: Float + Element>(
        &mut self,
        label: &str,
        data: impl DataArg<'b>,
        tolerance: TolerancePolicy,
    ) -> BunsenResult<()> {
        let header = AuditProbeEventHeader::new(
            Some(label.to_string()),
            Some(Location::caller().into()),
            None,
        );
        let data_cr = data.into_data_ref_as::<F>();

        self.try_header_tensor_approx_eq::<F>(header, data_cr.as_ref(), true, Some(tolerance))
    }
}

/// Type-specific event params.
#[derive(Debug, Clone, PartialEq)]
pub enum AuditProbeEventParams {
    /// `assert_approx_eq` event.
    AssertTensorEq {
        /// Strict dtype comparison.
        strict: bool,

        /// Tolerance.
        tolerance: Option<ToleranceDesc>,
    },
}

/// Applies per-type event equality.
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
        AuditProbeEventParams::AssertTensorEq { tolerance, strict } => {
            let [actual_data, expected_data] = unpack_event_data!([actual, expected], { data })?;

            match tolerance {
                Some(tolerance) => tolerance.try_assert_tensor_data_approx_eq(
                    actual_data.data,
                    expected_data.data,
                    *strict,
                ),
                None => actual_data.data.try_assert_eq(expected_data.data, *strict),
            }
        }
    }
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
        fn example<B: Backend>(probe: &mut AuditProbe) -> BunsenResult<()> {
            let device = Default::default();

            let a = TensorData::from([1.0, 2.0]);
            probe.try_tensor_eq("a", &a, false)?;

            probe.try_tensor_approx_eq::<f32>("a.32", &a, TolerancePolicy::Balanced)?;
            probe.try_tensor_approx_eq_as::<f64>("a.64", &a, TolerancePolicy::Balanced)?;

            let b: Tensor<B, 1, Int> = Tensor::arange(0..4, &device);
            probe.try_tensor_eq("b.tensor", &b, false)?;
            probe.try_tensor_eq("b.tensor", &b.to_data(), false)?;

            Ok(())
        }

        let mut recorder = AuditProbeVecRecorder::default();
        example::<PerformanceBackend>(&mut AuditProbe::new(vec![&mut recorder]))?;

        assert_eq!(recorder.events.len(), 5);
        let event_a = &recorder.events[0];
        assert_eq!(event_a.header.label.as_ref().unwrap(), "a");

        let mut verifier = recorder.into_verifier();
        example::<CpuBackend>(&mut AuditProbe::new(vec![&mut verifier]))?;

        Ok(())
    }
}
