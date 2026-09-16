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
        DType,
        Element,
        FloatDType,
        f16,
    },
};
use num_traits::Float;
use time::{
    OffsetDateTime,
    macros::format_description,
};

use crate::{
    burner::{
        descriptors::{
            ToleranceDesc,
            TolerancePolicy,
        },
        testing::audit::{
            AuditProbeEventHandler,
            AuditProbeEventHeader,
            AuditProbeEventStub,
            AuditProbeEventView,
            unpack_audit_probe_event_data,
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
pub trait DataArg<'a>: Sized {
    /// Build a default [`CloneRef`] to [`TensorData`].
    fn into_data_ref(self) -> CloneRef<'a, TensorData>;

    /// Build a [`CloneRef`] to [`TensorData`] with a specific element type.
    fn into_data_ref_as<E: Element>(self) -> CloneRef<'a, TensorData>;

    /// Build a [`CloneRef`] to [`TensorData`] with a specific element type.
    fn into_data_ref_cast(
        self,
        dtype: DType,
    ) -> CloneRef<'a, TensorData> {
        match dtype {
            DType::QFloat(_) => self.into_data_ref_as::<f32>(),
            DType::F64 => self.into_data_ref_as::<f64>(),
            DType::F16 => self.into_data_ref_as::<f16>(),
            DType::F32 => self.into_data_ref_as::<f32>(),
            DType::Flex32 => self.into_data_ref_as::<f32>(),
            DType::BF16 => self.into_data_ref_as::<f16>(),
            DType::I8 => self.into_data_ref_as::<i8>(),
            DType::I16 => self.into_data_ref_as::<i16>(),
            DType::I32 => self.into_data_ref_as::<i32>(),
            DType::I64 => self.into_data_ref_as::<i64>(),
            DType::U8 => self.into_data_ref_as::<u8>(),
            DType::U16 => self.into_data_ref_as::<u16>(),
            DType::U32 => self.into_data_ref_as::<u32>(),
            DType::U64 => self.into_data_ref_as::<u64>(),
            DType::Bool(_) => self.into_data_ref_as::<bool>(),
        }
    }
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

impl<'a> AuditProbeEventHandler for AuditProbe<'a> {
    fn on_event(
        &mut self,
        stub: &AuditProbeEventStub<'_>,
    ) -> BunsenResult<()> {
        for handler in &mut self.handlers {
            handler.on_event(stub)?;
        }
        Ok(())
    }
}

impl<'a> AuditProbe<'a> {
    /// Construct a new audit probe.
    pub fn new(handlers: Vec<&'a mut dyn AuditProbeEventHandler>) -> Self {
        Self { handlers }
    }

    /// [`Tensor`] / [`TensorData`] equality.
    ///
    /// # Arguments
    /// * `header` - Event header.
    /// * `data` - the [`TensorData`] to compare.
    ///
    /// # Panics and/or Err Returns
    /// If `policy` is out of range, or if the data do not match under
    /// verification.
    fn dispatch_assert_eq(
        &mut self,
        header: AuditProbeEventHeader,
        data: &TensorData,
    ) -> BunsenResult<()> {
        let data: HashMap<String, Vec<&TensorData>> =
            HashMap::from([("data".to_string(), vec![data])]);

        self.on_event(&AuditProbeEventStub {
            header,
            params: AuditProbeEventParams::AssertTensorEq,
            data,
        })
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
    /// * `tolerance` - the [`TolerancePolicy`] to compare under.
    ///
    /// # Panics and/or Err Returns
    /// If `policy` is out of range, or if the data do not match under
    /// verification.
    fn dispatch_assert_approx_eq(
        &mut self,
        header: AuditProbeEventHeader,
        data: &TensorData,
        tolerance: ToleranceDesc,
    ) -> BunsenResult<()> {
        let data: HashMap<String, Vec<&TensorData>> =
            HashMap::from([("data".to_string(), vec![data])]);

        self.on_event(&AuditProbeEventStub {
            header,
            params: AuditProbeEventParams::AssertTensorApproxEx { tolerance },
            data,
        })
    }

    /// [`Tensor`] / [`TensorData`] equality.
    ///
    /// Converts to `E` before storage/comparison.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `data` - the data to compare.
    ///
    /// # Result
    /// This *may* return an error if the data or data types do not match under
    /// verification.
    #[track_caller]
    pub fn assert_eq_as<'b, E: Element>(
        &mut self,
        label: &str,
        data: impl DataArg<'b>,
    ) -> BunsenResult<()> {
        let header = AuditProbeEventHeader::new(
            Some(label.to_string()),
            Some(Location::caller().into()),
            None,
        );
        let data_cr = data.into_data_ref_as::<E>();

        self.dispatch_assert_eq(header, data_cr.as_ref())
    }

    /// [`Tensor`] / [`TensorData`] equality.
    ///
    /// # Arguments
    /// * `label` - Event Label.
    /// * `dtype` - the data type to convert to / compare as.
    /// * `data` - the data to compare.
    ///
    /// # Result
    /// This *may* return an error if the data or data types do not match under
    /// verification.
    #[track_caller]
    pub fn assert_eq_cast<'b>(
        &mut self,
        label: &str,
        data: impl DataArg<'b>,
        dtype: DType,
    ) -> BunsenResult<()> {
        let header = AuditProbeEventHeader::new(
            Some(label.to_string()),
            Some(Location::caller().into()),
            None,
        );
        let data_cr = data.into_data_ref_cast(dtype);

        self.dispatch_assert_eq(header, data_cr.as_ref())
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
    pub fn assert_approx_eq_as<'b, F: Float + Element>(
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

        let tolerance = ToleranceDesc::of::<F>(tolerance)?;

        self.dispatch_assert_approx_eq(header, data_cr.as_ref(), tolerance)
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
    pub fn assert_approx_eq_cast<'b>(
        &mut self,
        label: &str,
        data: impl DataArg<'b>,
        tolerance: TolerancePolicy,
        dtype: FloatDType,
    ) -> BunsenResult<()> {
        let header = AuditProbeEventHeader::new(
            Some(label.to_string()),
            Some(Location::caller().into()),
            None,
        );
        let (data_cr, tolerance) = match dtype {
            FloatDType::F64 => (
                data.into_data_ref_as::<f64>(),
                ToleranceDesc::of::<f64>(tolerance)?,
            ),
            FloatDType::F32 => (
                data.into_data_ref_as::<f32>(),
                ToleranceDesc::of::<f32>(tolerance)?,
            ),
            FloatDType::Flex32 => (
                data.into_data_ref_as::<f32>(),
                ToleranceDesc::of::<f32>(tolerance)?,
            ),
            FloatDType::F16 => (
                data.into_data_ref_as::<f16>(),
                ToleranceDesc::of::<f16>(tolerance)?,
            ),
            FloatDType::BF16 => (
                data.into_data_ref_as::<f16>(),
                ToleranceDesc::of::<f16>(tolerance)?,
            ),
        };

        self.dispatch_assert_approx_eq(header, data_cr.as_ref(), tolerance)
    }
}

/// Type-specific event params.
#[derive(Debug, Clone, PartialEq, strum::Display, strum::EnumDiscriminants)]
#[strum_discriminants(derive(strum::AsRefStr))]
pub enum AuditProbeEventParams {
    /// `assert_eq` event.
    AssertTensorEq,

    /// `assert_approx_eq` event.
    AssertTensorApproxEx {
        /// Tolerance.
        tolerance: ToleranceDesc,
    },
}

impl AuditProbeEventParams {
    /// Get the variant name.
    pub fn variant_name(&self) -> String {
        let d = AuditProbeEventParamsDiscriminants::from(self);

        let name: &str = d.as_ref();

        name.to_string()
    }
}

/// Applies per-type event equality.
pub fn try_match_events(
    actual: &impl AuditProbeEventView,
    expected: &impl AuditProbeEventView,
) -> BunsenResult<()> {
    fn cmp(
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
            AuditProbeEventParams::AssertTensorEq => {
                let [actual_data, expected_data] =
                    unpack_audit_probe_event_data!([actual, expected], { data })?;
                actual_data.data.try_assert_eq(expected_data.data, true)
            }
            AuditProbeEventParams::AssertTensorApproxEx { tolerance } => {
                let [actual_data, expected_data] =
                    unpack_audit_probe_event_data!([actual, expected], { data })?;

                tolerance.try_assert_tensor_data_approx_eq(
                    actual_data.data,
                    expected_data.data,
                    true,
                )
            }
        }
    }
    cmp(actual, expected).map_err(|err| {
        let header = actual.header();
        let params = actual.params();

        let mut msg = format!("AuditEvent::{name}:", name = params.variant_name());
        if let Some(label) = &header.label {
            msg.push_str(&format!(" \"{label}\""));
        }

        let ts_utc: OffsetDateTime = header.ts.into();
        // 2. Format using a macro-compiled description
        let format = format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");
        let formatted = ts_utc.format(&format).unwrap();
        msg.push_str(&format!(" @{formatted}\n"));

        if let Some(loc) = &header.loc {
            msg.push_str(&format!(" at {loc}\n"));
        }
        msg.push_str(&format!("{params:#?}"));
        msg.push_str(&format!("{err}"));

        BunsenError::AssertionError(msg)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::testing::audit::AuditProbeVecRecorder,
        errors::WithOkOrPanic,
        support::testing::{
            CpuBackend,
            PerformanceBackend,
            default_device,
        },
    };

    #[test]
    #[serial_test::serial]
    #[should_panic = "iota"]
    fn test_missmatch() {
        let mut recorder = AuditProbeVecRecorder::default();
        let probe = &mut AuditProbe::new(vec![&mut recorder]);
        {
            type B = PerformanceBackend;
            let device = default_device();

            let iota: Tensor<B, 1> = Tensor::arange(0..10, &device).float();
            probe.assert_eq_as::<f64>("iota", &iota).ok_or_panic();
        }

        let mut verifier = recorder.into_verifier();
        let probe = &mut AuditProbe::new(vec![&mut verifier]);
        {
            type B = CpuBackend;
            let device = default_device();

            let iota: Tensor<B, 1> = Tensor::arange(5..15, &device).float();
            probe.assert_eq_as::<f64>("iota", &iota).ok_or_panic();
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_stream() -> BunsenResult<()> {
        fn example<B: Backend>(probe: &mut AuditProbe) -> BunsenResult<()> {
            let device = default_device();

            let iota: Tensor<B, 1> = Tensor::arange(0..10, &device).float();
            probe.assert_eq_as::<f64>("iota", &iota)?;
            probe.assert_eq_cast("iota", &iota, DType::F16)?;

            probe.assert_approx_eq_as::<f32>(
                "iota.exp",
                &iota.clone().exp(),
                TolerancePolicy::Balanced,
            )?;

            probe.assert_approx_eq_cast(
                "iota.sin",
                &iota.clone().sin(),
                TolerancePolicy::Balanced,
                FloatDType::F16,
            )?;

            Ok(())
        }

        let mut recorder = AuditProbeVecRecorder::default();
        example::<PerformanceBackend>(&mut AuditProbe::new(vec![&mut recorder]))?;

        let mut verifier = recorder.into_verifier();
        example::<CpuBackend>(&mut AuditProbe::new(vec![&mut verifier]))?;

        Ok(())
    }
}
