//! # Thermal Relaxation
use burn::{
    Tensor,
    tensor::{
        DType,
        Device,
    },
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::errors::{
    BunsenResult,
    ConstraintError,
    WithOkOrPanic,
};

/// The relaxation operator for
/// [`bgk_collision`](`super::collision::bgk_collision`).
///
/// Computes ``correction * (dist_a * (1 - omega) + dist_b * omega)``.
///
/// ## Correction
///
/// The `correction` term is a scale applied to the result. It is provided
/// as a way for dynamic corrections to be computed as fused operations;
/// and will default to `1.0` for `None`.
///
/// # Arguments
/// - `dist_a`: a `[H, W, VY=3, VX=3]` distribution.
/// - `dist_b`: a `[H, W, VY=3, VX=3]` distribution.
/// - `relaxation`: relaxation parameter.
/// - `correction`: fused correction factor for the relaxation operator;
///   defaults to 1.0.
///
/// # Returns
///
/// The `[H, W, VY=3, VX=3]` relaxed sum.
pub fn relaxed_sum<S: Into<OmegaSource>>(
    dist_a: Tensor<4>,
    dist_b: Tensor<4>,
    relaxation: S,
    correction: Option<f64>,
) -> Tensor<4> {
    let omega = relaxation
        .into()
        .omega(&dist_a.device(), dist_a.dtype())
        .unsqueeze_dims::<4>(&[2, 3]);

    let correction = correction.unwrap_or(1.0);
    dist_a * (correction * (1.0 - omega.clone())) + dist_b * (correction * omega)
}

/// Omegas for the BGK collision operator.
pub enum OmegaSource {
    /// Scalar relaxation frequency.
    Relaxation(RelaxationParam),

    /// Tensor relaxation frequency.
    Omega(Tensor<2>),
}

impl OmegaSource {
    /// Returns the omega tensor.
    pub fn omega(
        &self,
        device: &Device,
        dtype: DType,
    ) -> Tensor<2> {
        match self {
            OmegaSource::Relaxation(relaxation) => {
                Tensor::<1>::from_data([relaxation.as_omega_value()], (device, dtype)).unsqueeze()
            }
            OmegaSource::Omega(omega) => omega.clone().to_device(device).cast(dtype),
        }
    }
}

impl From<RelaxationParam> for OmegaSource {
    fn from(val: RelaxationParam) -> Self {
        OmegaSource::Relaxation(val)
    }
}

impl From<Tensor<2>> for OmegaSource {
    fn from(val: Tensor<2>) -> Self {
        OmegaSource::Omega(val)
    }
}

/// Wrapper for the BGK collision operator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RelaxationParam {
    /// Relaxation frequency (1/tau), typically in (0, 2)
    Omega(f64),

    /// Relaxation time (1/omega), typically > 0.5
    Tau(f64),
}

impl RelaxationParam {
    /// Checks that the relaxation is in range: `omega` in `[0, 2]`, or `tau`
    /// at least `0.5`.
    ///
    /// The fallible half of a `try_x` / `x` pair; the panicking half is
    /// [`validate`](Self::validate).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, if the value is out of range, or NaN.
    pub fn try_validate(&self) -> BunsenResult<()> {
        match *self {
            RelaxationParam::Omega(omega) if !(0.0..=2.0).contains(&omega) => Err(
                ConstraintError::out_of_range("RelaxationParam", "omega", omega, "[0, 2]").into(),
            ),
            RelaxationParam::Tau(tau) if !(0.5..).contains(&tau) => Err(
                ConstraintError::out_of_range("RelaxationParam", "tau", tau, "[0.5, +inf)").into(),
            ),
            _ => Ok(()),
        }
    }

    /// Checks that the relaxation is in range, or panics; the panicking twin
    /// of [`try_validate`](Self::try_validate).
    ///
    /// # Panics
    ///
    /// With the [`try_validate`](Self::try_validate) error's message, if the
    /// value is out of range.
    pub fn validate(&self) {
        self.try_validate().ok_or_panic()
    }

    /// Returns the relaxation frequency (1/tau), typically in (0, 2).
    pub fn as_omega_value(&self) -> f64 {
        match self {
            RelaxationParam::Omega(omega) => *omega,
            RelaxationParam::Tau(tau) => 1.0 / *tau,
        }
    }

    /// Returns the relaxation time (1/omega), typically > 0.5.
    pub fn as_tau_value(&self) -> f64 {
        match self {
            RelaxationParam::Omega(omega) => 1.0 / *omega,
            RelaxationParam::Tau(tau) => *tau,
        }
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::{
            DeviceMemoryGuard,
            performance_device,
        },
    };

    #[test]
    fn test_relaxation_param() {
        let relaxation = RelaxationParam::Omega(1.0);
        relaxation.validate();
        assert_eq!(relaxation.as_omega_value(), 1.0);
        assert_eq!(relaxation.as_tau_value(), 1.0 / 1.0);

        let relaxation = RelaxationParam::Tau(0.5);
        relaxation.validate();
        assert_eq!(relaxation.as_omega_value(), 2.0);
        assert_eq!(relaxation.as_tau_value(), 0.5);
    }

    /// An out-of-range value is an `Err` from `try_validate`, not a panic.
    #[test]
    fn test_try_validate_rejects_out_of_range() {
        RelaxationParam::Omega(2.0).try_validate().unwrap();
        RelaxationParam::Tau(0.5).try_validate().unwrap();

        for bad in [
            RelaxationParam::Omega(2.01),
            RelaxationParam::Omega(-0.1),
            RelaxationParam::Omega(f64::NAN),
            RelaxationParam::Tau(0.49),
            RelaxationParam::Tau(f64::NAN),
        ] {
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .has_cause::<ConstraintError>()
                .assert_err(&bad.try_validate());
        }
    }

    #[test]
    #[should_panic(expected = "RelaxationParam.tau: 0.49 is outside [0.5, +inf)")]
    fn test_bad_tau() {
        let relaxation = RelaxationParam::Tau(0.49);
        relaxation.validate();
    }

    #[test]
    #[should_panic(expected = "RelaxationParam.omega: 2.01 is outside [0, 2]")]
    fn test_bad_omega() {
        let relaxation = RelaxationParam::Omega(2.01);
        relaxation.validate();
    }

    #[test]
    #[serial]
    fn test_omega_source_from_relaxation() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let relaxation = RelaxationParam::Omega(1.0);
        let omega_source: OmegaSource = relaxation.into();
        let omega = omega_source.omega(&device, DType::F32);

        omega.to_data().assert_eq(
            &Tensor::<2>::from_data([[relaxation.as_omega_value()]], &device).to_data(),
            false,
        );
    }
}
