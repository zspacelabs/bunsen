//! # Thermal Relaxation
use burn::{
    Tensor,
    prelude::Backend,
    tensor::DType,
};
use serde::{
    Deserialize,
    Serialize,
};

/// The relaxation operator for [`operations::bgk_collision`].
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
/// - `dist_a`: a ``[H, W, VY=3, VX=3]`` distribution.
/// - `dist_b`: a ``[H, W, VY=3, VX=3]`` distribution.
/// - `relaxation`: relaxation parameter.
/// - `correction`: fused correction factor for the relaxation operator;
///   defaults to 1.0.
///
/// # Returns
///
/// The `[H, W, VY=3, VX=3]` relaxed sum.
pub fn relaxed_sum<B: Backend, S: Into<OmegaSource<B>>>(
    dist_a: Tensor<B, 4>,
    dist_b: Tensor<B, 4>,
    relaxation: S,
    correction: Option<f64>,
) -> Tensor<B, 4> {
    let omega = relaxation
        .into()
        .omega(&dist_a.device(), dist_a.dtype())
        .unsqueeze_dims::<4>(&[2, 3]);

    let correction = correction.unwrap_or(1.0);
    dist_a * (correction * (1.0 - omega.clone())) + dist_b * (correction * omega)
}

/// Omegas for the BGK collision operator.
pub enum OmegaSource<B: Backend> {
    /// Scalar relaxation frequency.
    Relaxation(RelaxationParam),

    /// Tensor relaxation frequency.
    Omega(Tensor<B, 2>),
}

impl<B: Backend> OmegaSource<B> {
    /// Get the omega tensor.
    pub fn omega(
        &self,
        device: &B::Device,
        dtype: DType,
    ) -> Tensor<B, 2> {
        match self {
            OmegaSource::Relaxation(relaxation) => {
                Tensor::<B, 1>::from_data([relaxation.as_omega_value()], (device, dtype))
                    .unsqueeze()
            }
            OmegaSource::Omega(omega) => omega.clone().to_device(device).cast(dtype),
        }
    }
}

impl<B: Backend> From<RelaxationParam> for OmegaSource<B> {
    fn from(val: RelaxationParam) -> Self {
        OmegaSource::Relaxation(val)
    }
}

impl<B: Backend> From<Tensor<B, 2>> for OmegaSource<B> {
    fn from(val: Tensor<B, 2>) -> Self {
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
    /// Validate the relaxation; or panic.
    pub fn validate(&self) {
        match self {
            RelaxationParam::Omega(omega) => {
                assert!(
                    (0.0..=2.0).contains(omega),
                    "omega ({omega}) must be in [0, 2.0] range"
                );
            }
            RelaxationParam::Tau(tau) => {
                assert!(*tau >= 0.5, "tau ({tau}) must be >= 0.5");
            }
        }
    }

    /// Get the relaxation frequency (1/tau), typically in (0, 2)
    pub fn as_omega_value(&self) -> f64 {
        match self {
            RelaxationParam::Omega(omega) => *omega,
            RelaxationParam::Tau(tau) => 1.0 / *tau,
        }
    }

    /// Get the relaxation time (1/omega), typically > 0.5
    pub fn as_tau_value(&self) -> f64 {
        match self {
            RelaxationParam::Omega(omega) => 1.0 / *omega,
            RelaxationParam::Tau(tau) => *tau,
        }
    }
}

#[cfg(test)]
mod tests {
    use bunsen::support::testing::PerfTestBackend;

    use super::*;

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

    #[test]
    #[should_panic(expected = "tau (0.49) must be >= 0.5")]
    fn test_bad_tau() {
        let relaxation = RelaxationParam::Tau(0.49);
        relaxation.validate();
    }

    #[test]
    #[should_panic(expected = "omega (2.01) must be in [0, 2.0] range")]
    fn test_bad_omega() {
        let relaxation = RelaxationParam::Omega(2.01);
        relaxation.validate();
    }

    #[test]
    fn test_omega_source_from_relaxation() {
        type B = PerfTestBackend;
        let device = Default::default();

        let relaxation = RelaxationParam::Omega(1.0);
        let omega_source: OmegaSource<B> = relaxation.into();
        let omega = omega_source.omega(&device, DType::F32);

        omega.to_data().assert_eq(
            &Tensor::<B, 2>::from_data([[relaxation.as_omega_value()]], &device).to_data(),
            false,
        );
    }
}
