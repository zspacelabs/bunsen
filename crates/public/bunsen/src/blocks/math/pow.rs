//! [`PowOpModule`]: `x^p` as a module.

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
};

/// Common meta for [`PowOpModule`] and [`PowOpModuleConfig`].
pub trait PowOpModuleMeta {
    /// The exponent `p`.
    fn exponent(&self) -> f64;
}

/// Config for [`PowOpModule`].
#[derive(Config, Debug)]
pub struct PowOpModuleConfig {
    /// The exponent `p`.
    pub exponent: f64,
}

impl PowOpModuleMeta for PowOpModuleConfig {
    fn exponent(&self) -> f64 {
        self.exponent
    }
}

impl PowOpModuleConfig {
    /// The module; it has no parameters.
    #[must_use]
    pub fn init(&self) -> PowOpModule {
        PowOpModule {
            exponent: self.exponent,
        }
    }
}

/// `x^p`, elementwise: a module with no parameters, which a module tree holds
/// as a part (the activation seam's `Pow` kind).
///
/// Built by [`PowOpModuleConfig`].
#[derive(Module, Debug)]
pub struct PowOpModule {
    /// The exponent `p`.
    pub exponent: f64,
}

impl PowOpModuleMeta for PowOpModule {
    fn exponent(&self) -> f64 {
        self.exponent
    }
}

impl PowOpModule {
    /// `input^p`, elementwise; in `input`'s dtype.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        input.powf_scalar(self.exponent)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        TensorData,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    /// The config and the module built from it answer the meta alike.
    #[test]
    fn test_meta() {
        let config = PowOpModuleConfig::new(3.0);
        let module = config.init();
        assert_eq!(config.exponent(), 3.0);
        assert_eq!(module.exponent(), config.exponent());
    }

    /// `x^p` against values computed by hand.
    #[test]
    fn test_forward() {
        let device = cpu_device();
        let x = Tensor::<1>::from_floats([-2.0, -0.5, 0.0, 1.0, 3.0], &device);
        PowOpModuleConfig::new(2.0)
            .init()
            .forward(x)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([4.0f32, 0.25, 0.0, 1.0, 9.0]),
                Tolerance::absolute(1e-6),
            );
    }
}
