//! [`PowActivation`]: `x^p` as a layer.

use burn::{
    config::Config,
    module::Module,
    prelude::{
        Device,
        Tensor,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`PowActivation`] and [`PowActivationConfig`].
pub trait PowActivationMeta {
    /// The exponent `p`.
    fn exponent(&self) -> f64;
}

/// Config for [`PowActivation`].
#[derive(Config, Debug)]
pub struct PowActivationConfig {
    /// The exponent `p`.
    pub exponent: f64,
}

impl PowActivationMeta for PowActivationConfig {
    fn exponent(&self) -> f64 {
        self.exponent
    }
}

/// The module; it has no parameters, so `device` is unused.
impl ModuleInit<PowActivation> for PowActivationConfig {
    fn try_init(
        &self,
        _device: &Device,
    ) -> BunsenResult<PowActivation> {
        Ok(PowActivation {
            exponent: self.exponent,
        })
    }
}

/// `x^p`, elementwise: a layer with no parameters, the activation seam's
/// [`Pow`](super::BunsenActivationConfig::Pow) kind.
///
/// Built by [`PowActivationConfig`].
#[derive(Module, Debug)]
pub struct PowActivation {
    /// The exponent `p`.
    pub exponent: f64,
}

impl PowActivationMeta for PowActivation {
    fn exponent(&self) -> f64 {
        self.exponent
    }
}

impl PowActivation {
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
        let config = PowActivationConfig::new(3.0);
        let module = config.init(&cpu_device());
        assert_eq!(config.exponent(), 3.0);
        assert_eq!(module.exponent(), config.exponent());
    }

    /// `x^p` against values computed by hand.
    #[test]
    fn test_forward() {
        let device = cpu_device();
        let x = Tensor::<1>::from_floats([-2.0, -0.5, 0.0, 1.0, 3.0], &device);
        PowActivationConfig::new(2.0)
            .init(&device)
            .forward(x)
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([4.0f32, 0.25, 0.0, 1.0, 9.0]),
                Tolerance::absolute(1e-6),
            );
    }
}
