//! The activation seam's enum and its config.

use burn::{
    nn::{
        Identity,
        activation::{
            Gelu,
            LeakyRelu,
            LeakyReluConfig,
            PRelu,
            PReluConfig,
            Relu,
            SiLU,
            Sigmoid,
            Tanh,
        },
    },
    prelude::*,
};

/// An [`Activation`]'s config: its kind, and that kind's config.
#[derive(Config, Debug)]
pub enum ActivationConfig {
    /// No activation: the input as it is.
    Identity,

    /// burn's [`Relu`].
    Relu,

    /// burn's exact [`Gelu`].
    Gelu,

    /// burn's [`Gelu`] in its tanh approximation (`transformers`'
    /// `gelu_pytorch_tanh`).
    GeluApproximate,

    /// burn's [`Sigmoid`].
    Sigmoid,

    /// burn's [`Tanh`].
    Tanh,

    /// burn's [`PRelu`], with a learned slope.
    PRelu(PReluConfig),

    /// burn's [`LeakyRelu`].
    LeakyRelu(LeakyReluConfig),

    /// `silu(x) = x · sigmoid(x)`: burn's [`SiLU`].
    Silu,

    /// `x^p`, for the exponent `p`.
    Pow(f64),

    /// Each kind in turn, the first applied first; empty, the identity.
    Chain(Vec<ActivationConfig>),
}

impl ActivationConfig {
    /// The activation, on `device` (where a kind has parameters).
    pub fn init(
        &self,
        device: &Device,
    ) -> Activation {
        match self {
            Self::Identity => Activation::Identity(Identity::new()),
            Self::Relu => Activation::Relu(Relu::new()),
            Self::Gelu => Activation::Gelu(Gelu::new()),
            Self::GeluApproximate => Activation::Gelu(Gelu::new_approximate()),
            Self::Sigmoid => Activation::Sigmoid(Sigmoid::new()),
            Self::Tanh => Activation::Tanh(Tanh::new()),
            Self::PRelu(config) => Activation::PRelu(config.init(device)),
            Self::LeakyRelu(config) => Activation::LeakyRelu(config.init()),
            Self::Silu => Activation::Silu(SiLU::new()),
            Self::Pow(exponent) => Activation::Pow(Pow {
                exponent: *exponent,
            }),
            Self::Chain(kinds) => {
                Activation::Chain(kinds.iter().map(|kind| kind.init(device)).collect())
            }
        }
    }
}

/// `x^p`: an [`Activation`] kind of bunsen's.
#[derive(Module, Debug)]
pub struct Pow {
    /// The exponent `p`.
    pub exponent: f64,
}

/// An activation layer: burn's, or bunsen's, behind bunsen's seam (the
/// [module docs](super)).
#[derive(Module, Debug)]
pub enum Activation {
    /// No activation.
    Identity(Identity),

    /// burn's [`Relu`].
    Relu(Relu),

    /// burn's [`Gelu`], exact or tanh-approximated.
    Gelu(Gelu),

    /// burn's [`Sigmoid`].
    Sigmoid(Sigmoid),

    /// burn's [`Tanh`].
    Tanh(Tanh),

    /// burn's [`PRelu`].
    PRelu(PRelu),

    /// burn's [`LeakyRelu`].
    LeakyRelu(LeakyRelu),

    /// burn's [`SiLU`].
    Silu(SiLU),

    /// `x^p`.
    Pow(Pow),

    /// Each in turn, the first applied first.
    Chain(Vec<Activation>),
}

impl Activation {
    /// `input`, activated elementwise; in `input`'s dtype.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        match self {
            Self::Identity(layer) => layer.forward(input),
            Self::Relu(layer) => layer.forward(input),
            Self::Gelu(layer) => layer.forward(input),
            Self::Sigmoid(layer) => layer.forward(input),
            Self::Tanh(layer) => layer.forward(input),
            Self::PRelu(layer) => layer.forward(input),
            Self::LeakyRelu(layer) => layer.forward(input),
            Self::Silu(layer) => layer.forward(input),
            Self::Pow(pow) => input.powf_scalar(pow.exponent),
            Self::Chain(layers) => layers.iter().fold(input, |x, layer| layer.forward(x)),
        }
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

    fn apply(
        config: ActivationConfig,
        values: [f32; 5],
    ) -> TensorData {
        let device = cpu_device();
        config
            .init(&device)
            .forward(Tensor::<1>::from_floats(values, &device))
            .into_data()
    }

    const X: [f32; 5] = [-2.0, -0.5, 0.0, 1.0, 3.0];

    /// Each kind against values computed in float64.
    #[test]
    fn test_kinds() {
        let tolerance = Tolerance::rel_abs(1e-5, 1e-6);
        for (config, expected) in [
            (ActivationConfig::Identity, X),
            (ActivationConfig::Relu, [0.0, 0.0, 0.0, 1.0, 3.0]),
            (
                ActivationConfig::Gelu,
                [-0.04550026, -0.1542688, 0.0, 0.8413447, 2.99595],
            ),
            (
                ActivationConfig::GeluApproximate,
                [-0.04540231, -0.154286, 0.0, 0.841192, 2.996363],
            ),
            (
                ActivationConfig::Silu,
                [-0.2384058, -0.1887703, 0.0, 0.7310586, 2.857722],
            ),
            (ActivationConfig::Pow(2.0), [4.0, 0.25, 0.0, 1.0, 9.0]),
            (
                ActivationConfig::Chain(vec![ActivationConfig::Relu, ActivationConfig::Pow(2.0)]),
                [0.0, 0.0, 0.0, 1.0, 9.0],
            ),
            (
                // Order matters: squared, then rectified.
                ActivationConfig::Chain(vec![ActivationConfig::Pow(2.0), ActivationConfig::Relu]),
                [4.0, 0.25, 0.0, 1.0, 9.0],
            ),
            (ActivationConfig::Chain(vec![]), X),
        ] {
            apply(config.clone(), X)
                .assert_approx_eq::<f32>(&TensorData::from(expected), tolerance);
        }
    }

    /// A config's serde form names burn's kinds as burn does, and bunsen's
    /// by their own names.
    #[test]
    fn test_serde_names() {
        let json = serde_json::to_string(&ActivationConfig::Chain(vec![
            ActivationConfig::Relu,
            ActivationConfig::GeluApproximate,
            ActivationConfig::Pow(2.0),
        ]))
        .expect("serialize");
        assert_eq!(json, r#"{"Chain":["Relu","GeluApproximate",{"Pow":2.0}]}"#);
        let burn_json =
            serde_json::to_string(&burn::nn::activation::ActivationConfig::GeluApproximate)
                .expect("serialize");
        assert_eq!(burn_json, r#""GeluApproximate""#);
    }
}
