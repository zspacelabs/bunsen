//! The activation seam's enum and its config.

use alloc::vec::Vec;

use burn::{
    nn::{
        Identity,
        activation::{
            Celu,
            CeluConfig,
            Elu,
            EluConfig,
            Gelu,
            HardShrink,
            HardShrinkConfig,
            HardSigmoid,
            HardSigmoidConfig,
            HardSwish,
            LeakyRelu,
            LeakyReluConfig,
            PRelu,
            PReluConfig,
            Relu,
            Selu,
            Shrink,
            ShrinkConfig,
            SiLU,
            Sigmoid,
            SoftShrink,
            SoftShrinkConfig,
            Softplus,
            SoftplusConfig,
            Softsign,
            Tanh,
            ThresholdedRelu,
            ThresholdedReluConfig,
        },
    },
    prelude::*,
};

use crate::blocks::math::{
    PowOpModule,
    PowOpModuleConfig,
};

/// An [`BunsenActivation`]'s config: its kind, and that kind's config.
#[derive(Config, Debug)]
pub enum BunsenActivationConfig {
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

    /// burn's [`Selu`].
    Selu,

    /// burn's [`HardSigmoid`]: `clamp(alpha · x + beta, 0, 1)`.
    HardSigmoid(HardSigmoidConfig),

    /// burn's [`HardSwish`]: `x · hard_sigmoid(x)`, at `alpha = 1/6`,
    /// `beta = 1/2`.
    HardSwish,

    /// burn's [`Softplus`]: `log(1 + exp(beta · x)) / beta`, and `x` where
    /// `beta · x` is over its threshold.
    Softplus(SoftplusConfig),

    /// burn's [`Softsign`]: `x / (1 + |x|)`.
    Softsign,

    /// burn's [`Elu`].
    Elu(EluConfig),

    /// burn's [`Celu`].
    Celu(CeluConfig),

    /// burn's [`ThresholdedRelu`]: `x` above `alpha`, else 0.
    ThresholdedRelu(ThresholdedReluConfig),

    /// burn's [`HardShrink`]: `x` outside `[-lambda, lambda]`, else 0.
    HardShrink(HardShrinkConfig),

    /// burn's [`SoftShrink`]: `x` moved toward 0 by `lambda`, and 0 inside
    /// `[-lambda, lambda]`.
    SoftShrink(SoftShrinkConfig),

    /// burn's [`Shrink`]: [`SoftShrink`](Self::SoftShrink) with its own
    /// `bias` for the move.
    Shrink(ShrinkConfig),

    /// `silu(x) = x · sigmoid(x)`: burn's [`SiLU`].
    Silu,

    /// `x^p`, for the exponent `p`.
    Pow(f64),

    /// Each kind in turn, the first applied first; empty, the identity.
    Chain(Vec<BunsenActivationConfig>),
}

impl BunsenActivationConfig {
    /// The activation, on `device` (where a kind has parameters).
    pub fn init(
        &self,
        device: &Device,
    ) -> BunsenActivation {
        match self {
            Self::Identity => BunsenActivation::Identity(Identity::new()),
            Self::Relu => BunsenActivation::Relu(Relu::new()),
            Self::Gelu => BunsenActivation::Gelu(Gelu::new()),
            Self::GeluApproximate => BunsenActivation::Gelu(Gelu::new_approximate()),
            Self::Sigmoid => BunsenActivation::Sigmoid(Sigmoid::new()),
            Self::Tanh => BunsenActivation::Tanh(Tanh::new()),
            Self::PRelu(config) => BunsenActivation::PRelu(config.init(device)),
            Self::LeakyRelu(config) => BunsenActivation::LeakyRelu(config.init()),
            Self::Selu => BunsenActivation::Selu(Selu::new()),
            Self::HardSigmoid(config) => BunsenActivation::HardSigmoid(config.init()),
            Self::HardSwish => BunsenActivation::HardSwish(HardSwish::new()),
            Self::Softplus(config) => BunsenActivation::Softplus(config.init()),
            Self::Softsign => BunsenActivation::Softsign(Softsign::new()),
            Self::Elu(config) => BunsenActivation::Elu(config.init()),
            Self::Celu(config) => BunsenActivation::Celu(config.init()),
            Self::ThresholdedRelu(config) => BunsenActivation::ThresholdedRelu(config.init()),
            Self::HardShrink(config) => BunsenActivation::HardShrink(config.init()),
            Self::SoftShrink(config) => BunsenActivation::SoftShrink(config.init()),
            Self::Shrink(config) => BunsenActivation::Shrink(config.init()),
            Self::Silu => BunsenActivation::Silu(SiLU::new()),
            Self::Pow(exponent) => BunsenActivation::Pow(PowOpModuleConfig::new(*exponent).init()),
            Self::Chain(kinds) => {
                BunsenActivation::Chain(kinds.iter().map(|kind| kind.init(device)).collect())
            }
        }
    }
}

/// An activation layer: burn's, or bunsen's, behind bunsen's seam (the
/// [module docs](super)).
#[derive(Module, Debug)]
pub enum BunsenActivation {
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

    /// burn's [`Selu`].
    Selu(Selu),

    /// burn's [`HardSigmoid`].
    HardSigmoid(HardSigmoid),

    /// burn's [`HardSwish`].
    HardSwish(HardSwish),

    /// burn's [`Softplus`].
    Softplus(Softplus),

    /// burn's [`Softsign`].
    Softsign(Softsign),

    /// burn's [`Elu`].
    Elu(Elu),

    /// burn's [`Celu`].
    Celu(Celu),

    /// burn's [`ThresholdedRelu`].
    ThresholdedRelu(ThresholdedRelu),

    /// burn's [`HardShrink`].
    HardShrink(HardShrink),

    /// burn's [`SoftShrink`].
    SoftShrink(SoftShrink),

    /// burn's [`Shrink`].
    Shrink(Shrink),

    /// burn's [`SiLU`].
    Silu(SiLU),

    /// `x^p`: [`PowOpModule`].
    Pow(PowOpModule),

    /// Each in turn, the first applied first.
    Chain(Vec<BunsenActivation>),
}

impl BunsenActivation {
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
            Self::Selu(layer) => layer.forward(input),
            Self::HardSigmoid(layer) => layer.forward(input),
            Self::HardSwish(layer) => layer.forward(input),
            Self::Softplus(layer) => layer.forward(input),
            Self::Softsign(layer) => layer.forward(input),
            Self::Elu(layer) => layer.forward(input),
            Self::Celu(layer) => layer.forward(input),
            Self::ThresholdedRelu(layer) => layer.forward(input),
            Self::HardShrink(layer) => layer.forward(input),
            Self::SoftShrink(layer) => layer.forward(input),
            Self::Shrink(layer) => layer.forward(input),
            Self::Silu(layer) => layer.forward(input),
            Self::Pow(op) => op.forward(input),
            Self::Chain(layers) => layers.iter().fold(input, |x, layer| layer.forward(x)),
        }
    }
}

impl<const D: usize> crate::burner::module::Forward1Module<Tensor<D>> for BunsenActivation {
    type Output = Tensor<D>;

    /// [`BunsenActivation::forward`].
    fn forward(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        BunsenActivation::forward(self, input)
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
        config: BunsenActivationConfig,
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
            (BunsenActivationConfig::Identity, X),
            (BunsenActivationConfig::Relu, [0.0, 0.0, 0.0, 1.0, 3.0]),
            (
                BunsenActivationConfig::Gelu,
                [-0.04550026, -0.1542688, 0.0, 0.8413447, 2.99595],
            ),
            (
                BunsenActivationConfig::GeluApproximate,
                [-0.04540231, -0.154286, 0.0, 0.841192, 2.996363],
            ),
            (
                BunsenActivationConfig::Silu,
                [-0.2384058, -0.1887703, 0.0, 0.7310586, 2.857722],
            ),
            (BunsenActivationConfig::Pow(2.0), [4.0, 0.25, 0.0, 1.0, 9.0]),
            (
                BunsenActivationConfig::Chain(vec![
                    BunsenActivationConfig::Relu,
                    BunsenActivationConfig::Pow(2.0),
                ]),
                [0.0, 0.0, 0.0, 1.0, 9.0],
            ),
            (
                // Order matters: squared, then rectified.
                BunsenActivationConfig::Chain(vec![
                    BunsenActivationConfig::Pow(2.0),
                    BunsenActivationConfig::Relu,
                ]),
                [4.0, 0.25, 0.0, 1.0, 9.0],
            ),
            (BunsenActivationConfig::Chain(vec![]), X),
        ] {
            apply(config.clone(), X)
                .assert_approx_eq::<f32>(&TensorData::from(expected), tolerance);
        }
    }

    /// A config's serde form names burn's kinds as burn does, and bunsen's
    /// by their own names.
    #[test]
    fn test_serde_names() {
        let json = serde_json::to_string(&BunsenActivationConfig::Chain(vec![
            BunsenActivationConfig::Relu,
            BunsenActivationConfig::GeluApproximate,
            BunsenActivationConfig::Pow(2.0),
        ]))
        .expect("serialize");
        assert_eq!(json, r#"{"Chain":["Relu","GeluApproximate",{"Pow":2.0}]}"#);
        let burn_json =
            serde_json::to_string(&burn::nn::activation::ActivationConfig::GeluApproximate)
                .expect("serialize");
        assert_eq!(burn_json, r#""GeluApproximate""#);
    }

    /// Each of burn's weightless elementwise kinds (every
    /// [`ActivationConfig`](burn::nn::activation::ActivationConfig) kind but
    /// `SwiGlu`, which owns two `Linear`s) reads from burn's serde form, writes
    /// it back unchanged, and is burn's layer: the same values as burn's
    /// `Activation`, alone and first in a
    /// [`Chain`](BunsenActivationConfig::Chain).
    #[test]
    fn test_burn_kinds_are_burn_s() {
        use burn::nn::activation::ActivationConfig;

        let device = cpu_device();
        let x = Tensor::<2>::from_floats(
            [[-3.0, -1.5, -0.75, -0.25, 0.0], [0.25, 0.6, 1.0, 2.5, 25.0]],
            &device,
        );
        for burn_config in [
            ActivationConfig::Identity,
            ActivationConfig::Relu,
            ActivationConfig::Gelu,
            ActivationConfig::GeluApproximate,
            ActivationConfig::Sigmoid,
            ActivationConfig::Tanh,
            ActivationConfig::PRelu(PReluConfig::new().with_alpha(0.1)),
            ActivationConfig::LeakyRelu(LeakyReluConfig::new().with_negative_slope(0.2)),
            ActivationConfig::Selu,
            ActivationConfig::HardSigmoid(HardSigmoidConfig::new().with_alpha(0.3).with_beta(0.4)),
            ActivationConfig::HardSwish,
            ActivationConfig::Softplus(SoftplusConfig::new().with_beta(2.0).with_threshold(10.0)),
            ActivationConfig::Softsign,
            ActivationConfig::Elu(EluConfig::new().with_alpha(0.7)),
            ActivationConfig::Celu(CeluConfig::new().with_alpha(1.3)),
            ActivationConfig::ThresholdedRelu(ThresholdedReluConfig::new().with_alpha(0.5)),
            ActivationConfig::HardShrink(HardShrinkConfig::new().with_lambda(0.3)),
            ActivationConfig::SoftShrink(SoftShrinkConfig::new().with_lambda(0.3)),
            ActivationConfig::Shrink(ShrinkConfig::new().with_lambda(0.3).with_bias(0.1)),
        ] {
            let json = serde_json::to_string(&burn_config).expect("serialize burn's");
            let config: BunsenActivationConfig =
                serde_json::from_str(&json).unwrap_or_else(|e| panic!("{json}: {e}"));
            assert_eq!(serde_json::to_string(&config).expect("serialize"), json);

            let expected = burn_config.init(&device).forward(x.clone());
            config
                .init(&device)
                .forward(x.clone())
                .into_data()
                .assert_eq(&expected.clone().into_data(), true);

            let chained = BunsenActivationConfig::Chain(vec![config, BunsenActivationConfig::Tanh]);
            chained
                .init(&device)
                .forward(x.clone())
                .into_data()
                .assert_eq(&expected.tanh().into_data(), true);
        }
    }
}
