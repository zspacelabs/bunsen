//! `EvoNorm-S0`: a normalization with its own activation.

use burn::{
    config::Config,
    module::{
        Module,
        Param,
    },
    prelude::Tensor,
    tensor::{
        Device,
        activation::sigmoid,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// The config of an [`EvoNormS0`].
#[derive(Config, Debug)]
pub struct EvoNormS0Config {
    /// The channels it normalizes.
    pub num_features: usize,

    /// The groups the channels split into for the deviation.
    #[config(default = 32)]
    pub groups: usize,

    /// Added to each group's variance.
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl ModuleInit<EvoNormS0> for EvoNormS0Config {
    /// The norm: its gain and its sigmoid's scale ones, its shift zeros
    /// (timm's init).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for channels
    /// the groups don't split.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<EvoNormS0> {
        let c = self.num_features;
        if self.groups == 0 {
            return Err(ConstraintError::zero_or_empty("EvoNormS0Config", "groups").into());
        }
        if !c.is_multiple_of(self.groups) {
            return Err(ConstraintError::new(
                "EvoNormS0Config",
                "num_features",
                Rule::NotMultiple {
                    value: c,
                    of: alloc::format!("groups ({})", self.groups),
                },
            )
            .into());
        }
        Ok(EvoNormS0 {
            weight: Param::from_tensor(Tensor::ones([c], device)),
            bias: Param::from_tensor(Tensor::zeros([c], device)),
            v: Param::from_tensor(Tensor::ones([c], device)),
            groups: self.groups,
            epsilon: self.epsilon,
        })
    }
}

/// `EvoNorm-S0` ([arXiv:2004.02967](https://arxiv.org/abs/2004.02967)),
/// timm's `EvoNorm2dS0`: `x * sigmoid(v * x) / group_std(x) * weight +
/// bias`, the deviation each group's of `channels / groups` channels over
/// every position (the variance biased, `epsilon` added under the root).
/// Its activation is its own: a net puts no activation after it.
///
/// Built by [`EvoNormS0Config`].
#[derive(Module, Debug)]
pub struct EvoNormS0 {
    /// The gain, `[channels]`.
    pub weight: Param<Tensor<1>>,

    /// The shift, `[channels]`.
    pub bias: Param<Tensor<1>>,

    /// The sigmoid's scale, `[channels]`.
    pub v: Param<Tensor<1>>,

    /// The groups.
    pub groups: usize,

    /// Added to each group's variance.
    pub epsilon: f64,
}

impl EvoNormS0 {
    /// The channels it normalizes.
    pub fn num_features(&self) -> usize {
        self.weight.dims()[0]
    }

    /// `input`, normalized.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, channels, ...]`.
    ///
    /// # Returns
    ///
    /// The same shape.
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
    ) -> Tensor<D> {
        let dims = input.dims();
        let (batch, channels) = (dims[0], dims[1]);
        let [g] = crate::contracts::unpack_shape_contract!(
            ["batch", "g" * "per_group", ...],
            &dims,
            &["g"],
            &[("batch", batch), ("g", self.groups)]
        );
        let per_channel = |p: &Param<Tensor<1>>| {
            let mut shape = [1usize; D];
            shape[1] = channels;
            p.val().reshape(shape)
        };
        // Each group's deviation, over its channels and every position.
        let grouped = input.clone().reshape([batch as i32, g as i32, -1]);
        let mean = grouped.clone().mean_dim(2);
        let var = (grouped.clone() - mean).square().mean_dim(2);
        let std = (var + self.epsilon)
            .sqrt()
            .expand(grouped.dims())
            .reshape(dims);
        let x = input.clone() * sigmoid(input * per_channel(&self.v)) / std;
        x * per_channel(&self.weight) + per_channel(&self.bias)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    /// At init (unit gain and scale, zero shift), each value is `x *
    /// sigmoid(x)` over its group's deviation; one group per channel here.
    #[test]
    fn test_evo_norm_s0_at_init() {
        let device = cpu_device();
        let norm: EvoNormS0 = EvoNormS0Config::new(4)
            .with_groups(4)
            .with_epsilon(0.0)
            .init(&device);
        let x = Tensor::<4>::random([2, 4, 3, 3], Distribution::Default, &device);
        let flat = x.clone().reshape([2, 4, 9]);
        let mean = flat.clone().mean_dim(2);
        let std = (flat - mean)
            .square()
            .mean_dim(2)
            .sqrt()
            .reshape([2, 4, 1, 1]);
        let want = x.clone() * sigmoid(x.clone()) / std;
        norm.forward(x)
            .into_data()
            .assert_approx_eq::<f32>(&want.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
    }
}
