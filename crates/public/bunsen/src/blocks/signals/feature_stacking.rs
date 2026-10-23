//! Stacking frames into fewer, wider ones.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    tensor::Device,
};

use crate::{
    burner::{
        module::{
            ModuleInit,
            Rekey,
        },
        tensor::{
            permute_data,
            reshape_data,
        },
    },
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
    },
};

/// How [`FeatureStacking`] lays a group's frames side by side for its
/// projection.
///
/// A layout choice with a loading consequence: the projection's weight
/// columns follow it, so a checkpoint stacked the other way loads through
/// a key map that permutes them ([`FeatureStackingMeta::restack_from`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StackOrder {
    /// Frames outer, features inner (`[f0 | f1 | ...]`).
    #[default]
    FramesFeatures,

    /// Features outer, frames inner.
    FeaturesFrames,
}

/// Common meta for [`FeatureStacking`] and [`FeatureStackingConfig`], and
/// the key-map helper that follows from it.
pub trait FeatureStackingMeta {
    /// The input frame's width.
    fn d_input(&self) -> usize;

    /// The output frame's width.
    fn d_output(&self) -> usize;

    /// The frames stacked into one.
    fn factor(&self) -> usize;

    /// The output length for `len` input frames: `ceil(len / factor)`.
    fn output_len(
        &self,
        len: usize,
    ) -> usize {
        len.div_ceil(self.factor())
    }

    /// How a group's frames are laid side by side.
    fn order(&self) -> StackOrder;

    /// `rekey`, and, when `from`, the order a checkpoint stacks in, is not
    /// this one's, the projection's weight (`projection.weight`) permuted to
    /// match: in `PyTorch`'s `[out, in]`, `[d_output, a * b]` to
    /// `[d_output, b * a]`.
    fn restack_from(
        &self,
        from: StackOrder,
        rekey: Rekey,
    ) -> Rekey {
        if from == self.order() {
            return rekey;
        }
        let (k, d) = (self.factor(), self.d_input());
        let (a, b) = match from {
            StackOrder::FramesFeatures => (k, d),
            StackOrder::FeaturesFrames => (d, k),
        };
        rekey.transform("projection.weight", move |data| {
            let out = data.shape().num_elements() / (a * b);
            let data = reshape_data(data, vec![out, a, b]);
            reshape_data(permute_data(data, &[0, 2, 1]), vec![out, b * a])
        })
    }
}

/// Config for [`FeatureStacking`].
///
/// Implements [`FeatureStackingMeta`].
#[derive(Config, Debug)]
pub struct FeatureStackingConfig {
    /// The input frame's width.
    pub d_input: usize,

    /// The output frame's width.
    pub d_output: usize,

    /// The frames stacked into one.
    #[config(default = "8")]
    pub factor: usize,

    /// Whether the projection carries a bias.
    #[config(default = "false")]
    pub bias: bool,

    /// How a group's frames are laid side by side.
    #[config(default = "StackOrder::FramesFeatures")]
    pub order: StackOrder,
}

impl FeatureStackingMeta for FeatureStackingConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn d_output(&self) -> usize {
        self.d_output
    }

    fn factor(&self) -> usize {
        self.factor
    }

    fn order(&self) -> StackOrder {
        self.order
    }
}

impl ModuleInit<FeatureStacking> for FeatureStackingConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<FeatureStacking> {
        for (field, value) in [
            ("d_input", self.d_input),
            ("d_output", self.d_output),
            ("factor", self.factor),
        ] {
            if value == 0 {
                return Err(ConstraintError::zero_or_empty("FeatureStackingConfig", field).into());
            }
        }
        Ok(FeatureStacking {
            projection: LinearConfig::new(self.factor * self.d_input, self.d_output)
                .with_bias(self.bias)
                .init(device),
            factor: self.factor,
            order: self.order,
        })
    }
}

/// Stacks `factor` consecutive frames into one and projects it: a learned
/// downsampling by `factor`.
///
/// The input is zero-padded at its end to a whole number of groups; group
/// `j` is frames `j * factor .. (j + 1) * factor`, laid side by side
/// (`[f0 | f1 | ...]`), then `[factor * d_input] -> [d_output]`.
///
/// Built by [`FeatureStackingConfig`].
/// Implements [`FeatureStackingMeta`].
#[derive(Module, Debug)]
pub struct FeatureStacking {
    /// `[factor * d_input] -> [d_output]`.
    pub projection: Linear,

    /// The frames stacked into one.
    pub factor: usize,

    /// How a group's frames are laid side by side.
    #[module(skip)]
    pub order: StackOrder,
}

impl FeatureStackingMeta for FeatureStacking {
    fn d_input(&self) -> usize {
        self.projection.weight.dims()[0] / self.factor
    }

    fn d_output(&self) -> usize {
        self.projection.weight.dims()[1]
    }

    fn factor(&self) -> usize {
        self.factor
    }

    fn order(&self) -> StackOrder {
        self.order
    }
}

impl FeatureStacking {
    /// Stacks and projects.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_input]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    ///
    /// # Returns
    /// `[batch, ceil(time / factor), d_output]`, and each item's output
    /// length, `ceil(length / factor)`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        let [batch, time, d_input] = unpack_shape_contract!(
            ["batch", "time", "d_input"],
            &x.dims(),
            &["batch", "time", "d_input"],
            &[("d_input", self.d_input())],
        );
        let groups = self.output_len(time);
        let pad = groups * self.factor - time;
        let x = if pad > 0 {
            Tensor::cat(
                vec![x.clone(), Tensor::zeros([batch, pad, d_input], &x.device())],
                1,
            )
        } else {
            x
        };
        let x = match self.order {
            StackOrder::FramesFeatures => x.reshape([batch, groups, self.factor * d_input]),
            StackOrder::FeaturesFrames => x
                .reshape([batch, groups, self.factor, d_input])
                .swap_dims(2, 3)
                .reshape([batch, groups, d_input * self.factor]),
        };
        let y = self.projection.forward(x);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "groups", "d_output"],
            &y.dims(),
            &[
                ("batch", batch),
                ("groups", groups),
                ("d_output", self.d_output())
            ],
        );

        let lengths = match lengths {
            Some(lengths) => lengths.iter().map(|&n| self.output_len(n)).collect(),
            None => vec![groups; batch],
        };
        (y, lengths)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::TensorData,
        tensor::Tolerance,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = FeatureStackingConfig::new(5, 7)
            .with_factor(3)
            .with_bias(true);
        let module: FeatureStacking = config.init(&cpu_device());
        assert_eq!(module.d_input(), config.d_input());
        assert_eq!(module.d_output(), config.d_output());
        assert_eq!(module.factor(), config.factor());
        assert!(module.projection.bias.is_some());
        assert_eq!(module.output_len(7), 3);
        assert!(
            ModuleInit::<FeatureStacking>::try_init(&config.with_factor(0), &cpu_device()).is_err()
        );
    }

    /// Group `j` is its frames side by side, the last group zero-padded.
    #[test]
    fn test_stacks_frames_side_by_side() {
        let device = cpu_device();
        let mut module: FeatureStacking = FeatureStackingConfig::new(2, 6)
            .with_factor(3)
            .init(&device);
        // An identity projection: the output is the stacked group.
        let eye: Vec<f32> = (0..36)
            .map(|k| if k / 6 == k % 6 { 1.0 } else { 0.0 })
            .collect();
        module.projection.weight = module
            .projection
            .weight
            .map(|_| Tensor::from_data(TensorData::new(eye, [6, 6]), &device));

        let x: Vec<f32> = (0..10).map(|v| v as f32).collect();
        let (y, lengths) = module.forward(
            Tensor::from_data(TensorData::new(x, [1, 5, 2]), &device),
            Some(&[4]),
        );
        assert_eq!(lengths, [2]);
        let want = TensorData::new(
            vec![
                0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0,
            ],
            [1, 2, 6],
        );
        y.into_data()
            .assert_approx_eq::<f32>(&want, Tolerance::absolute(1e-6));
    }
}
