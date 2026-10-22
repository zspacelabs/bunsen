//! Sub-pixel convolution: upsampling a sequence by a whole factor.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
    },
};

/// Common meta for [`SubpixelUpsample1d`] and [`SubpixelUpsample1dConfig`].
pub trait SubpixelUpsample1dMeta {
    /// The frame width, in and out.
    fn d_model(&self) -> usize;

    /// The frames each input frame becomes.
    fn factor(&self) -> usize;
}

/// Config for [`SubpixelUpsample1d`].
///
/// Implements [`SubpixelUpsample1dMeta`].
#[derive(Config, Debug)]
pub struct SubpixelUpsample1dConfig {
    /// The frame width, in and out.
    pub d_model: usize,

    /// The frames each input frame becomes.
    #[config(default = "8")]
    pub factor: usize,

    /// The convolution's kernel size; odd, padded to keep the length.
    #[config(default = "3")]
    pub kernel_size: usize,
}

impl SubpixelUpsample1dMeta for SubpixelUpsample1dConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn factor(&self) -> usize {
        self.factor
    }
}

impl ModuleInit<SubpixelUpsample1d> for SubpixelUpsample1dConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SubpixelUpsample1d> {
        for (field, value) in [("d_model", self.d_model), ("factor", self.factor)] {
            if value == 0 {
                return Err(
                    ConstraintError::zero_or_empty("SubpixelUpsample1dConfig", field).into(),
                );
            }
        }
        if self.kernel_size.is_multiple_of(2) {
            return Err(crate::errors::BunsenError::illegal(format!(
                "SubpixelUpsample1dConfig.kernel_size: {} is even; an odd kernel keeps the length",
                self.kernel_size
            )));
        }
        let pad = self.kernel_size / 2;
        Ok(SubpixelUpsample1d {
            conv: Conv1dConfig::new(self.d_model, self.d_model * self.factor, self.kernel_size)
                .with_padding(PaddingConfig1d::Explicit(pad, pad))
                .init(device),
            factor: self.factor,
        })
    }
}

/// Upsamples a sequence by `factor`: a same-length convolution to
/// `factor * d_model` channels, then each frame's channels split into
/// `factor` consecutive frames, sub-frame `j` taking channels
/// `j * d_model..(j + 1) * d_model`.
///
/// Built by [`SubpixelUpsample1dConfig`].
/// Implements [`SubpixelUpsample1dMeta`].
#[derive(Module, Debug)]
pub struct SubpixelUpsample1d {
    /// `[d_model] -> [factor * d_model]` channels, length kept.
    pub conv: Conv1d,

    /// The frames each input frame becomes.
    pub factor: usize,
}

impl SubpixelUpsample1dMeta for SubpixelUpsample1d {
    fn d_model(&self) -> usize {
        self.conv.weight.dims()[1]
    }

    fn factor(&self) -> usize {
        self.factor
    }
}

impl SubpixelUpsample1d {
    /// Upsamples.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    ///
    /// # Returns
    /// `[batch, time * factor, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [batch, time, d_model] = unpack_shape_contract!(
            ["batch", "time", "d_model"],
            &x.dims(),
            &["batch", "time", "d_model"],
            &[("d_model", self.d_model())],
        );
        let y = self
            .conv
            .forward(x.swap_dims(1, 2))
            .swap_dims(1, 2)
            .reshape([batch, time * self.factor, d_model]);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time" * "factor", "d_model"],
            &y.dims(),
            &[
                ("batch", batch),
                ("time", time),
                ("factor", self.factor),
                ("d_model", d_model)
            ],
        );
        y
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
        let config = SubpixelUpsample1dConfig::new(4)
            .with_factor(3)
            .with_kernel_size(5);
        let module: SubpixelUpsample1d = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.factor(), config.factor());
        assert_eq!(module.conv.weight.dims(), [12, 4, 5]);
        assert!(
            ModuleInit::<SubpixelUpsample1d>::try_init(&config.with_kernel_size(4), &cpu_device())
                .is_err()
        );
    }

    /// With its centre tap a repeated identity and the rest zero, the
    /// upsample repeats each frame `factor` times: nearest-neighbour.
    #[test]
    fn test_identity_taps_repeat_frames() {
        let device = cpu_device();
        let (d, r) = (2, 3);
        let mut module: SubpixelUpsample1d = SubpixelUpsample1dConfig::new(d)
            .with_factor(r)
            .init(&device);
        let mut w = vec![0.0f32; d * r * d * 3];
        for o in 0..d * r {
            // weight[o, o % d, 1] = 1
            w[(o * d + o % d) * 3 + 1] = 1.0;
        }
        module.conv.weight = module
            .conv
            .weight
            .map(|_| Tensor::from_data(TensorData::new(w, [d * r, d, 3]), &device));
        module.conv.bias = module.conv.bias.map(|b| b.map(|t| t.zeros_like()));

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y = module.forward(Tensor::from_data(TensorData::new(x, [1, 2, d]), &device));
        let want = TensorData::new(
            vec![
                1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0,
            ],
            [1, 6, d],
        );
        y.into_data()
            .assert_approx_eq::<f32>(&want, Tolerance::absolute(1e-6));
    }
}
