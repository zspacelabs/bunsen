//! # Conditioning
//!
//! Layers that let a vector outside the sequence steer a transformer layer.
//!
//! [`AdaptiveScale`] maps a conditioning vector to a per-channel scale,
//! `1 + f(cond)`, that multiplies a sublayer's normalized input: the adaptive
//! normalization of diffusion transformers (adaLN), and how Voxtral Realtime
//! tells its decoder the transcription delay. A
//! [`CausalDecoderBlock`](super::decoders::causal::CausalDecoderBlock) with an
//! [`mlp_scale`](super::decoders::causal::CausalDecoderBlockConfig::mlp_scale)
//! applies one to its feed-forward's input.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        mlp::{
            BunsenMlp,
            BunsenMlpConfig,
            BunsenMlpMeta,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`AdaptiveScale`] and [`AdaptiveScaleConfig`].
pub trait AdaptiveScaleMeta {
    /// The width of the conditioning vector, and of the scale.
    fn d_model(&self) -> usize;

    /// The bottleneck width.
    fn d_hidden(&self) -> usize;
}

/// Config for [`AdaptiveScale`].
///
/// Implements [`AdaptiveScaleMeta`].
#[derive(Config, Debug)]
pub struct AdaptiveScaleConfig {
    /// The width of the conditioning vector, and of the scale.
    pub d_model: usize,

    /// The bottleneck width.
    pub d_hidden: usize,

    /// The activation between the two projections.
    #[config(default = "BunsenActivationConfig::Gelu")]
    pub activation: BunsenActivationConfig,

    /// Whether the two projections carry a bias.
    #[config(default = "false")]
    pub bias: bool,
}

impl AdaptiveScaleMeta for AdaptiveScaleConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn d_hidden(&self) -> usize {
        self.d_hidden
    }
}

impl ModuleInit<AdaptiveScale> for AdaptiveScaleConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<AdaptiveScale> {
        Ok(AdaptiveScale {
            mlp: BunsenMlpConfig::new(self.d_model, self.d_hidden, self.d_model)
                .with_activation(self.activation.clone())
                .with_bias(self.bias)
                .try_init(device)?,
        })
    }
}

/// A per-channel scale learned from a conditioning vector:
/// `1 + mlp(cond)`, a [`BunsenMlp`] `d_model -> d_hidden -> d_model`, which
/// multiplies a normalized activation (adaptive normalization).
///
/// Built by [`AdaptiveScaleConfig`].
/// Implements [`AdaptiveScaleMeta`].
#[derive(Module, Debug)]
pub struct AdaptiveScale {
    /// The scale's MLP, `d_model -> d_hidden -> d_model`.
    pub mlp: BunsenMlp,
}

impl crate::burner::module::InputDTypes for AdaptiveScale {
    /// Its projections' dtype.
    fn preferred_dtype(&self) -> burn::tensor::FloatDType {
        crate::burner::module::InputDTypes::preferred_dtype(&self.mlp)
    }
}

/// [`scale`](AdaptiveScale::scale) casts its condition, which a timestep
/// embedding makes in the device's default float.
impl crate::burner::module::CoercesInput for AdaptiveScale {}

impl crate::burner::module::ResultDType for AdaptiveScale {
    /// Its own: the scale is in its projections' dtype.
    fn result_dtype(
        &self,
        _input: burn::tensor::FloatDType,
    ) -> burn::tensor::FloatDType {
        crate::burner::module::InputDTypes::preferred_dtype(self)
    }
}

impl AdaptiveScaleMeta for AdaptiveScale {
    fn d_model(&self) -> usize {
        self.mlp.d_input()
    }

    fn d_hidden(&self) -> usize {
        self.mlp.d_hidden()
    }
}

impl AdaptiveScale {
    /// The scale, `1 + f(cond)`.
    ///
    /// # Arguments
    /// * `cond` - `[batch, seq, d_model]`; a `[1, 1, d_model]` condition
    ///   broadcasts over whatever it scales.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`.
    pub fn scale(
        &self,
        cond: Tensor<3>,
    ) -> Tensor<3> {
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &cond.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())]
        );
        let cond = crate::burner::module::InputDTypes::coerce_input(self, cond);
        let y = self.mlp.forward(cond).add_scalar(1.0);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("d_model", self.d_model())]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (batch, seq);
        y
    }

    /// `x` scaled by the condition: `x * (1 + f(cond))`.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq, d_model]`.
    /// * `cond` - `[batch, seq, d_model]`, or anything that broadcasts to it.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        cond: Tensor<3>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &x.dims(),
            &[("d_model", self.d_model())]
        );
        x * self.scale(cond)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    /// A config non-default in every field, and the block built from it,
    /// answer the meta alike.
    #[test]
    fn test_meta() {
        let config = AdaptiveScaleConfig::new(12, 4)
            .with_activation(BunsenActivationConfig::Relu)
            .with_bias(true);
        let block: AdaptiveScale = config.init(&cpu_device());
        assert_eq!(block.d_model(), config.d_model());
        assert_eq!(block.d_hidden(), config.d_hidden());
        assert!(block.mlp.linear1.bias.is_some() && block.mlp.linear2.bias.is_some());
        assert!(matches!(
            block.mlp.act,
            crate::blocks::activation::BunsenActivation::Relu(_)
        ));
    }

    /// The scale is `1 + mlp(cond)`, its MLP GELU by default, broadcast over
    /// the sequence; with `linear2` zeroed it is the identity.
    #[test]
    fn test_scale() {
        let device = cpu_device();
        let mut block: AdaptiveScale = AdaptiveScaleConfig::new(6, 3).init(&device);
        let x = seeded_tensor::<3>(1, [2, 5, 6], Distribution::Default, &device);
        let cond = seeded_tensor::<3>(2, [1, 1, 6], Distribution::Default, &device);

        let expected = x.clone()
            * (block.mlp.linear2.forward(burn::tensor::activation::gelu(
                block.mlp.linear1.forward(cond.clone()),
            )) + 1.0);
        assert_tensors_close(
            &block.forward(x.clone(), cond.clone()),
            &expected,
            Tolerance::default(),
        );

        block.mlp.linear2.weight = Param::from_tensor(block.mlp.linear2.weight.val().zeros_like());
        assert_tensors_close(&block.forward(x.clone(), cond), &x, Tolerance::default());
    }
}
