//! The Conformer block.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNorm,
        LayerNormConfig,
    },
    prelude::Bool,
    tensor::Device,
};

use super::{
    ConformerConvolution,
    ConformerConvolutionConfig,
    ConformerFeedForward,
    ConformerFeedForwardConfig,
};
use crate::{
    blocks::transformers::attention::rel_pos::{
        RelPosSelfAttention,
        RelPosSelfAttentionConfig,
        RelPosSelfAttentionMeta,
    },
    burner::module::{
        ModuleInit,
        Rekey,
    },
    errors::BunsenResult,
};

/// Common meta for [`ConformerBlock`] and [`ConformerBlockConfig`].
pub trait ConformerBlockMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The attention head count.
    fn n_heads(&self) -> usize;
}

/// Config for [`ConformerBlock`].
///
/// Implements [`ConformerBlockMeta`].
#[derive(Config, Debug)]
pub struct ConformerBlockConfig {
    /// The model width.
    pub d_model: usize,

    /// The attention head count.
    pub n_heads: usize,

    /// The feed-forward hidden width.
    pub d_ff: usize,

    /// The convolution module's depthwise kernel size; odd.
    pub conv_kernel_size: usize,

    /// Whether the attention and feed-forward projections carry a bias.
    #[config(default = "true")]
    pub bias: bool,

    /// Whether the convolution module's convolutions carry a bias.
    #[config(default = "true")]
    pub conv_bias: bool,
}

impl ConformerBlockMeta for ConformerBlockConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}

impl ModuleInit<ConformerBlock> for ConformerBlockConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConformerBlock> {
        let norm = || LayerNormConfig::new(self.d_model).init(device);
        let ff = || {
            ConformerFeedForwardConfig::new(self.d_model, self.d_ff)
                .with_bias(self.bias)
                .try_init(device)
        };
        Ok(ConformerBlock {
            norm_feed_forward1: norm(),
            feed_forward1: ff()?,
            norm_self_att: norm(),
            self_attn: RelPosSelfAttentionConfig::new(self.d_model, self.n_heads)
                .with_bias(self.bias)
                .try_init(device)?,
            norm_conv: norm(),
            conv: ConformerConvolutionConfig::new(self.d_model, self.conv_kernel_size)
                .with_bias(self.conv_bias)
                .try_init(device)?,
            norm_feed_forward2: norm(),
            feed_forward2: ff()?,
            norm_out: norm(),
        })
    }
}

/// The Conformer block (Gulati et al. 2020): two half-step feed-forward
/// modules, macaron style, around relative-position self-attention and a
/// convolution module, each pre-normalized and residual, and a final
/// normalization.
///
/// ```text
/// x = x + ff1(norm(x)) / 2
/// x = x + self_attn(norm(x))
/// x = x + conv(norm(x))
/// x = x + ff2(norm(x)) / 2
/// y = norm(x)
/// ```
///
/// Built by [`ConformerBlockConfig`].
/// Implements [`ConformerBlockMeta`].
#[derive(Module, Debug)]
pub struct ConformerBlock {
    /// The first feed-forward module's normalization.
    pub norm_feed_forward1: LayerNorm,

    /// The first feed-forward module.
    pub feed_forward1: ConformerFeedForward,

    /// The attention's normalization.
    pub norm_self_att: LayerNorm,

    /// Relative-position self-attention.
    pub self_attn: RelPosSelfAttention,

    /// The convolution module's normalization.
    pub norm_conv: LayerNorm,

    /// The convolution module.
    pub conv: ConformerConvolution,

    /// The second feed-forward module's normalization.
    pub norm_feed_forward2: LayerNorm,

    /// The second feed-forward module.
    pub feed_forward2: ConformerFeedForward,

    /// The final normalization.
    pub norm_out: LayerNorm,
}

impl ConformerBlockMeta for ConformerBlock {
    fn d_model(&self) -> usize {
        self.self_attn.d_model()
    }

    fn n_heads(&self) -> usize {
        self.self_attn.n_heads()
    }
}

impl ConformerBlock {
    /// Applies the block.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    /// * `positions` - `[1, 2 * time - 1, d_model]`, the relative-position
    ///   table.
    /// * `valid` - `[batch, time]`, true at the frames that are not padding;
    ///   `None` means none is.
    ///
    /// # Returns
    /// `[batch, time, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        positions: Tensor<3>,
        valid: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [batch, time] = crate::contracts::unpack_shape_contract!(
            ["batch", "time", "d_model"],
            &x.dims(),
            &["batch", "time"],
            &[("d_model", self.d_model())],
        );
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            [1, 2 * "time" - 1, "d_model"],
            &positions.dims(),
            &[("time", time), ("d_model", self.d_model())],
        );
        #[cfg(any(debug_assertions, test))]
        if let Some(valid) = &valid {
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "time"],
                &valid.dims(),
                &[("batch", batch), ("time", time)],
            );
        }

        let x = x.clone()
            + self
                .feed_forward1
                .forward(self.norm_feed_forward1.forward(x))
                .mul_scalar(0.5);
        let x = x.clone()
            + self
                .self_attn
                .forward(self.norm_self_att.forward(x), positions, valid.clone());
        let x = x.clone() + self.conv.forward(self.norm_conv.forward(x), valid);
        let x = x.clone()
            + self
                .feed_forward2
                .forward(self.norm_feed_forward2.forward(x))
                .mul_scalar(0.5);
        let y = self.norm_out.forward(x);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "d_model"],
            &y.dims(),
            &[
                ("batch", batch),
                ("time", time),
                ("d_model", self.d_model())
            ],
        );

        y
    }
}

impl ConformerBlock {
    /// Its tensors as `transformers`' `ParakeetEncoderBlock` names them,
    /// relative to it: the attention's projections and positional biases, and
    /// the convolution module's depthwise convolution and norm, renamed;
    /// everything else as it is.
    pub fn transformers_rekey() -> Rekey {
        Rekey::keep_rest()
            .rename("self_attn.q_proj.", "self_attn.query.")
            .rename("self_attn.k_proj.", "self_attn.key.")
            .rename("self_attn.v_proj.", "self_attn.value.")
            .rename("self_attn.o_proj.", "self_attn.output.")
            .rename("self_attn.relative_k_proj.", "self_attn.pos.")
            .exact("self_attn.bias_u", "self_attn.pos_bias_u")
            .exact("self_attn.bias_v", "self_attn.pos_bias_v")
            .rename("conv.depthwise_conv.", "conv.depthwise.conv.")
            .rename("conv.norm.", "conv.depthwise.norm.")
    }

    /// Its tensors as `NeMo`'s `ConformerLayer` names them, relative to it:
    /// the attention's projections and the convolution module's depthwise
    /// convolution and batch norm, renamed; everything else as it is.
    pub fn nemo_rekey() -> Rekey {
        Rekey::keep_rest()
            .rename("self_attn.linear_q.", "self_attn.query.")
            .rename("self_attn.linear_k.", "self_attn.key.")
            .rename("self_attn.linear_v.", "self_attn.value.")
            .rename("self_attn.linear_out.", "self_attn.output.")
            .rename("self_attn.linear_pos.", "self_attn.pos.")
            .rename("conv.depthwise_conv.", "conv.depthwise.conv.")
            .rename("conv.batch_norm.", "conv.depthwise.norm.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = ConformerBlockConfig::new(8, 2, 16, 3)
            .with_bias(false)
            .with_conv_bias(false);
        let module: ConformerBlock = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.n_heads(), config.n_heads());
    }
}
