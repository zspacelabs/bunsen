//! The Conformer block.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::LayerNormConfig,
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
    blocks::{
        norm::{
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
        transformers::{
            attention::rel_pos::{
                RelPosAttnCall,
                RelPosSelfAttention,
                RelPosSelfAttentionConfig,
                RelPosSelfAttentionMeta,
            },
            encoders::conformer::ConformerConvCall,
            sublayer::{
                ResidualSublayer,
                ResidualSublayerConfig,
            },
        },
    },
    burner::module::{
        CanonicalKeys,
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

    /// The normalization before each module and after the block, sized to
    /// `d_model` whatever feature count it names: a `LayerNorm` by default,
    /// the Conformer's.
    #[config(default = "BunsenNormalizationConfig::Layer(LayerNormConfig::new(0))")]
    pub norm: BunsenNormalizationConfig,
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
        let norm = || {
            self.norm
                .clone()
                .with_num_features(self.d_model)
                .init(device)
        };
        let ff = || {
            ConformerFeedForwardConfig::new(self.d_model, self.d_ff)
                .with_bias(self.bias)
                .try_init(device)
        };
        let ff_sublayer = ResidualSublayerConfig::pre_norm(Some(self.norm.clone()));
        Ok(ConformerBlock {
            feed_forward1: ff_sublayer.init(ff()?, self.d_model, device),
            attn: ff_sublayer.init(
                RelPosSelfAttentionConfig::new(self.d_model, self.n_heads)
                    .with_bias(self.bias)
                    .try_init(device)?,
                self.d_model,
                device,
            ),
            conv: ff_sublayer.init(
                ConformerConvolutionConfig::new(self.d_model, self.conv_kernel_size)
                    .with_bias(self.conv_bias)
                    .try_init(device)?,
                self.d_model,
                device,
            ),
            feed_forward2: ff_sublayer.init(ff()?, self.d_model, device),
            out_norm: norm(),
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
/// x = x + attn(norm(x))
/// x = x + conv(norm(x))
/// x = x + ff2(norm(x)) / 2
/// y = norm(x)
/// ```
///
/// Built by [`ConformerBlockConfig`].
/// Implements [`ConformerBlockMeta`].
#[derive(Module, Debug)]
pub struct ConformerBlock {
    /// The first feed-forward sublayer: its norm (`pre`) and the module
    /// (`branch`).
    pub feed_forward1: ResidualSublayer<ConformerFeedForward>,

    /// The attention sublayer: its norm (`pre`) and relative-position
    /// self-attention (`branch`).
    pub attn: ResidualSublayer<RelPosSelfAttention>,

    /// The convolution sublayer: its norm (`pre`) and the convolution module
    /// (`branch`).
    pub conv: ResidualSublayer<ConformerConvolution>,

    /// The second feed-forward sublayer: its norm (`pre`) and the module
    /// (`branch`).
    pub feed_forward2: ResidualSublayer<ConformerFeedForward>,

    /// The final normalization.
    pub out_norm: BunsenNormalization,
}

impl ConformerBlockMeta for ConformerBlock {
    fn d_model(&self) -> usize {
        self.attn.branch.d_model()
    }

    fn n_heads(&self) -> usize {
        self.attn.branch.n_heads()
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

        // The macaron feed-forwards: each half a step.
        let x = self
            .feed_forward1
            .residual_update_with(x, |ff, z| ff.forward(z).mul_scalar(0.5));
        let x = self.attn.residual_update(
            x,
            RelPosAttnCall {
                positions,
                valid: valid.clone(),
            },
        );
        let x = self.conv.residual_update(x, ConformerConvCall { valid });
        let x = self
            .feed_forward2
            .residual_update_with(x, |ff, z| ff.forward(z).mul_scalar(0.5));
        let y = self.out_norm.forward(x);

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

impl CanonicalKeys for ConformerBlock {
    fn canonical_rekey(&self) -> Rekey {
        // Each sublayer is a `ResidualSublayer`: its norm is its `pre`, its
        // module its `branch`.
        Rekey::keep_rest()
            .rename("feed_forward1_norm.", "feed_forward1.pre.")
            .rename("feed_forward1.", "feed_forward1.branch.")
            .rename("attn_norm.", "attn.pre.")
            .rename("attn.", "attn.branch.")
            .rename("conv_norm.", "conv.pre.")
            .rename("conv.", "conv.branch.")
            .rename("feed_forward2_norm.", "feed_forward2.pre.")
            .rename("feed_forward2.", "feed_forward2.branch.")
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
