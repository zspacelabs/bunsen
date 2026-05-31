use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNorm,
        LayerNormConfig,
        attention::{
            MultiHeadAttention,
            MultiHeadAttentionConfig,
        },
    },
    prelude::Backend,
};

use crate::blocks::transformers::{
    attention::layer_norm_self_attn,
    mlp::{
        Mlp,
        MlpConfig,
        layer_norm_mlp,
    },
};

/// Common meta for [`ResidualEncoderAttentionBlock`] and
/// [`ResidualEncoderAttentionBlockConfig`].
pub trait ResidualEncoderAttentionBlockMeta {
    /// Return the number of states.
    fn n_states(&self) -> usize;

    /// Return the number of heads.
    fn n_heads(&self) -> usize;
}

/// Config for [`ResidualEncoderAttentionBlock`].
#[derive(Config, Debug)]
pub struct ResidualEncoderAttentionBlockConfig {
    /// Number of States.
    pub n_states: usize,

    /// Number of Heads.
    pub n_heads: usize,
}

impl ResidualEncoderAttentionBlockMeta for ResidualEncoderAttentionBlockConfig {
    fn n_states(&self) -> usize {
        self.n_states
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }
}

impl ResidualEncoderAttentionBlockConfig {
    /// Initialize a block.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ResidualEncoderAttentionBlock<B> {
        let mha_cfg = MultiHeadAttentionConfig::new(self.n_states, self.n_heads).with_dropout(0.0);
        let ln_cfg = LayerNormConfig::new(self.n_states);

        ResidualEncoderAttentionBlock {
            attn_ln: ln_cfg.init(device),
            attn: mha_cfg.init(device),
            mlp_ln: ln_cfg.init(device),
            mlp: MlpConfig::new(self.n_states).init(device),
        }
    }
}

/// Residual Encoder Attention Block for Whisper.
#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    /// Attention Normalization.
    pub attn_ln: LayerNorm<B>,

    /// Attention.
    pub attn: MultiHeadAttention<B>,

    /// MLP Normalization.
    pub mlp_ln: LayerNorm<B>,

    /// MLP.
    pub mlp: Mlp<B>,
}

impl<B: Backend> ResidualEncoderAttentionBlockMeta for ResidualEncoderAttentionBlock<B> {
    fn n_states(&self) -> usize {
        self.attn.d_model
    }

    fn n_heads(&self) -> usize {
        self.attn.n_heads
    }
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    /// Forward pass of the residual decoder attention block.
    ///
    /// ## Arguments
    /// * `x` - ``[batch, seq_len, n_states]`` input.
    ///
    /// ## Returns
    /// ``[batch, seq_len, n_states]``
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let self_attn = layer_norm_self_attn(&self.attn_ln, &self.attn, x.clone(), None);
        let x = x + self_attn.context;

        let mlp = layer_norm_mlp(&self.mlp_ln, &self.mlp, x.clone());
        x + mlp
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Shape,
        tensor::Distribution,
    };

    use super::*;
    use crate::contracts::assert_shape_contract;

    #[test]
    #[serial_test::serial]
    fn test_residual_decoder_forward() {
        type B = crate::support::testing::PerformanceBackend;
        let device = Default::default();

        let n_heads = 4;
        let n_states = 32 * n_heads;

        let cfg = ResidualEncoderAttentionBlockConfig::new(n_states, n_heads);

        assert_eq!(cfg.n_states(), n_states);
        assert_eq!(cfg.n_heads(), n_heads);

        let block: ResidualEncoderAttentionBlock<B> = cfg.init(&device);

        assert_eq!(block.n_states(), n_states);
        assert_eq!(block.n_heads(), n_heads);

        let batch = 2;
        let seq_len = 10;
        let shape: Shape = [batch, seq_len, n_states].into();

        let x: Tensor<B, 3> = Tensor::random(shape.clone(), Distribution::Default, &device);

        let output = block.forward(x.clone());

        let expected = {
            let self_attn = layer_norm_self_attn(&block.attn_ln, &block.attn, x.clone(), None);
            let x = x + self_attn.context;

            let mlp = layer_norm_mlp(&block.mlp_ln, &block.mlp, x.clone());
            x + mlp
        };

        output
            .clone()
            .into_data()
            .assert_approx_eq::<f64>(&expected.into_data(), Default::default());

        assert_shape_contract!(
            ["batch", "seq_len", "n_states"],
            &output,
            &[
                ("batch", batch),
                ("seq_len", seq_len),
                ("n_states", n_states),
            ],
        );
    }
}
