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

use super::WHISPER_DEFAULT_D_MODEL;
use crate::{
    blocks::transformers::{
        attention::layer_norm_self_attn,
        mlp::{
            Mlp,
            MlpConfig,
            layer_norm_mlp,
        },
    },
    burner::module::ModuleInit,
};

/// Common meta for [`ResidualEncoderAttentionBlock`] and
/// [`ResidualEncoderAttentionBlockConfig`].
pub trait ResidualEncoderAttentionBlockMeta {
    /// Returns the embedding dimensionality.
    fn d_model(&self) -> usize;

    /// Returns the number of heads.
    fn n_heads(&self) -> usize;

    /// Returns the dropout.
    fn dropout(&self) -> f64;
}

/// Config for [`ResidualEncoderAttentionBlock`].
#[derive(Config, Debug)]
pub struct ResidualEncoderAttentionBlockConfig {
    /// Returns the embedding dimensionality.
    pub d_model: usize,

    /// Head Dimensionality.
    #[config(defaul_value = "WHISPER_DEFAULT_D_MODEL")]
    pub d_head: usize,

    /// Dropout.
    #[config(default = "0.0")]
    pub dropout: f64,
}

impl ResidualEncoderAttentionBlockMeta for ResidualEncoderAttentionBlockConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_heads(&self) -> usize {
        self.d_model / self.d_head
    }

    fn dropout(&self) -> f64 {
        self.dropout
    }
}

impl<B: Backend> ModuleInit<B, ResidualEncoderAttentionBlock<B>>
    for ResidualEncoderAttentionBlockConfig
{
    fn try_init(
        &self,
        device: &B::Device,
    ) -> crate::errors::BunsenResult<ResidualEncoderAttentionBlock<B>> {
        let mha_cfg =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads()).with_dropout(self.dropout);
        let ln_cfg = LayerNormConfig::new(self.d_model);

        // Whisper doesn't use a key bias;
        // MHA doesn't let us configure this.
        let mut attn = mha_cfg.init(device);
        attn.key.bias = None;

        Ok(ResidualEncoderAttentionBlock {
            attn_ln: ln_cfg.init(device),
            attn,
            mlp_ln: ln_cfg.init(device),
            // Whisper's MLP projections carry a bias. The default `Row`
            // layout is correct here: `PyTorchToBurnAdapter` already
            // transposes the incoming `[d_output, d_input]` weight, so `Col`
            // would transpose a second time.
            mlp: MlpConfig::new(self.d_model)
                .with_bias(true)
                .try_init(device)?,
        })
    }
}

/// Residual Encoder Attention Block for Whisper.
///
/// One Whisper encoder layer: pre-norm multi-head self-attention followed by a
/// pre-norm MLP, each wrapped in a residual connection. Stacked inside the
/// Whisper audio encoder.
///
/// Built by [`ResidualEncoderAttentionBlockConfig`].
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
    fn d_model(&self) -> usize {
        self.attn.d_model
    }

    fn n_heads(&self) -> usize {
        self.attn.n_heads
    }

    fn dropout(&self) -> f64 {
        self.attn.dropout.prob
    }
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    /// Forward pass of the residual decoder attention block.
    ///
    /// # Arguments
    /// * `x` : `[batch, seq_len, d_model]` input.
    ///
    /// # Returns
    /// `[batch, seq_len, d_model]`
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

        let d_model = 128;

        let cfg = ResidualEncoderAttentionBlockConfig::new(d_model);
        let n_heads = cfg.n_heads();

        assert_eq!(cfg.d_model(), d_model);
        assert_eq!(cfg.n_heads(), n_heads);

        let block: ResidualEncoderAttentionBlock<B> = cfg.init(&device);

        assert_eq!(block.d_model(), d_model);
        assert_eq!(block.n_heads(), n_heads);

        let batch = 2;
        let seq_len = 10;
        let shape: Shape = [batch, seq_len, d_model].into();

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
            ["batch", "seq_len", "d_model"],
            &output,
            &[("batch", batch), ("seq_len", seq_len), ("d_model", d_model),],
        );
    }
}
