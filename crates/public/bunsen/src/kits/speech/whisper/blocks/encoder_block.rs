use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNorm,
        LayerNormConfig,
        activation::ActivationConfig,
        attention::{
            MultiHeadAttention,
            MultiHeadAttentionConfig,
        },
    },
    tensor::Device,
};

use super::WHISPER_DEFAULT_D_MODEL;
use crate::{
    blocks::transformers::mlp::{
        Mlp,
        MlpConfig,
        layer_norm_mlp,
    },
    burner::module::ModuleInit,
    ops::transformers::attention::layer_norm_self_attn,
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

impl ModuleInit<ResidualEncoderAttentionBlock> for ResidualEncoderAttentionBlockConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> crate::errors::BunsenResult<ResidualEncoderAttentionBlock> {
        let mha_cfg =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads()).with_dropout(self.dropout);
        let ln_cfg = LayerNormConfig::new(self.d_model);

        // Whisper doesn't use a key bias;
        // MHA doesn't let us configure this.
        let mut attn = mha_cfg.init(device);
        attn.key.bias = None;

        // Whisper's MLP projections carry a bias, and it runs
        // `Linear -> GELU -> Linear`; the `MlpConfig` default is ReLU.
        let mlp: Mlp = MlpConfig::new(self.d_model)
            .with_activation(ActivationConfig::Gelu)
            .with_bias(true)
            .try_init(device)?;

        Ok(ResidualEncoderAttentionBlock {
            attn_ln: ln_cfg.init(device),
            attn,
            mlp_ln: ln_cfg.init(device),
            mlp,
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
pub struct ResidualEncoderAttentionBlock {
    /// Attention Normalization.
    pub attn_ln: LayerNorm,

    /// Attention.
    pub attn: MultiHeadAttention,

    /// MLP Normalization.
    pub mlp_ln: LayerNorm,

    /// MLP.
    pub mlp: Mlp,
}

impl ResidualEncoderAttentionBlockMeta for ResidualEncoderAttentionBlock {
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

impl ResidualEncoderAttentionBlock {
    /// Forward pass of the residual decoder attention block.
    ///
    /// # Arguments
    /// * `x` : `[batch, seq_len, d_model]` input.
    ///
    /// # Returns
    /// `[batch, seq_len, d_model]`
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
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
    use crate::{
        contracts::assert_shape_contract,
        support::testing::DeviceMemoryGuard,
    };

    #[test]
    #[serial_test::serial]
    fn test_residual_decoder_forward() {
        use crate::support::testing::performance_device;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let d_model = 128;

        let cfg = ResidualEncoderAttentionBlockConfig::new(d_model);
        let n_heads = cfg.n_heads();

        assert_eq!(cfg.d_model(), d_model);
        assert_eq!(cfg.n_heads(), n_heads);

        let block: ResidualEncoderAttentionBlock = cfg.init(&device);

        assert_eq!(block.d_model(), d_model);
        assert_eq!(block.n_heads(), n_heads);

        let batch = 2;
        let seq_len = 10;
        let shape: Shape = [batch, seq_len, d_model].into();

        let x: Tensor<3> = Tensor::random(shape.clone(), Distribution::Default, &device);

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
