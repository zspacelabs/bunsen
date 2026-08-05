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
    prelude::{
        Backend,
        Bool,
    },
};

use super::WHISPER_DEFAULT_D_MODEL;
use crate::{
    blocks::transformers::{
        attention::{
            layer_norm_cross_attn,
            layer_norm_self_attn,
        },
        mlp::{
            Mlp,
            MlpConfig,
            layer_norm_mlp,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`ResidualDecoderAttentionBlock`] and
/// [`ResidualDecoderAttentionBlockConfig`].
pub trait ResidualDecoderAttentionBlockMeta {
    /// Returns the embedding dimensionality.
    fn d_model(&self) -> usize;

    /// Returns the number of heads.
    fn n_heads(&self) -> usize;

    /// Returns the dropout.
    fn dropout(&self) -> f64;
}

/// Config for [`ResidualDecoderAttentionBlock`].
#[derive(Config, Debug)]
pub struct ResidualDecoderAttentionBlockConfig {
    /// Returns the embedding dimensionality.
    pub d_model: usize,

    /// Head Dimensionality.
    #[config(defaul_value = "WHISPER_DEFAULT_D_MODEL")]
    pub d_head: usize,

    /// Dropout.
    #[config(default = "0.0")]
    pub dropout: f64,
}

impl ResidualDecoderAttentionBlockMeta for ResidualDecoderAttentionBlockConfig {
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

impl<B: Backend> ModuleInit<B, ResidualDecoderAttentionBlock<B>>
    for ResidualDecoderAttentionBlockConfig
{
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<ResidualDecoderAttentionBlock<B>> {
        let mha_cfg =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads()).with_dropout(self.dropout);
        let ln_cfg = LayerNormConfig::new(self.d_model);

        // Whisper doesn't use a key bias;
        // MHA doesn't let us configure this.
        let mut attn = mha_cfg.init(device);
        let mut cross_attn = mha_cfg.init(device);
        attn.key.bias = None;
        cross_attn.key.bias = None;

        Ok(ResidualDecoderAttentionBlock {
            attn_ln: ln_cfg.init(device),
            attn,
            cross_attn_ln: ln_cfg.init(device),
            cross_attn,
            mlp_ln: ln_cfg.init(device),
            mlp: MlpConfig::new(self.d_model).try_init(device)?,
        })
    }
}

/// Residual Decoder Attention Block for Whisper.
///
/// One Whisper decoder layer: pre-norm masked multi-head self-attention,
/// pre-norm cross-attention over the encoder output, and a pre-norm MLP, each
/// wrapped in a residual connection. Stacked inside the Whisper text decoder,
/// and also returns its cross-attention weights via [`DecodeRecord`].
///
/// Built by [`ResidualDecoderAttentionBlockConfig`].
#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    /// Attention Normalization.
    pub attn_ln: LayerNorm<B>,

    /// Attention.
    pub attn: MultiHeadAttention<B>,

    /// Cross Attention Normalization.
    pub cross_attn_ln: LayerNorm<B>,

    /// Cross Attention.
    pub cross_attn: MultiHeadAttention<B>,

    /// MLP Normalization.
    pub mlp_ln: LayerNorm<B>,

    /// MLP.
    pub mlp: Mlp<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlockMeta for ResidualDecoderAttentionBlock<B> {
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

/// Decode record for [`ResidualDecoderAttentionBlock::forward`].
#[derive(Debug, Clone)]
pub struct DecodeRecord<B: Backend> {
    /// Block Output: `[batch, seq_len, d_model]`.
    pub output: Tensor<B, 3>,

    /// Cross-Attention Weights: `[batch, n_heads, seq_len, seq_len]`.
    pub ca_weights: Tensor<B, 4>,
}

impl<B: Backend> DecodeRecord<B> {
    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.output.shape()[0]
    }

    /// Returns the embedding size.
    pub fn d_model(&self) -> usize {
        self.output.shape()[2]
    }

    /// Returns the sequence length.
    pub fn seq_len(&self) -> usize {
        self.output.shape()[1]
    }
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    /// Forward pass of the residual decoder attention block.
    ///
    /// # Arguments
    /// * `x` : `[batch, seq_len, d_model]` input.
    /// * `xa` : `[batch, seq_len, d_model]` cross-attention input.
    /// * `mask` : `[batch, seq_len, seq_len]` attention mask.
    ///
    /// # Returns
    /// `DecodeRecord` - forward record.
    /// * `fr.output` : `[batch, seq_len, d_model]`.
    /// * `fr.ca_weights` : `[batch, n_heads, seq_len, seq_len]`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
        mask: Option<Tensor<B, 3, Bool>>,
    ) -> DecodeRecord<B> {
        let self_attn = layer_norm_self_attn(&self.attn_ln, &self.attn, x.clone(), mask);
        let x = x + self_attn.context;

        let cross_attn =
            layer_norm_cross_attn(&self.cross_attn_ln, &self.cross_attn, x.clone(), xa);
        let x = x + cross_attn.context;

        let mlp = layer_norm_mlp(&self.mlp_ln, &self.mlp, x.clone());
        let x = x + mlp;

        DecodeRecord {
            output: x,
            ca_weights: cross_attn.weights,
        }
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

        let cfg = ResidualDecoderAttentionBlockConfig::new(d_model);
        let n_heads = cfg.n_heads();

        assert_eq!(cfg.d_model(), d_model);
        assert_eq!(cfg.n_heads(), n_heads);

        let block: ResidualDecoderAttentionBlock<B> = cfg.init(&device);

        assert_eq!(block.d_model(), d_model);
        assert_eq!(block.n_heads(), n_heads);

        let batch = 2;
        let seq_len = 10;
        let shape: Shape = [batch, seq_len, d_model].into();

        let x: Tensor<B, 3> = Tensor::random(shape.clone(), Distribution::Default, &device);
        let xa: Tensor<B, 3> = Tensor::random(shape.clone(), Distribution::Default, &device);

        let mask: Tensor<B, 3> =
            Tensor::random([1, seq_len, seq_len], Distribution::Bernoulli(0.5), &device);
        let mask = mask.bool();

        let result = block.forward(x.clone(), xa.clone(), mask.clone().into());

        assert_eq!(result.batch_size(), batch);
        assert_eq!(result.d_model(), d_model);
        assert_eq!(result.seq_len(), seq_len);

        let expected = {
            let self_attn =
                layer_norm_self_attn(&block.attn_ln, &block.attn, x.clone(), mask.clone().into());
            let x = x + self_attn.context;

            let cross_attn = layer_norm_cross_attn(
                &block.cross_attn_ln,
                &block.cross_attn,
                x.clone(),
                xa.clone(),
            );
            let x = x + cross_attn.context;

            let mlp = layer_norm_mlp(&block.mlp_ln, &block.mlp, x.clone());
            let x = x + mlp;

            DecodeRecord::<B> {
                output: x,
                ca_weights: cross_attn.weights,
            }
        };

        result
            .output
            .clone()
            .into_data()
            .assert_approx_eq::<f64>(&expected.output.to_data(), Default::default());
        result
            .ca_weights
            .clone()
            .into_data()
            .assert_approx_eq::<f64>(&expected.ca_weights.to_data(), Default::default());

        assert_shape_contract!(
            ["batch", "seq_len", "d_model"],
            &result.output,
            &[("batch", batch), ("seq_len", seq_len), ("d_model", d_model),],
        );

        assert_shape_contract!(
            ["batch", "n_heads", "seq_len", "seq_len"],
            &result.ca_weights,
            &[("batch", batch), ("n_heads", n_heads), ("seq_len", seq_len)],
        );
    }
}
