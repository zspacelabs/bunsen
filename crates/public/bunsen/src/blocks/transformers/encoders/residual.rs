//! The transformer encoder block, pre- or post-norm.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNorm,
        LayerNormConfig,
        activation::ActivationConfig,
        attention::{
            MhaInput,
            MultiHeadAttention,
            MultiHeadAttentionConfig,
        },
    },
    tensor::Device,
};

use crate::{
    blocks::transformers::mlp::{
        Mlp,
        MlpConfig,
        layer_norm_mlp,
    },
    burner::module::ModuleInit,
    ops::transformers::attention::layer_norm_self_attn,
};

/// Where a [`ResidualEncoderAttentionBlock`] normalizes, around each of its
/// two sublayers (self-attention, then the MLP).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NormPlacement {
    /// Before the sublayer, inside its residual: `x + f(norm(x))`. Whisper's
    /// and Qwen3-ASR's.
    Pre,

    /// After the residual sum: `norm(x + f(x))`. The original Transformer's,
    /// and `NeMo`'s `TransformerEncoder` without `pre_ln`: Sortformer v2's.
    Post,
}

/// Common meta for [`ResidualEncoderAttentionBlock`] and
/// [`ResidualEncoderAttentionBlockConfig`].
pub trait ResidualEncoderAttentionBlockMeta {
    /// Returns the embedding dimensionality.
    fn d_model(&self) -> usize;

    /// Returns the number of heads.
    fn n_heads(&self) -> usize;

    /// Returns the dropout.
    fn dropout(&self) -> f64;

    /// Where the block normalizes.
    fn norm(&self) -> NormPlacement;
}

/// Config for [`ResidualEncoderAttentionBlock`].
#[derive(Config, Debug)]
pub struct ResidualEncoderAttentionBlockConfig {
    /// Returns the embedding dimensionality.
    pub d_model: usize,

    /// Head Dimensionality.
    #[config(default = "64")]
    pub d_head: usize,

    /// Dropout.
    #[config(default = "0.0")]
    pub dropout: f64,

    /// The MLP's hidden width over `d_model`.
    #[config(default = "4")]
    pub expansion_factor: usize,

    /// Whether the key projection carries a bias. Whisper's does not;
    /// Qwen3-ASR's audio encoder's does. The query, value, output and MLP
    /// projections always do.
    #[config(default = "false")]
    pub key_bias: bool,

    /// Where the block normalizes.
    #[config(default = "NormPlacement::Pre")]
    pub norm: NormPlacement,

    /// The MLP's activation: GELU for Whisper and Qwen3-ASR, `ReLU` for
    /// Sortformer v2.
    #[config(default = "ActivationConfig::Gelu")]
    pub activation: ActivationConfig,
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

    fn norm(&self) -> NormPlacement {
        self.norm
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

        // burn's `MultiHeadAttention` gives every projection a bias; a block
        // without a key bias drops it.
        let mut attn = mha_cfg.init(device);
        if !self.key_bias {
            attn.key.bias = None;
        }

        // The MLP projections carry a bias: `Linear -> act -> Linear`.
        let mlp: Mlp = MlpConfig::new(self.d_model)
            .with_expansion_factor(self.expansion_factor)
            .with_activation(self.activation.clone())
            .with_bias(true)
            .try_init(device)?;

        Ok(ResidualEncoderAttentionBlock {
            attn_ln: ln_cfg.init(device),
            attn,
            mlp_ln: ln_cfg.init(device),
            mlp,
            norm: self.norm,
        })
    }
}

/// Residual Encoder Attention Block: a transformer encoder layer.
///
/// Multi-head self-attention followed by an MLP, each wrapped in a residual
/// connection and normalized before the sublayer ([`NormPlacement::Pre`]) or
/// after the residual sum ([`NormPlacement::Post`]). Pre-norm with a GELU
/// MLP, it is the encoder layer of Whisper's audio encoder, and of
/// Qwen3-ASR's (which adds a key bias and attends within windows, through
/// [`forward_masked`](Self::forward_masked)); post-norm with a `ReLU` MLP and
/// a key bias, of Sortformer v2's transformer.
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

    /// Where the block normalizes.
    #[module(skip)]
    pub norm: NormPlacement,
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

    fn norm(&self) -> NormPlacement {
        self.norm
    }
}

impl ResidualEncoderAttentionBlock {
    /// Forward pass of the residual encoder attention block.
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
        self.forward_masked(x, None)
    }

    /// [`forward`](Self::forward), attending only where `mask` allows.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq, d_model]`.
    /// * `mask` - `[batch, seq, seq]` (or `[1, seq, seq]`), true where query
    ///   `i` may **not** attend to key `j`; `None` attends everywhere.
    pub fn forward_masked(
        &self,
        x: Tensor<3>,
        mask: Option<Tensor<3, burn::prelude::Bool>>,
    ) -> Tensor<3> {
        match self.norm {
            NormPlacement::Pre => {
                let self_attn = layer_norm_self_attn(&self.attn_ln, &self.attn, x.clone(), mask);
                let x = x + self_attn.context;

                let mlp = layer_norm_mlp(&self.mlp_ln, &self.mlp, x.clone());
                x + mlp
            }
            NormPlacement::Post => {
                let input = MhaInput::self_attn(x.clone());
                let input = match mask {
                    Some(mask) => input.mask_attn(mask),
                    None => input,
                };
                let x = self.attn_ln.forward(x + self.attn.forward(input).context);
                self.mlp_ln.forward(x.clone() + self.mlp.forward(x))
            }
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
    use crate::{
        contracts::assert_shape_contract,
        support::testing::DeviceMemoryGuard,
    };

    /// The defaults are Whisper's block: pre-norm, GELU, no key bias. Its
    /// users build it from them.
    #[test]
    fn test_defaults_are_whispers() {
        let cfg = ResidualEncoderAttentionBlockConfig::new(64);
        assert_eq!(cfg.norm, NormPlacement::Pre);
        assert!(matches!(cfg.activation, ActivationConfig::Gelu));
        assert!(!cfg.key_bias);
        let block: ResidualEncoderAttentionBlock = cfg.init(&crate::support::testing::cpu_device());
        assert_eq!(block.norm(), NormPlacement::Pre);
        assert!(block.attn.key.bias.is_none());
    }

    /// Post-norm normalizes each residual sum: `norm(x + f(x))`.
    #[test]
    fn test_post_norm_forward() {
        let device = crate::support::testing::cpu_device();
        let cfg = ResidualEncoderAttentionBlockConfig::new(32)
            .with_d_head(8)
            .with_key_bias(true)
            .with_norm(NormPlacement::Post)
            .with_activation(ActivationConfig::Relu);
        let block: ResidualEncoderAttentionBlock = cfg.init(&device);
        assert!(block.attn.key.bias.is_some());
        assert_eq!(block.norm(), NormPlacement::Post);

        let x: Tensor<3> = Tensor::random([2, 7, 32], Distribution::Default, &device);
        let mask: Tensor<3> = Tensor::random([1, 7, 7], Distribution::Bernoulli(0.3), &device);
        let mask = mask.bool();
        let output = block.forward_masked(x.clone(), Some(mask.clone()));

        let expected = {
            let context = block
                .attn
                .forward(MhaInput::self_attn(x.clone()).mask_attn(mask))
                .context;
            let x = block.attn_ln.forward(x + context);
            block.mlp_ln.forward(x.clone() + block.mlp.forward(x))
        };
        output
            .into_data()
            .assert_approx_eq::<f64>(&expected.into_data(), Default::default());
    }

    #[test]
    #[serial_test::serial]
    fn test_residual_encoder_forward() {
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
