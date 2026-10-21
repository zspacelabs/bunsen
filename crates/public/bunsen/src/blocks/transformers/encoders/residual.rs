//! The transformer encoder block, pre- or post-norm.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNormConfig,
        activation::ActivationConfig,
        attention::{
            MhaInput,
            MultiHeadAttention,
            MultiHeadAttentionConfig,
        },
        norm::{
            Normalization,
            NormalizationConfig,
        },
    },
    tensor::Device,
};

use crate::{
    blocks::transformers::mlp::{
        Mlp,
        MlpConfig,
    },
    burner::module::ModuleInit,
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

    /// Where the block normalizes, when it does.
    fn norm_placement(&self) -> NormPlacement;
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

    /// The normalization of each sublayer, sized to `d_model` whatever
    /// feature count it names; `None` normalizes nothing, and builds no norm.
    #[config(default = "Some(NormalizationConfig::Layer(LayerNormConfig::new(0)))")]
    pub norm: Option<NormalizationConfig>,

    /// Where the block normalizes, when it does.
    #[config(default = "NormPlacement::Pre")]
    pub norm_placement: NormPlacement,

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

    fn norm_placement(&self) -> NormPlacement {
        self.norm_placement
    }
}

impl ModuleInit<ResidualEncoderAttentionBlock> for ResidualEncoderAttentionBlockConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> crate::errors::BunsenResult<ResidualEncoderAttentionBlock> {
        let mha_cfg = MultiHeadAttentionConfig::new(self.d_model, self.n_heads())
            .with_dropout(self.dropout)
            .with_key_bias(self.key_bias);
        let norm = || {
            self.norm
                .as_ref()
                .map(|n| n.clone().with_num_features(self.d_model).init(device))
        };

        let attn = mha_cfg.init(device);

        // The MLP projections carry a bias: `Linear -> act -> Linear`.
        let mlp: Mlp = MlpConfig::new(self.d_model)
            .with_expansion_factor(self.expansion_factor)
            .with_activation(self.activation.clone())
            .with_bias(true)
            .try_init(device)?;

        Ok(ResidualEncoderAttentionBlock {
            attn_ln: norm(),
            attn,
            mlp_ln: norm(),
            mlp,
            norm_placement: self.norm_placement,
        })
    }
}

/// Residual Encoder Attention Block: a transformer encoder layer.
///
/// Multi-head self-attention followed by an MLP, each wrapped in a residual
/// connection and normalized (a `LayerNorm` by default; any
/// [`Normalization`], or none) before the sublayer ([`NormPlacement::Pre`]) or
/// after the residual sum ([`NormPlacement::Post`]). Pre-norm with a GELU
/// MLP, it is the encoder layer of Whisper's audio encoder, and of
/// Qwen3-ASR's (which adds a key bias and attends within windows, through
/// [`forward_masked`](Self::forward_masked)); post-norm with a `ReLU` MLP and
/// a key bias, of Sortformer v2's transformer.
///
/// Built by [`ResidualEncoderAttentionBlockConfig`].
#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock {
    /// Attention normalization, when the block normalizes.
    pub attn_ln: Option<Normalization>,

    /// Attention.
    pub attn: MultiHeadAttention,

    /// MLP normalization, when the block normalizes.
    pub mlp_ln: Option<Normalization>,

    /// MLP.
    pub mlp: Mlp,

    /// Where the block normalizes, when it does.
    #[module(skip)]
    pub norm_placement: NormPlacement,
}

/// `x` through `norm`, or as it is without one.
fn normalize(
    norm: &Option<Normalization>,
    x: Tensor<3>,
) -> Tensor<3> {
    match norm {
        Some(norm) => norm.forward(x),
        None => x,
    }
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

    fn norm_placement(&self) -> NormPlacement {
        self.norm_placement
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
        let attend = |x: Tensor<3>| {
            let input = MhaInput::self_attn(x);
            let input = match mask {
                Some(mask) => input.mask_attn(mask),
                None => input,
            };
            self.attn.forward(input).context
        };
        match self.norm_placement {
            NormPlacement::Pre => {
                let x = x.clone() + attend(normalize(&self.attn_ln, x));
                x.clone() + self.mlp.forward(normalize(&self.mlp_ln, x))
            }
            NormPlacement::Post => {
                let x = normalize(&self.attn_ln, x.clone() + attend(x));
                normalize(&self.mlp_ln, x.clone() + self.mlp.forward(x))
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
        blocks::transformers::mlp::layer_norm_mlp,
        contracts::assert_shape_contract,
        ops::transformers::attention::layer_norm_self_attn,
        support::testing::DeviceMemoryGuard,
    };

    /// The `LayerNorm` inside a block's norm.
    fn layer_norm(norm: &Option<Normalization>) -> &burn::nn::LayerNorm {
        match norm {
            Some(Normalization::Layer(norm)) => norm,
            other => panic!("not a LayerNorm: {other:?}"),
        }
    }

    /// The defaults are Whisper's block: pre-norm `LayerNorm`s sized to the
    /// model, GELU, no key bias. Its users build it from them.
    #[test]
    fn test_defaults_are_whispers() {
        let cfg = ResidualEncoderAttentionBlockConfig::new(64);
        assert!(matches!(cfg.norm, Some(NormalizationConfig::Layer(_))));
        assert_eq!(cfg.norm_placement, NormPlacement::Pre);
        assert!(matches!(cfg.activation, ActivationConfig::Gelu));
        assert!(!cfg.key_bias);
        let block: ResidualEncoderAttentionBlock = cfg.init(&crate::support::testing::cpu_device());
        assert_eq!(block.norm_placement(), NormPlacement::Pre);
        assert_eq!(layer_norm(&block.attn_ln).gamma.dims(), [64]);
        assert_eq!(layer_norm(&block.mlp_ln).gamma.dims(), [64]);
        assert!(block.attn.key.bias.is_none());
    }

    /// Another normalization is sized to the model too, and `None` builds no
    /// norm and normalizes nothing, wherever it is placed.
    #[test]
    fn test_norm_is_configurable() {
        use burn::nn::RmsNormConfig;
        let device = crate::support::testing::cpu_device();
        let rms: ResidualEncoderAttentionBlock = ResidualEncoderAttentionBlockConfig::new(32)
            .with_d_head(8)
            .with_norm(Some(RmsNormConfig::new(0).into()))
            .init(&device);
        match &rms.attn_ln {
            Some(Normalization::Rms(norm)) => assert_eq!(norm.gamma.dims(), [32]),
            other => panic!("not an RmsNorm: {other:?}"),
        }

        let x: Tensor<3> = Tensor::random([2, 5, 32], Distribution::Default, &device);
        for placement in [NormPlacement::Pre, NormPlacement::Post] {
            let block: ResidualEncoderAttentionBlock = ResidualEncoderAttentionBlockConfig::new(32)
                .with_d_head(8)
                .with_norm(None)
                .with_norm_placement(placement)
                .init(&device);
            assert!(block.attn_ln.is_none() && block.mlp_ln.is_none());
            assert!(block.num_params() < rms.num_params());

            let expected = {
                let x = x.clone() + block.attn.forward(MhaInput::self_attn(x.clone())).context;
                x.clone() + block.mlp.forward(x)
            };
            block
                .forward(x.clone())
                .into_data()
                .assert_approx_eq::<f64>(&expected.into_data(), Default::default());
        }
    }

    /// Post-norm normalizes each residual sum: `norm(x + f(x))`.
    #[test]
    fn test_post_norm_forward() {
        let device = crate::support::testing::cpu_device();
        let cfg = ResidualEncoderAttentionBlockConfig::new(32)
            .with_d_head(8)
            .with_key_bias(true)
            .with_norm_placement(NormPlacement::Post)
            .with_activation(ActivationConfig::Relu);
        let block: ResidualEncoderAttentionBlock = cfg.init(&device);
        assert!(block.attn.key.bias.is_some());
        assert_eq!(block.norm_placement(), NormPlacement::Post);

        let x: Tensor<3> = Tensor::random([2, 7, 32], Distribution::Default, &device);
        let mask: Tensor<3> = Tensor::random([1, 7, 7], Distribution::Bernoulli(0.3), &device);
        let mask = mask.bool();
        let output = block.forward_masked(x.clone(), Some(mask.clone()));

        let expected = {
            let context = block
                .attn
                .forward(MhaInput::self_attn(x.clone()).mask_attn(mask))
                .context;
            let x = layer_norm(&block.attn_ln).forward(x + context);
            layer_norm(&block.mlp_ln).forward(x.clone() + block.mlp.forward(x))
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
            let self_attn =
                layer_norm_self_attn(layer_norm(&block.attn_ln), &block.attn, x.clone(), None);
            let x = x + self_attn.context;

            let mlp = layer_norm_mlp(layer_norm(&block.mlp_ln), &block.mlp, x.clone());
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
