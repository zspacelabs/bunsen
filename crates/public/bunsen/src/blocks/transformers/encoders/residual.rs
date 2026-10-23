//! The residual transformer encoder: its block, pre- or post-norm, and a stack
//! of them.
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
    burner::module::{
        ModuleInit,
        Rekey,
    },
    contracts::unpack_shape_contract,
    errors::BunsenResult,
    ops::transformers::attention::length_mask,
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
            attn_norm: norm(),
            attn,
            mlp_norm: norm(),
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
    pub attn_norm: Option<Normalization>,

    /// Attention.
    pub attn: MultiHeadAttention,

    /// MLP normalization, when the block normalizes.
    pub mlp_norm: Option<Normalization>,

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
                let x = x.clone() + attend(normalize(&self.attn_norm, x));
                x.clone() + self.mlp.forward(normalize(&self.mlp_norm, x))
            }
            NormPlacement::Post => {
                let x = normalize(&self.attn_norm, x.clone() + attend(x));
                normalize(&self.mlp_norm, x.clone() + self.mlp.forward(x))
            }
        }
    }
}

/// Config for [`ResidualEncoderTransformer`]: `n_layers` of one block.
#[derive(Config, Debug)]
pub struct ResidualEncoderTransformerConfig {
    /// Every layer's block.
    pub block: ResidualEncoderAttentionBlockConfig,

    /// The layers.
    pub n_layers: usize,
}

/// Common meta for [`ResidualEncoderTransformer`] and
/// [`ResidualEncoderTransformerConfig`].
pub trait ResidualEncoderTransformerMeta {
    /// The width it reads and writes.
    fn d_model(&self) -> usize;

    /// The number of layers.
    fn n_layers(&self) -> usize;
}

impl ResidualEncoderTransformerMeta for ResidualEncoderTransformerConfig {
    fn d_model(&self) -> usize {
        self.block.d_model()
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }
}

impl ModuleInit<ResidualEncoderTransformer> for ResidualEncoderTransformerConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ResidualEncoderTransformer> {
        Ok(ResidualEncoderTransformer {
            layers: (0..self.n_layers)
                .map(|_| self.block.try_init(device))
                .collect::<BunsenResult<_>>()?,
        })
    }
}

/// A stack of [`ResidualEncoderAttentionBlock`]s, without position
/// embeddings: a model that wants them adds them to the input. Padding
/// frames, past each item's length, are masked out as keys.
///
/// Post-norm with a `ReLU` MLP and a key bias, it is Sortformer v2's
/// transformer (`NeMo`'s `transformer_encoders.TransformerEncoder`, whose
/// padding mask this matches).
///
/// Built by [`ResidualEncoderTransformerConfig`].
/// Implements [`ResidualEncoderTransformerMeta`].
#[derive(Module, Debug)]
pub struct ResidualEncoderTransformer {
    /// The layers.
    pub layers: Vec<ResidualEncoderAttentionBlock>,
}

impl ResidualEncoderTransformerMeta for ResidualEncoderTransformer {
    fn d_model(&self) -> usize {
        self.layers.first().map_or(0, |l| l.d_model())
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

impl ResidualEncoderTransformer {
    /// Runs the layers.
    ///
    /// # Arguments
    /// * `x` - `[batch, frames, d_model]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `frames`.
    ///
    /// # Returns
    /// `[batch, frames, d_model]`. Frames past an item's length hold values
    /// that mean nothing.
    pub fn forward(
        &self,
        x: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        let [batch, frames] = unpack_shape_contract!(
            ["batch", "frames", "d_model"],
            &x.dims(),
            &["batch", "frames"],
            &[("d_model", self.d_model())],
        );
        // `[batch, frames, frames]`, true where key `j` is padding.
        let mask = lengths.map(|lengths| {
            length_mask(lengths, frames, &x.device())
                .bool_not()
                .unsqueeze_dim::<3>(1)
                .expand([batch, frames, frames])
        });
        self.layers
            .iter()
            .fold(x, |x, layer| layer.forward_masked(x, mask.clone()))
    }
}

impl ResidualEncoderAttentionBlock {
    /// Its tensors as a `NeMo` `transformer_encoders.TransformerEncoder`
    /// layer names them, relative to it: `first_sub_layer` the attention,
    /// `second_sub_layer` the MLP, `layer_norm_{1,2}` their norms. Anything
    /// else is dropped.
    pub fn nemo_rekey() -> Rekey {
        Rekey::drop_rest()
            .rename("first_sub_layer.query_net.", "attn.query.")
            .rename("first_sub_layer.key_net.", "attn.key.")
            .rename("first_sub_layer.value_net.", "attn.value.")
            .rename("first_sub_layer.out_projection.", "attn.output.")
            .rename("second_sub_layer.dense_in.", "mlp.linear1.")
            .rename("second_sub_layer.dense_out.", "mlp.linear2.")
            .rename("layer_norm_1.", "attn_norm.")
            .rename("layer_norm_2.", "mlp_norm.")
    }
}

impl ResidualEncoderTransformer {
    /// Its tensors as `NeMo`'s `transformer_encoders.TransformerEncoder` names
    /// them, relative to it: each `layers.{n}` by
    /// [`ResidualEncoderAttentionBlock::nemo_rekey`]. Anything else is
    /// dropped.
    pub fn nemo_rekey() -> Rekey {
        Rekey::drop_rest().mount_indexed(
            "layers.",
            "layers.",
            ResidualEncoderAttentionBlock::nemo_rekey(),
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Shape,
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        blocks::transformers::mlp::layer_norm_mlp,
        contracts::assert_shape_contract,
        ops::transformers::attention::layer_norm_self_attn,
        support::testing::{
            DeviceMemoryGuard,
            cpu_device,
        },
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
        assert_eq!(layer_norm(&block.attn_norm).gamma.dims(), [64]);
        assert_eq!(layer_norm(&block.mlp_norm).gamma.dims(), [64]);
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
        match &rms.attn_norm {
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
            assert!(block.attn_norm.is_none() && block.mlp_norm.is_none());
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
            let x = layer_norm(&block.attn_norm).forward(x + context);
            layer_norm(&block.mlp_norm).forward(x.clone() + block.mlp.forward(x))
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
                layer_norm_self_attn(layer_norm(&block.attn_norm), &block.attn, x.clone(), None);
            let x = x + self_attn.context;

            let mlp = layer_norm_mlp(layer_norm(&block.mlp_norm), &block.mlp, x.clone());
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

    fn toy_transformer() -> ResidualEncoderTransformer {
        ResidualEncoderTransformerConfig::new(
            ResidualEncoderAttentionBlockConfig::new(8)
                .with_d_head(4)
                .with_key_bias(true)
                .with_norm_placement(NormPlacement::Post),
            2,
        )
        .init(&cpu_device())
    }

    /// Each layer runs in turn, and padding is masked out as keys: a short
    /// item's valid frames, batched behind a longer one, are what it gives
    /// alone.
    #[test]
    fn test_transformer_layers_and_padding() {
        let device = cpu_device();
        let model = toy_transformer();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.d_model(), 8);

        let x = Tensor::<3>::random([2, 6, 8], Distribution::Default, &device);
        let whole = model.forward(x.clone(), None);
        let by_hand = model.layers[1].forward(model.layers[0].forward(x.clone()));
        whole
            .to_data()
            .assert_approx_eq::<f32>(&by_hand.to_data(), Tolerance::absolute(1e-6));

        let batched = model.forward(x.clone(), Some(&[6, 4]));
        let alone = model.forward(x.slice([1..2, 0..4]), None);
        batched
            .slice([1..2, 0..4])
            .to_data()
            .assert_approx_eq::<f32>(&alone.to_data(), Tolerance::absolute(1e-5));
    }

    /// `NeMo`'s `TransformerEncoder` names, layer by layer.
    #[test]
    fn test_transformer_nemo_rekey() {
        assert_eq!(
            ResidualEncoderTransformer::nemo_rekey()
                .apply("layers.17.first_sub_layer.key_net.bias")
                .as_deref(),
            Some("layers.17.attn.key.bias")
        );
        assert_eq!(
            ResidualEncoderTransformer::nemo_rekey()
                .apply("layers.0.second_sub_layer.dense_out.weight")
                .as_deref(),
            Some("layers.0.mlp.linear2.weight")
        );
        assert_eq!(
            ResidualEncoderTransformer::nemo_rekey()
                .apply("layers.2.layer_norm_2.bias")
                .as_deref(),
            Some("layers.2.mlp_norm.bias")
        );
        assert_eq!(
            ResidualEncoderTransformer::nemo_rekey().apply("final_layer_norm.weight"),
            None
        );
    }
}
