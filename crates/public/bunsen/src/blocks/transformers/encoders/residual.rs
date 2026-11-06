//! The residual transformer encoder: its block, pre- or post-norm, and a stack
//! of them.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNormConfig,
        attention::{
            MhaInput,
            MultiHeadAttention,
            MultiHeadAttentionConfig,
        },
    },
    tensor::Device,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        norm::BunsenNormalizationConfig,
        transformers::{
            mlp::{
                TransformerMlp,
                TransformerMlpConfig,
            },
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
        TensorLoadRowSplit,
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
    #[config(default = "Some(BunsenNormalizationConfig::Layer(LayerNormConfig::new(0)))")]
    pub norm: Option<BunsenNormalizationConfig>,

    /// Where the block normalizes, when it does.
    #[config(default = "NormPlacement::Pre")]
    pub norm_placement: NormPlacement,

    /// The MLP's activation: GELU for Whisper and Qwen3-ASR, `ReLU` for
    /// Sortformer v2.
    #[config(default = "BunsenActivationConfig::Gelu")]
    pub activation: BunsenActivationConfig,
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

        // The MLP projections carry a bias: `Linear -> act -> Linear`.
        let mlp: TransformerMlp = TransformerMlpConfig::new(self.d_model)
            .with_expansion_factor(self.expansion_factor)
            .with_activation(self.activation.clone())
            .with_bias(true)
            .try_init(device)?;

        let attn = mha_cfg.init(device);
        let sublayer = match self.norm_placement {
            NormPlacement::Pre => ResidualSublayerConfig::pre_norm(self.norm.clone()),
            NormPlacement::Post => ResidualSublayerConfig::post_norm(self.norm.clone()),
        };
        Ok(ResidualEncoderAttentionBlock {
            attn: sublayer.init(attn, self.d_model, device),
            mlp: sublayer.init(mlp, self.d_model, device),
            norm_placement: self.norm_placement,
        })
    }
}

/// Residual Encoder Attention Block: a transformer encoder layer.
///
/// Multi-head self-attention followed by an MLP, each wrapped in a residual
/// connection and normalized (a `LayerNorm` by default; any
/// [`BunsenNormalization`](crate::blocks::norm::BunsenNormalization), or none)
/// before the sublayer ([`NormPlacement::Pre`]) or after the residual sum
/// ([`NormPlacement::Post`]). Pre-norm with a GELU MLP, it is the encoder layer
/// of Whisper's audio encoder, and of Qwen3-ASR's (which adds a key bias and
/// attends within windows, through [`forward_masked`](Self::forward_masked));
/// post-norm with a `ReLU` MLP and a key bias, of Sortformer v2's transformer.
///
/// Built by [`ResidualEncoderAttentionBlockConfig`].
#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock {
    /// The attention sublayer: the attention (`branch`), and its norm, when
    /// the block normalizes: before it (`pre`) or after the add (`out`), by
    /// the block's placement.
    pub attn: ResidualSublayer<MultiHeadAttention>,

    /// The MLP sublayer: the MLP (`branch`), and its norm, when the block
    /// normalizes: before it (`pre`) or after the add (`out`), by the
    /// block's placement.
    pub mlp: ResidualSublayer<TransformerMlp>,

    /// Where the block normalizes, when it does.
    #[module(skip)]
    pub norm_placement: NormPlacement,
}

impl ResidualEncoderAttentionBlockMeta for ResidualEncoderAttentionBlock {
    fn d_model(&self) -> usize {
        self.attn.branch.d_model
    }

    fn n_heads(&self) -> usize {
        self.attn.branch.n_heads
    }

    fn dropout(&self) -> f64 {
        self.attn.branch.dropout.prob
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
        let attend = |attn: &MultiHeadAttention, x: Tensor<3>| {
            let input = MhaInput::self_attn(x);
            let input = match mask {
                Some(mask) => input.mask_attn(mask),
                None => input,
            };
            attn.forward(input).context
        };
        let x = self.attn.residual_update(x, attend);
        self.mlp.residual_update(x, |mlp, z| mlp.forward(z))
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
    /// `transformers`' Whisper encoder layer names (Whisper's own, and
    /// Qwen3-ASR's audio tower's) onto this block's, relative to it:
    /// `self_attn`'s `{q,k,v,out}_proj` the attention,
    /// `self_attn_layer_norm` and `final_layer_norm` the norms, `fc{1,2}`
    /// the MLP; everything else as it is.
    pub fn whisper_transformers_rekey() -> Rekey {
        Rekey::keep_rest()
            .rename("self_attn.q_proj.", "attn.query.")
            .rename("self_attn.k_proj.", "attn.key.")
            .rename("self_attn.v_proj.", "attn.value.")
            .rename("self_attn.out_proj.", "attn.output.")
            .rename("self_attn_layer_norm.", "attn_norm.")
            .rename("final_layer_norm.", "mlp_norm.")
            .rename("fc1.", "mlp.linear1.")
            .rename("fc2.", "mlp.linear2.")
    }

    /// The fused projections of a `PyTorch` `nn.TransformerEncoderLayer`,
    /// split by rows into the attention's three: `self_attn.in_proj_weight`
    /// and `self_attn.in_proj_bias` into
    /// `self_attn.{q,k,v}_proj.{weight,bias}`.
    pub fn torch_row_splits() -> [TensorLoadRowSplit; 2] {
        [
            TensorLoadRowSplit::equal(
                ".self_attn.in_proj_weight",
                [
                    ".self_attn.q_proj.weight",
                    ".self_attn.k_proj.weight",
                    ".self_attn.v_proj.weight",
                ],
            ),
            TensorLoadRowSplit::equal(
                ".self_attn.in_proj_bias",
                [
                    ".self_attn.q_proj.bias",
                    ".self_attn.k_proj.bias",
                    ".self_attn.v_proj.bias",
                ],
            ),
        ]
    }

    /// A `PyTorch` `nn.TransformerEncoderLayer`'s names, once
    /// [`torch_row_splits`](Self::torch_row_splits) has split its
    /// projections, onto this block's, relative to it: `self_attn`'s
    /// `{q,k,v,out}_proj` the attention, `norm1` and `norm2` the norms,
    /// `linear{1,2}` the MLP; everything else as it is. With `norm_first`, the
    /// block is pre-norm; its `ReLU` is the block's `activation`.
    pub fn torch_rekey() -> Rekey {
        Rekey::keep_rest()
            .rename("self_attn.q_proj.", "attn.query.")
            .rename("self_attn.k_proj.", "attn.key.")
            .rename("self_attn.v_proj.", "attn.value.")
            .rename("self_attn.out_proj.", "attn.output.")
            .rename("norm1.", "attn_norm.")
            .rename("norm2.", "mlp_norm.")
            .rename("linear1.", "mlp.linear1.")
            .rename("linear2.", "mlp.linear2.")
    }
}

impl CanonicalKeys for ResidualEncoderAttentionBlock {
    fn canonical_rekey(&self) -> Rekey {
        // Each sublayer is a `ResidualSublayer`: its norm is its `pre`, or its
        // `out` in a post-norm block; its module its `branch`.
        let norm = match self.norm_placement {
            NormPlacement::Pre => "pre",
            NormPlacement::Post => "out",
        };
        Rekey::keep_rest()
            .rename("attn_norm.", format!("attn.{norm}."))
            .rename("attn.", "attn.branch.")
            .rename("mlp_norm.", format!("mlp.{norm}."))
            .rename("mlp.", "mlp.branch.")
    }
}

impl CanonicalKeys for ResidualEncoderTransformer {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("layers.", &self.layers)
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
        blocks::norm::BunsenNormalization,
        contracts::assert_shape_contract,
        ops::transformers::attention::self_attn,
        support::testing::{
            DeviceMemoryGuard,
            cpu_device,
        },
    };

    /// The `LayerNorm` inside a block's norm.
    fn layer_norm(norm: &Option<BunsenNormalization>) -> &crate::blocks::norm::LayerNorm {
        match norm {
            Some(BunsenNormalization::Layer(norm)) => norm,
            other => panic!("not a LayerNorm: {other:?}"),
        }
    }

    /// The defaults are Whisper's block: pre-norm `LayerNorm`s sized to the
    /// model, GELU, no key bias. Its users build it from them.
    #[test]
    fn test_defaults_are_whispers() {
        let cfg = ResidualEncoderAttentionBlockConfig::new(64);
        assert!(matches!(
            cfg.norm,
            Some(BunsenNormalizationConfig::Layer(_))
        ));
        assert_eq!(cfg.norm_placement, NormPlacement::Pre);
        assert!(matches!(cfg.activation, BunsenActivationConfig::Gelu));
        assert!(!cfg.key_bias);
        let block: ResidualEncoderAttentionBlock = cfg.init(&crate::support::testing::cpu_device());
        assert_eq!(block.norm_placement(), NormPlacement::Pre);
        assert_eq!(layer_norm(&block.attn.pre).gamma.dims(), [64]);
        assert_eq!(layer_norm(&block.mlp.pre).gamma.dims(), [64]);
        assert!(block.attn.branch.key.bias.is_none());
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
        match &rms.attn.pre {
            Some(BunsenNormalization::Rms(norm)) => assert_eq!(norm.gamma.dims(), [32]),
            other => panic!("not an RmsNorm: {other:?}"),
        }

        let x: Tensor<3> = Tensor::random([2, 5, 32], Distribution::Default, &device);
        for placement in [NormPlacement::Pre, NormPlacement::Post] {
            let block: ResidualEncoderAttentionBlock = ResidualEncoderAttentionBlockConfig::new(32)
                .with_d_head(8)
                .with_norm(None)
                .with_norm_placement(placement)
                .init(&device);
            assert!(
                block.attn.pre.is_none()
                    && block.attn.out.is_none()
                    && block.mlp.pre.is_none()
                    && block.mlp.out.is_none()
            );
            assert!(block.num_params() < rms.num_params());

            let expected = {
                let x = x.clone()
                    + block
                        .attn
                        .branch
                        .forward(MhaInput::self_attn(x.clone()))
                        .context;
                x.clone() + block.mlp.branch.forward(x)
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
            .with_activation(BunsenActivationConfig::Relu);
        let block: ResidualEncoderAttentionBlock = cfg.init(&device);
        assert!(block.attn.branch.key.bias.is_some());
        assert_eq!(block.norm_placement(), NormPlacement::Post);

        let x: Tensor<3> = Tensor::random([2, 7, 32], Distribution::Default, &device);
        let mask: Tensor<3> = Tensor::random([1, 7, 7], Distribution::Bernoulli(0.3), &device);
        let mask = mask.bool();
        let output = block.forward_masked(x.clone(), Some(mask.clone()));

        let expected = {
            let context = block
                .attn
                .branch
                .forward(MhaInput::self_attn(x.clone()).mask_attn(mask))
                .context;
            let x = layer_norm(&block.attn.out).forward(x + context);
            layer_norm(&block.mlp.out).forward(x.clone() + block.mlp.branch.forward(x))
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
            let attended = self_attn(
                &block.attn.branch,
                block.attn.pre.as_ref().unwrap().forward(x.clone()),
                None,
            );
            let x = x + attended.context;

            x.clone()
                + block
                    .mlp
                    .branch
                    .forward(block.mlp.pre.as_ref().unwrap().forward(x))
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
}
