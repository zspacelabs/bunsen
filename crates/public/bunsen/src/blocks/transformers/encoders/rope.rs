//! # A pre-norm transformer encoder with rotary position embeddings.
//!
//! [`RopeTransformerEncoder`]: a `LayerNorm` over the input, a stack of
//! [`RopeEncoderBlock`]s sharing one rotary table over the whole input, and a
//! final `LayerNorm`. The encoder of NVIDIA's `Nemotron-3-Diarization`
//! (`NeMo`'s `TransformerEncoder` with `rope` and `pre_block_norm`).

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNorm,
        LayerNormConfig,
        activation::ActivationConfig,
    },
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        attention::rope::{
            RopeSelfAttention,
            RopeSelfAttentionConfig,
            RopeSelfAttentionMeta,
        },
        embedding::{
            RotaryDirection,
            RotaryEmbeddingConfig,
        },
        mlp::{
            Mlp,
            MlpConfig,
        },
    },
    burner::module::{
        ModuleInit,
        Rekey,
    },
    contracts::unpack_shape_contract,
    errors::BunsenResult,
    ops::transformers::attention::length_mask,
};

/// Common meta for [`RopeEncoderBlock`] and [`RopeEncoderBlockConfig`].
pub trait RopeEncoderBlockMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The head count.
    fn n_heads(&self) -> usize;

    /// The feed-forward's hidden width.
    fn d_ff(&self) -> usize;
}

/// Config for [`RopeEncoderBlock`].
///
/// Implements [`RopeEncoderBlockMeta`].
#[derive(Config, Debug)]
pub struct RopeEncoderBlockConfig {
    /// The model width.
    pub d_model: usize,

    /// The head count.
    pub n_heads: usize,

    /// The feed-forward's hidden width over `d_model`.
    #[config(default = "4")]
    pub ff_expansion: usize,

    /// The `LayerNorm` epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,

    /// Whether the query, key and value projections carry a bias.
    #[config(default = "false")]
    pub qkv_bias: bool,
}

impl RopeEncoderBlockMeta for RopeEncoderBlockConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }

    fn d_ff(&self) -> usize {
        self.d_model * self.ff_expansion
    }
}

impl ModuleInit<RopeEncoderBlock> for RopeEncoderBlockConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<RopeEncoderBlock> {
        let norm = || {
            LayerNormConfig::new(self.d_model)
                .with_epsilon(self.norm_eps)
                .init(device)
        };
        Ok(RopeEncoderBlock {
            norm1: norm(),
            attn: RopeSelfAttentionConfig::new(self.d_model, self.n_heads)
                .with_qkv_bias(self.qkv_bias)
                .try_init(device)?,
            norm2: norm(),
            mlp: MlpConfig::new(self.d_model)
                .with_expansion_factor(self.ff_expansion)
                .with_activation(ActivationConfig::Gelu)
                .with_bias(true)
                .try_init(device)?,
        })
    }
}

/// A pre-norm encoder layer with rotary attention: `x + attn(norm1(x))`, then
/// `x + mlp(norm2(x))`, the feed-forward `Linear -> GELU -> Linear`.
///
/// Built by [`RopeEncoderBlockConfig`].
/// Implements [`RopeEncoderBlockMeta`].
#[derive(Module, Debug)]
pub struct RopeEncoderBlock {
    /// The attention's normalization.
    pub norm1: LayerNorm,

    /// The attention.
    pub attn: RopeSelfAttention,

    /// The feed-forward's normalization.
    pub norm2: LayerNorm,

    /// The feed-forward.
    pub mlp: Mlp,
}

impl RopeEncoderBlockMeta for RopeEncoderBlock {
    fn d_model(&self) -> usize {
        self.attn.d_model()
    }

    fn n_heads(&self) -> usize {
        self.attn.n_heads()
    }

    fn d_ff(&self) -> usize {
        self.mlp.linear1.weight.dims()[1]
    }
}

impl RopeEncoderBlock {
    /// One layer.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    /// * `rope` - the rotary tables over positions `0..time`.
    /// * `valid` - `[batch, time]`, true at the frames that are not padding.
    ///
    /// # Returns
    /// `[batch, time, d_model]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        rope: &crate::blocks::transformers::embedding::RotaryEmbedding,
        valid: Option<Tensor<2, burn::prelude::Bool>>,
    ) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.norm1.forward(x), rope, valid);
        x.clone() + self.mlp.forward(self.norm2.forward(x))
    }
}

/// Common meta for [`RopeTransformerEncoder`] and
/// [`RopeTransformerEncoderConfig`].
pub trait RopeTransformerEncoderMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The head count.
    fn n_heads(&self) -> usize;
}

/// Config for [`RopeTransformerEncoder`].
///
/// Implements [`RopeTransformerEncoderMeta`].
#[derive(Config, Debug)]
pub struct RopeTransformerEncoderConfig {
    /// The layer every block is built from.
    pub block: RopeEncoderBlockConfig,

    /// The layer count.
    pub n_layers: usize,

    /// The rotary embedding's base, `theta`.
    #[config(default = "10000")]
    pub rope_base: usize,
}

impl RopeTransformerEncoderMeta for RopeTransformerEncoderConfig {
    fn d_model(&self) -> usize {
        self.block.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn n_heads(&self) -> usize {
        self.block.n_heads
    }
}

impl ModuleInit<RopeTransformerEncoder> for RopeTransformerEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<RopeTransformerEncoder> {
        let norm = || {
            LayerNormConfig::new(self.block.d_model)
                .with_epsilon(self.block.norm_eps)
                .init(device)
        };
        Ok(RopeTransformerEncoder {
            embed_norm: norm(),
            layers: (0..self.n_layers)
                .map(|_| self.block.try_init(device))
                .collect::<BunsenResult<_>>()?,
            final_norm: norm(),
            rope_base: self.rope_base,
        })
    }
}

/// A pre-norm transformer encoder with rotary position embeddings: a
/// `LayerNorm` over the input, the [`RopeEncoderBlock`]s, and a final
/// `LayerNorm`.
///
/// The rotary table runs over positions `0..time` of the whole input, in the
/// `transformers` (`rotate_half`) convention.
///
/// Built by [`RopeTransformerEncoderConfig`].
/// Implements [`RopeTransformerEncoderMeta`].
#[derive(Module, Debug)]
pub struct RopeTransformerEncoder {
    /// The input's normalization.
    pub embed_norm: LayerNorm,

    /// The layers.
    pub layers: Vec<RopeEncoderBlock>,

    /// The output's normalization.
    pub final_norm: LayerNorm,

    /// The rotary embedding's base.
    #[module(skip)]
    pub rope_base: usize,
}

impl RopeTransformerEncoderMeta for RopeTransformerEncoder {
    fn d_model(&self) -> usize {
        self.embed_norm.gamma.dims()[0]
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn n_heads(&self) -> usize {
        self.layers.first().map_or(0, |l| l.n_heads())
    }
}

impl RopeTransformerEncoder {
    /// Encodes.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, d_model]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    ///
    /// # Returns
    /// `[batch, time, d_model]`. Frames past an item's length hold values
    /// that mean nothing.
    pub fn forward(
        &self,
        x: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        let [batch, time] = unpack_shape_contract!(
            ["batch", "time", "d_model"],
            &x.dims(),
            &["batch", "time"],
            &[("d_model", self.d_model())],
        );
        let device = x.device();
        let head_dim = self.d_model() / self.n_heads().max(1);
        let rope = RotaryEmbeddingConfig::new(time, head_dim)
            .with_base(self.rope_base)
            .with_direction(RotaryDirection::Forward)
            .init(&device);
        let valid = lengths.map(|lengths| length_mask(lengths, time, &device));

        let mut x = self.embed_norm.forward(x);
        for layer in &self.layers {
            x = layer.forward(x, &rope, valid.clone());
        }
        let y = self.final_norm.forward(x);

        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;
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

impl RopeEncoderBlock {
    /// Its tensors as `transformers`' `Nemotron3DiarizationEncoderLayer`
    /// names them, relative to it: `layer_norm{1,2}` its norms, `self_attn`'s
    /// `{q,k,v,o}_proj` the attention, `mlp.fc{1,2}` the MLP. Anything else is
    /// dropped.
    pub fn transformers_rekey() -> Rekey {
        Rekey::drop_rest()
            .rename("layer_norm1.", "norm1.")
            .rename("layer_norm2.", "norm2.")
            .rename("self_attn.q_proj.", "attn.query.")
            .rename("self_attn.k_proj.", "attn.key.")
            .rename("self_attn.v_proj.", "attn.value.")
            .rename("self_attn.o_proj.", "attn.output.")
            .rename("mlp.fc1.", "mlp.linear1.")
            .rename("mlp.fc2.", "mlp.linear2.")
    }
}

impl RopeTransformerEncoder {
    /// Its tensors as `transformers`' `Nemotron3DiarizationEncoder` names
    /// them, relative to it: `input_layer_norm` the embedding norm,
    /// `layer_norm` the final one, each `layers.{n}` by
    /// [`RopeEncoderBlock::transformers_rekey`]. Anything else (the rotary
    /// table's `inv_freq`, a buffer) is dropped.
    pub fn transformers_rekey() -> Rekey {
        Rekey::drop_rest()
            .rename("input_layer_norm.", "embed_norm.")
            .rename("layer_norm.", "final_norm.")
            .mount_indexed("layers.", "layers.", RopeEncoderBlock::transformers_rekey())
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        cpu_device,
        seeded_tensor,
    };

    /// A toy config, non-default in every field the metas read.
    fn toy() -> RopeTransformerEncoderConfig {
        RopeTransformerEncoderConfig::new(
            RopeEncoderBlockConfig::new(16, 2)
                .with_ff_expansion(3)
                .with_qkv_bias(true),
            3,
        )
        .with_rope_base(500)
    }

    #[test]
    fn test_meta() {
        let config = toy();
        let module: RopeTransformerEncoder = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.n_layers(), config.n_layers());
        assert_eq!(module.n_heads(), config.n_heads());
        assert_eq!(module.rope_base, 500);

        let block = &module.layers[0];
        assert_eq!(block.d_model(), config.block.d_model());
        assert_eq!(block.n_heads(), config.block.n_heads());
        assert_eq!(block.d_ff(), config.block.d_ff());
        assert!(block.attn.query.bias.is_some());
    }

    /// The stack is its parts, in order: input norm, blocks, output norm.
    #[test]
    fn test_composition() {
        let device = cpu_device();
        let module: RopeTransformerEncoder = toy().init(&device);
        let x = seeded_tensor::<3>(1, [2, 7, 16], Distribution::Default, &device);
        let got = module.forward(x.clone(), None);

        let rope = RotaryEmbeddingConfig::new(7, 8)
            .with_base(500)
            .with_direction(RotaryDirection::Forward)
            .init(&device);
        let mut want = module.embed_norm.forward(x);
        for layer in &module.layers {
            let attended = want.clone()
                + layer
                    .attn
                    .forward(layer.norm1.forward(want.clone()), &rope, None);
            want = attended.clone() + layer.mlp.forward(layer.norm2.forward(attended));
        }
        let want = module.final_norm.forward(want);
        got.into_data()
            .assert_approx_eq::<f32>(&want.into_data(), Tolerance::absolute(1e-5));
    }

    /// A padded item's valid frames are the item's own.
    #[test]
    fn test_padding_is_ignored() {
        let device = cpu_device();
        let module: RopeTransformerEncoder = toy().init(&device);
        let x = seeded_tensor::<3>(2, [1, 5, 16], Distribution::Default, &device);
        let junk = seeded_tensor::<3>(3, [1, 4, 16], Distribution::Default, &device);
        let alone = module.forward(x.clone(), None);
        let padded = module
            .forward(Tensor::cat(vec![x, junk], 1), Some(&[5]))
            .slice_dim(1, 0..5);
        padded
            .into_data()
            .assert_approx_eq::<f32>(&alone.into_data(), Tolerance::absolute(1e-4));
    }
}
