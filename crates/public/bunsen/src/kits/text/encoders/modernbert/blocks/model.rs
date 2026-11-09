//! The `ModernBERT` encoder.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::LayerNormConfig,
    prelude::Int,
    tensor::Device,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        norm::BunsenNormalizationConfig,
        transformers::{
            attention::csa::{
                AttentionMask,
                CausalSelfAttentionConfig,
            },
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::{
                    DecoderStack,
                    DecoderStackMeta,
                    DecoderStackStructureConfig,
                    PatternedStackConfig,
                },
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
                TokenEmbedding,
                TokenEmbeddingConfig,
                TokenEmbeddingMeta,
            },
            mlp::TransformerGatedMlpConfig,
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
        ToStructureConfig,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// Common meta for [`ModernBertModel`] and [`ModernBertConfig`].
pub trait ModernBertMeta {
    /// The token count: the embedding's rows.
    fn n_vocab(&self) -> usize;

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;
}

/// A `ModernBERT` layer's attention: over every position, or a two-sided
/// window about the query (`transformers`' `layer_types`:
/// `full_attention`, `sliding_attention`).
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum ModernBertLayerKind {
    /// Attention over every position, with the global rotary base.
    Global,

    /// Attention over the positions within
    /// [`local_radius`](ModernBertConfig::local_radius) of the query, on
    /// either side, with the local rotary base.
    Local,
}

/// Config for [`ModernBertModel`]: `transformers`' `ModernBertConfig`, as
/// far as the encoder's forward needs it.
///
/// Implements [`ModernBertMeta`].
#[derive(Config, Debug)]
pub struct ModernBertConfig {
    /// The token count.
    pub n_vocab: usize,

    /// The model width.
    pub d_model: usize,

    /// The layer count.
    pub n_layers: usize,

    /// The attention head count; each head is `d_model / n_heads` wide.
    pub n_heads: usize,

    /// The feed-forward hidden width: each half of the fused `Wi`.
    pub d_ff: usize,

    /// The `LayerNorm` epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,

    /// Each layer's attention kind; empty, `transformers`' pattern: every
    /// third layer global, from the first.
    #[config(default = "Vec::new()")]
    pub layer_kinds: Vec<ModernBertLayerKind>,

    /// How far a local layer's query sees on either side, in positions:
    /// `local_attention / 2` (64 of 128).
    #[config(default = "64")]
    pub local_radius: usize,

    /// The global layers' rotary base.
    #[config(default = "160000.0")]
    pub global_rope_theta: f64,

    /// The local layers' rotary base.
    #[config(default = "10000.0")]
    pub local_rope_theta: f64,
}

impl ModernBertMeta for ModernBertConfig {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }
}

impl ModernBertConfig {
    /// `ModernBERT-base`'s geometry: 22 layers of width 768, 12 heads, a
    /// hidden width of 1152, 50368 tokens.
    pub fn modernbert_base() -> Self {
        Self::new(50368, 768, 22, 12, 1152)
    }

    /// `ModernBERT-large`'s geometry: 28 layers of width 1024, 16 heads, a
    /// hidden width of 2624, 50368 tokens.
    pub fn modernbert_large() -> Self {
        Self::new(50368, 1024, 28, 16, 2624)
    }

    /// Layer `i`'s attention kind.
    pub fn layer_kind(
        &self,
        i: usize,
    ) -> ModernBertLayerKind {
        match self.layer_kinds.get(i) {
            Some(&kind) => kind,
            None if self.layer_kinds.is_empty() && !i.is_multiple_of(3) => {
                ModernBertLayerKind::Local
            }
            None => ModernBertLayerKind::Global,
        }
    }

    /// A `LayerNorm` without a bias, its width set where it is used.
    fn norm(&self) -> BunsenNormalizationConfig {
        BunsenNormalizationConfig::Layer(
            LayerNormConfig::new(0)
                .with_epsilon(self.norm_eps)
                .with_bias(false),
        )
    }

    /// The token embedding, its rows normalized.
    pub fn embedding(&self) -> TokenEmbeddingConfig {
        TokenEmbeddingConfig::new(self.n_vocab, self.d_model).with_norm(Some(self.norm()))
    }

    /// A layer of `kind`, with or without its attention norm.
    fn layer(
        &self,
        kind: ModernBertLayerKind,
        attn_norm: bool,
    ) -> CausalDecoderBlockConfig {
        let window = match kind {
            ModernBertLayerKind::Global => None,
            ModernBertLayerKind::Local => Some(self.local_radius + 1),
        };
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_heads, self.n_heads, self.d_model)
                .with_norm(None)
                .with_mask(AttentionMask::Bidirectional)
                .with_sliding_window(window)
                .into(),
            TransformerGatedMlpConfig::new(self.d_model, self.d_ff)
                .with_activation(BunsenActivationConfig::Gelu)
                .into(),
        )
        .with_norm(self.norm())
        .with_attn_norm(attn_norm)
    }

    /// A rotary table of base `theta`, in the `rotate_half` convention.
    fn rotary(
        &self,
        theta: f64,
    ) -> RotaryEmbeddingConfig {
        RotaryEmbeddingConfig::new(0, self.d_model / self.n_heads)
            .with_base(theta as usize)
            .with_direction(RotaryDirection::Forward)
    }

    /// The encoder stack: three kinds of layer (the first, global and
    /// without an attention norm; global; local), the local kind with its
    /// rotary table; a `LayerNorm` over the stack's output.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for `layer_kinds` of another length than
    /// the layers, or heads that do not divide the width.
    pub fn stack(&self) -> BunsenResult<PatternedStackConfig> {
        if !self.layer_kinds.is_empty() && self.layer_kinds.len() != self.n_layers {
            return Err(ConstraintError::new(
                "ModernBertConfig",
                "layer_kinds",
                Rule::Relation {
                    lhs: ("layer_kinds.len".into(), self.layer_kinds.len().to_string()),
                    op: "==",
                    rhs: ("n_layers".into(), self.n_layers.to_string()),
                },
            )
            .into());
        }
        if self.n_heads == 0 || !self.d_model.is_multiple_of(self.n_heads) {
            return Err(ConstraintError::new(
                "ModernBertConfig",
                "n_heads",
                Rule::Relation {
                    lhs: ("n_heads".into(), self.n_heads.to_string()),
                    op: "divides",
                    rhs: ("d_model".into(), self.d_model.to_string()),
                },
            )
            .into());
        }
        // The first layer's kind, then each kind with its attention norm.
        let first = self.layer_kind(0);
        let kinds = vec![
            self.layer(first, false),
            self.layer(ModernBertLayerKind::Global, true),
            self.layer(ModernBertLayerKind::Local, true),
        ];
        let layer_kinds = (0..self.n_layers)
            .map(|i| match (i, self.layer_kind(i)) {
                (0, _) => 0,
                (_, ModernBertLayerKind::Global) => 1,
                (_, ModernBertLayerKind::Local) => 2,
            })
            .collect();
        let local = Some(0);
        Ok(PatternedStackConfig::new(
            kinds,
            layer_kinds,
            self.norm(),
            self.rotary(self.global_rope_theta),
        )
        .with_rotary_alts(vec![self.rotary(self.local_rope_theta)])
        .with_kind_rotary(vec![
            (first == ModernBertLayerKind::Local).then_some(0),
            None,
            local,
        ]))
    }
}

impl ToStructureConfig for ModernBertConfig {
    type Structure = ModernBertStructureConfig;

    /// # Errors
    /// As [`stack`](Self::stack) and its lowering.
    fn try_to_structure(&self) -> BunsenResult<ModernBertStructureConfig> {
        Ok(ModernBertStructureConfig::new(
            self.embedding(),
            self.stack()?.try_to_structure()?,
        ))
    }
}

/// Structure config for [`ModernBertModel`]: its token embedding and its
/// encoder stack, unrolled. [`ModernBertConfig`] lowers to it.
#[derive(Config, Debug)]
pub struct ModernBertStructureConfig {
    /// The token embedding, its rows normalized.
    pub embedding: TokenEmbeddingConfig,

    /// The encoder: its layers, final norm and rotary tables.
    pub stack: DecoderStackStructureConfig,
}

impl ModuleInit<ModernBertModel> for ModernBertStructureConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ModernBertModel> {
        Ok(ModernBertModel {
            embeddings: self.embedding.try_init(device)?,
            stack: self.stack.try_init(device)?,
        })
    }
}

/// Answer.AI's `ModernBERT` encoder: a normalized token embedding, then a
/// [`DecoderStack`] of bidirectional `GeGLU` layers, global and local by
/// layer ([`ModernBertConfig::stack`]), and a final `LayerNorm`.
///
/// [`transformers_rekey`](crate::kits::text::encoders::modernbert::weights::transformers_rekey)
/// maps a `transformers` state dict's names onto it, after
/// [`transformers_row_splits`](crate::kits::text::encoders::modernbert::weights::transformers_row_splits)
/// has split its fused projections.
///
/// Built by [`ModernBertConfig`] (high-level) or
/// [`ModernBertStructureConfig`].
/// Implements [`ModernBertMeta`].
#[derive(Module, Debug)]
pub struct ModernBertModel {
    /// The token embedding, its rows normalized.
    pub embeddings: TokenEmbedding,

    /// The encoder: its layers, final norm and rotary tables.
    pub stack: DecoderStack,
}

impl ModernBertMeta for ModernBertModel {
    fn n_vocab(&self) -> usize {
        self.embeddings.n_vocab()
    }

    fn d_model(&self) -> usize {
        self.embeddings.d_model()
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers()
    }
}

impl ModernBertModel {
    /// The final hidden states: the tokens embedded, then encoded, each
    /// item's padding keys masked by its length.
    ///
    /// # Arguments
    /// * `ids` - `[batch, seq]`.
    /// * `lengths` - each item's tokens, the rest padding; `None`, no padding.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, normalized. A padding position's state means
    /// nothing.
    pub fn forward(
        &self,
        ids: Tensor<2, Int>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        self.stack
            .encoder_forward(self.embeddings.forward(ids), lengths)
    }
}

impl CanonicalKeys for ModernBertModel {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// `transformers`' default pattern: every third layer global, from the
    /// first; the first layer alone without an attention norm; the local
    /// layers windowed `local_radius + 1` (two-sided) with the local table.
    #[test]
    fn test_layer_pattern() {
        let config = ModernBertConfig::new(32, 16, 5, 2, 24).with_local_radius(3);
        let kinds: Vec<_> = (0..5).map(|i| config.layer_kind(i)).collect();
        use ModernBertLayerKind::*;
        assert_eq!(kinds, [Global, Local, Local, Global, Local]);

        let model: ModernBertModel = config.init(&cpu_device());
        let windows = model.stack.layer_windows();
        assert_eq!(windows, [None, Some(4), Some(4), None, Some(4)]);
        let norms: Vec<bool> = model
            .stack
            .layers
            .iter()
            .map(|l| l.attn.pre.is_some())
            .collect();
        assert_eq!(norms, [false, true, true, true, true]);
        assert_eq!(
            model.stack.layer_rotary,
            [None, Some(0), Some(0), None, Some(0)]
        );
        assert!(model.embeddings.norm.is_some());
    }

    #[test]
    fn test_layer_kinds_must_cover_the_layers() {
        let config = ModernBertConfig::new(32, 16, 3, 2, 24)
            .with_layer_kinds(vec![ModernBertLayerKind::Global]);
        assert!(config.stack().is_err());
        assert!(
            ModernBertConfig::new(32, 16, 3, 3, 24).stack().is_err(),
            "3 heads over 16"
        );
    }
}
