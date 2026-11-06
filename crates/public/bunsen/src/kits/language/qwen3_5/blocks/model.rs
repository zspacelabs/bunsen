//! The Qwen3.5 decoder.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::norm::RmsNormConfig,
    prelude::Int,
    tensor::{
        Device,
        FloatDType,
    },
};

use crate::{
    blocks::{
        norm::BunsenNormalizationConfig,
        transformers::{
            attention::csa::{
                CausalSelfAttentionConfig,
                QkNormPlacement,
            },
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::{
                    DecoderStack,
                    DecoderStackMeta,
                    PatternedStackConfig,
                },
            },
            embedding::{
                RotaryDirection,
                RotaryEmbeddingConfig,
                RotarySet,
                TokenEmbedding,
                TokenEmbeddingConfig,
                TokenEmbeddingMeta,
            },
            head::{
                LmHead,
                LmHeadConfig,
            },
            mixers::GatedDeltaNetConfig,
            mlp::TransformerGatedMlpConfig,
        },
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// Common meta for [`Qwen3_5Model`] and [`Qwen3_5Config`].
pub trait Qwen3_5Meta {
    /// The token count: the embedding's rows.
    fn n_vocab(&self) -> usize;

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The full-attention layers' key/value head count.
    fn n_kv_heads(&self) -> usize;

    /// The full-attention layers' head width.
    fn head_dim(&self) -> usize;
}

/// A Qwen3.5 layer's token mixer (`transformers`' `layer_types`:
/// `linear_attention`, `full_attention`).
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3_5LayerKind {
    /// A gated `DeltaNet`: linear attention, with a state in place of a KV
    /// history.
    Linear,

    /// Gated causal self-attention over every earlier position.
    Full,
}

/// Config for [`Qwen3_5Model`]: `transformers`' `Qwen3_5TextConfig`, as far
/// as a decode needs it.
///
/// A hybrid stack: most layers are linear attention (a gated `DeltaNet`), and
/// every few a full attention layer, with QK-norm, a sigmoid gate on its
/// output, and a rotary embedding over a quarter of each head. Every layer
/// has a `SwiGLU`; every norm but the `DeltaNet`'s gated one is a `1 + w` RMS
/// norm (loaded as `1 + w`, the norms applying the gain as stored).
///
/// Implements [`Qwen3_5Meta`].
#[derive(Config, Debug)]
pub struct Qwen3_5Config {
    /// The token count.
    pub n_vocab: usize,

    /// The model width.
    pub d_model: usize,

    /// The layer count.
    pub n_layers: usize,

    /// The full-attention layers' query head count.
    pub n_heads: usize,

    /// The full-attention layers' key/value head count.
    pub n_kv_heads: usize,

    /// The full-attention layers' head width.
    pub head_dim: usize,

    /// The feed-forward hidden width.
    pub d_ff: usize,

    /// The linear-attention layers' query and key heads.
    pub linear_key_heads: usize,

    /// The linear-attention layers' value heads, a multiple of the key heads.
    pub linear_value_heads: usize,

    /// The linear-attention layers' query and key head width.
    pub linear_key_dim: usize,

    /// The linear-attention layers' value head width.
    pub linear_value_dim: usize,

    /// The linear-attention layers' convolution taps.
    #[config(default = "4")]
    pub linear_conv_kernel: usize,

    /// The RMS normalization epsilon, every norm's.
    #[config(default = "1e-6")]
    pub rms_eps: f64,

    /// The rotary embedding's base.
    #[config(default = "10_000_000")]
    pub rope_theta: usize,

    /// The share of each head the rotary embedding rotates, from its first
    /// dimension: `rotary_dim = head_dim · partial_rotary_factor`.
    #[config(default = "0.25")]
    pub partial_rotary_factor: f64,

    /// Whether the full-attention layers gate their output
    /// (`attn_output_gate`).
    #[config(default = "true")]
    pub attn_output_gate: bool,

    /// Each layer's mixer; empty, `transformers`' default, every fourth layer
    /// full ([`every_nth_full`](Self::every_nth_full)).
    #[config(default = "Vec::new()")]
    pub layer_kinds: Vec<Qwen3_5LayerKind>,

    /// Whether the output head is the embedding (`tie_word_embeddings`);
    /// otherwise it is a linear map of its own.
    #[config(default = "false")]
    pub tied_head: bool,

    /// The dtype attention's softmax runs in: f32, as `transformers` runs it.
    #[config(default = "Some(FloatDType::F32)")]
    pub softmax_dtype: Option<FloatDType>,

    /// The dtype the logits leave the model in: f32, as `transformers`
    /// generates from.
    #[config(default = "Some(FloatDType::F32)")]
    pub logits_dtype: Option<FloatDType>,
}

impl Qwen3_5Meta for Qwen3_5Config {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl Qwen3_5Config {
    /// A published dense Qwen3.5's text model: 248320 tokens, full-attention
    /// heads of width 256 over `heads` (query, key/value), 16 linear key
    /// heads and `linear_value_heads` value heads of width 128, every fourth
    /// layer full, a rotary base of 10^7 over a quarter of each head.
    fn published(
        d_model: usize,
        n_layers: usize,
        heads: [usize; 2],
        d_ff: usize,
        linear_value_heads: usize,
        tied_head: bool,
    ) -> Self {
        let [n_heads, n_kv_heads] = heads;
        Self::new(
            248_320,
            d_model,
            n_layers,
            n_heads,
            n_kv_heads,
            256,
            d_ff,
            16,
            linear_value_heads,
            128,
            128,
        )
        .with_layer_kinds(Self::every_nth_full(n_layers, 4))
        .with_tied_head(tied_head)
    }

    /// `Qwen3.5-0.8B`'s geometry: 24 layers of width 1024, 8 query and 2
    /// key/value heads, 16 linear value heads, a tied head.
    pub fn qwen3_5_0_8b() -> Self {
        Self::published(1024, 24, [8, 2], 3584, 16, true)
    }

    /// `Qwen3.5-2B`'s geometry: 24 layers of width 2048, 8 query and 2
    /// key/value heads, 16 linear value heads, a tied head.
    pub fn qwen3_5_2b() -> Self {
        Self::published(2048, 24, [8, 2], 6144, 16, true)
    }

    /// `Qwen3.5-4B`'s geometry: 32 layers of width 2560, 16 query and 4
    /// key/value heads, 32 linear value heads, a tied head.
    pub fn qwen3_5_4b() -> Self {
        Self::published(2560, 32, [16, 4], 9216, 32, true)
    }

    /// `Qwen3.5-9B`'s geometry: 32 layers of width 4096, 16 query and 4
    /// key/value heads, 32 linear value heads, a head of its own.
    pub fn qwen3_5_9b() -> Self {
        Self::published(4096, 32, [16, 4], 12288, 32, false)
    }

    /// `Qwen3.5-27B`'s geometry: 64 layers of width 5120, 24 query and 4
    /// key/value heads, 48 linear value heads, a head of its own.
    pub fn qwen3_5_27b() -> Self {
        Self::published(5120, 64, [24, 4], 17408, 48, false)
    }

    /// `n` layers, every `interval`-th full and the rest linear:
    /// `transformers`' pattern from `full_attention_interval`.
    pub fn every_nth_full(
        n: usize,
        interval: usize,
    ) -> Vec<Qwen3_5LayerKind> {
        (0..n)
            .map(|i| {
                if (i + 1).is_multiple_of(interval) {
                    Qwen3_5LayerKind::Full
                } else {
                    Qwen3_5LayerKind::Linear
                }
            })
            .collect()
    }

    /// Layer `i`'s mixer.
    pub fn layer_kind(
        &self,
        i: usize,
    ) -> Qwen3_5LayerKind {
        match self.layer_kinds.get(i) {
            Some(kind) => *kind,
            None if (i + 1).is_multiple_of(4) => Qwen3_5LayerKind::Full,
            None => Qwen3_5LayerKind::Linear,
        }
    }

    /// The head dimensions the rotary embedding rotates:
    /// `head_dim · partial_rotary_factor`, rounded down as `transformers`
    /// rounds it.
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    /// The token embedding.
    pub fn embedding(&self) -> TokenEmbeddingConfig {
        TokenEmbeddingConfig::new(self.n_vocab, self.d_model)
    }

    /// The output head: tied to the embedding or a projection of its own
    /// (`tied_head`), its logits in `logits_dtype`.
    pub fn head(&self) -> LmHeadConfig {
        LmHeadConfig::new(self.d_model, self.n_vocab)
            .with_tied(self.tied_head)
            .with_logits_dtype(self.logits_dtype)
    }

    /// The rotary embedding: over the first [`rotary_dim`](Self::rotary_dim)
    /// dimensions of each head, in the `rotate_half` convention.
    pub fn rotary(&self) -> RotaryEmbeddingConfig {
        RotaryEmbeddingConfig::new(0, self.head_dim)
            .with_base(self.rope_theta)
            .with_rotary_dim(Some(self.rotary_dim()))
            .with_direction(RotaryDirection::Forward)
    }

    /// A layer of `kind`: its mixer, then a `SwiGLU`, each under an RMS
    /// norm.
    fn layer(
        &self,
        kind: Qwen3_5LayerKind,
    ) -> CausalDecoderBlockConfig {
        let norm =
            |n| BunsenNormalizationConfig::Rms(RmsNormConfig::new(n).with_epsilon(self.rms_eps));
        let mixer = match kind {
            Qwen3_5LayerKind::Full => {
                CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, self.d_model)
                    .with_head_dim(Some(self.head_dim))
                    .with_norm(Some(norm(self.head_dim)))
                    .with_qk_norm(QkNormPlacement::BeforeRope)
                    .with_output_gate(self.attn_output_gate)
                    .with_softmax_dtype(self.softmax_dtype)
                    .into()
            }
            Qwen3_5LayerKind::Linear => GatedDeltaNetConfig::new(
                self.d_model,
                self.linear_key_heads,
                self.linear_value_heads,
                self.linear_key_dim,
                self.linear_value_dim,
            )
            .with_conv_kernel(self.linear_conv_kernel)
            .with_norm_eps(self.rms_eps)
            .into(),
        };
        CausalDecoderBlockConfig::new(
            mixer,
            TransformerGatedMlpConfig::new(self.d_model, self.d_ff).into(),
        )
        .with_norm(norm(0))
    }

    /// The decoder stack: two kinds of layer, full and linear, sharing one
    /// rotary table (the linear layers take none); RMS norms over its output.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for `layer_kinds` of another length than
    /// the layers.
    pub fn stack(&self) -> BunsenResult<PatternedStackConfig> {
        if !self.layer_kinds.is_empty() && self.layer_kinds.len() != self.n_layers {
            return Err(ConstraintError::new(
                "Qwen3_5Config",
                "layer_kinds",
                Rule::Relation {
                    lhs: ("layer_kinds.len".into(), self.layer_kinds.len().to_string()),
                    op: "==",
                    rhs: ("n_layers".into(), self.n_layers.to_string()),
                },
            )
            .into());
        }
        let norm = BunsenNormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(self.rms_eps));
        Ok(PatternedStackConfig::new(
            vec![
                self.layer(Qwen3_5LayerKind::Full),
                self.layer(Qwen3_5LayerKind::Linear),
            ],
            (0..self.n_layers)
                .map(|i| match self.layer_kind(i) {
                    Qwen3_5LayerKind::Full => 0,
                    Qwen3_5LayerKind::Linear => 1,
                })
                .collect(),
            norm,
            self.rotary(),
        ))
    }
}

impl ModuleInit<Qwen3_5Model> for Qwen3_5Config {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3_5Model> {
        Ok(Qwen3_5Model {
            embed_tokens: self.embedding().try_init(device)?,
            lm_head: self.head().try_init(device)?,
            stack: self.stack()?.try_init(device)?,
        })
    }
}

/// The Qwen3.5 decoder-only language model (Qwen team, 2026), the text model
/// of the dense Qwen3.5 checkpoints: a hybrid [`DecoderStack`] of gated
/// `DeltaNet` layers and gated full-attention layers
/// ([`Qwen3_5Config::stack`]), its head tied to the embedding or a linear
/// map of its own.
///
/// Its embedding and head carry the names `transformers`'
/// `Qwen3_5ForCausalLM` gives them;
/// [`transformers_rekey`](crate::kits::language::qwen3_5::weights::transformers_rekey)
/// maps the layers' names onto the stack's, and loads each `1 + w` norm's
/// `w` as `1 + w`.
///
/// It decodes through its stack's
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache),
/// which carries a KV cache for its full-attention layers and a state for
/// each linear one.
///
/// Built by [`Qwen3_5Config`].
/// Implements [`Qwen3_5Meta`].
#[derive(Module, Debug)]
pub struct Qwen3_5Model {
    /// The token embedding, and the head's table when it is tied.
    pub embed_tokens: TokenEmbedding,

    /// The output head ([`Qwen3_5Config::head`]).
    pub lm_head: LmHead,

    /// The decoder: its layers, final norm and rotary table.
    pub stack: DecoderStack,
}

impl Qwen3_5Meta for Qwen3_5Model {
    fn n_vocab(&self) -> usize {
        self.embed_tokens.n_vocab()
    }

    fn d_model(&self) -> usize {
        self.embed_tokens.d_model()
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers()
    }

    fn n_kv_heads(&self) -> usize {
        self.stack.n_kv_heads()
    }

    fn head_dim(&self) -> usize {
        self.stack.head_dim()
    }
}

impl Qwen3_5Model {
    /// The rotary table for positions `0..seq_len`.
    pub fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotarySet {
        self.stack.rotary(seq_len, device)
    }

    /// Embeds tokens.
    ///
    /// # Arguments
    /// * `tokens` - `[batch, seq]`.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`.
    pub fn embed(
        &self,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.embed_tokens.forward(tokens)
    }

    /// Runs the decoder over embeddings at positions `0..seq`, uncached: a
    /// decode runs through
    /// [`StackedLm`](crate::kits::support::decoding::StackedLm).
    ///
    /// # Arguments
    /// * `embeds` - `[batch, seq, d_model]`.
    /// * `r_emb` - rotary tables covering `seq` positions.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, normalized: the head's input.
    pub fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
    ) -> Tensor<3> {
        self.stack.forward_hidden(embeds, r_emb, None)
    }

    /// The head: logits over the vocabulary.
    ///
    /// # Arguments
    /// * `hidden` - `[batch, seq, d_model]`, from
    ///   [`forward_hidden`](Self::forward_hidden).
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]`, in the config's `logits_dtype`.
    pub fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.lm_head.forward(hidden, &self.embed_tokens.weight)
    }
}

impl CanonicalKeys for Qwen3_5Model {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        blocks::transformers::{
            attention::csa::CausalSelfAttentionMeta,
            mixers::{
                GatedDeltaNetMeta,
                TokenMixer,
            },
        },
        support::testing::cpu_device,
    };

    fn config() -> Qwen3_5Config {
        Qwen3_5Config::new(40, 16, 4, 4, 2, 8, 24, 2, 4, 4, 3)
            .with_partial_rotary_factor(0.5)
            .with_layer_kinds(vec![
                Qwen3_5LayerKind::Linear,
                Qwen3_5LayerKind::Full,
                Qwen3_5LayerKind::Linear,
                Qwen3_5LayerKind::Full,
            ])
    }

    /// The config's choices reach the stack: each layer's mixer, the gated
    /// attention, the partial rotary, the `DeltaNet`'s geometry; the model
    /// answers the meta as the config does.
    #[test]
    fn test_config_and_meta() {
        let device = cpu_device();
        let config = config();
        assert_eq!(config.rotary_dim(), 4);
        let model: Qwen3_5Model = config.init(&device);
        assert_eq!(model.n_vocab(), 40);
        assert_eq!(model.d_model(), 16);
        assert_eq!(model.n_layers(), 4);
        assert_eq!(model.n_kv_heads(), 2);
        assert_eq!(model.head_dim(), 8);
        assert_eq!(
            model.stack.layer_attends(),
            [false, true, false, true],
            "the layer kinds"
        );
        match &model.stack.layers[1].attn.branch {
            TokenMixer::Attention(attn) => {
                assert!(attn.output_gate && attn.q_norm.is_some());
                assert_eq!(attn.n_head(), 4);
            }
            other => panic!("a full layer attends: {other:?}"),
        }
        match &model.stack.layers[0].attn.branch {
            TokenMixer::GatedDeltaNet(net) => {
                assert_eq!(
                    (
                        net.n_key_heads(),
                        net.n_value_heads(),
                        net.key_dim(),
                        net.value_dim()
                    ),
                    (2, 4, 4, 3)
                );
            }
            other => panic!("a linear layer is a DeltaNet: {other:?}"),
        }
        assert_eq!(model.rotary(5, &device).tables[0].cos.dims(), [1, 5, 1, 2]);

        let default = Qwen3_5Config::new(40, 16, 8, 4, 2, 8, 24, 2, 4, 4, 3);
        assert_eq!(
            (0..8).map(|i| default.layer_kind(i)).collect::<Vec<_>>(),
            Qwen3_5Config::every_nth_full(8, 4),
            "every fourth layer full by default"
        );
        assert!(
            config
                .with_layer_kinds(vec![Qwen3_5LayerKind::Full])
                .stack()
                .is_err()
        );
    }
}
