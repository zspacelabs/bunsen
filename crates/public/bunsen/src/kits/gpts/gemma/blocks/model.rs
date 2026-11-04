//! The Gemma decoder.
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
        activation::ActivationConfig,
        norm::NormConfig,
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
                RopeScaling,
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
            mlp::GatedMlpConfig,
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
};

/// The positions a Gemma cache allocates up front by default: one growth
/// chunk.
pub const GEMMA_KV_CACHE_PRE_SIZE: usize = 1024;

/// Common meta for [`GemmaModel`] and [`GemmaConfig`].
pub trait GemmaMeta {
    /// The token count: the embedding's rows.
    fn n_vocab(&self) -> usize;

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The key/value head count.
    fn n_kv_heads(&self) -> usize;

    /// The head width.
    fn head_dim(&self) -> usize;
}

/// A Gemma layer's attention: over a sliding window, or every earlier
/// position (`transformers`' `layer_types`: `sliding_attention`,
/// `full_attention`).
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum GemmaLayerKind {
    /// Attention over the last `sliding_window` positions, with the local
    /// rotary table where the model has one.
    Sliding,

    /// Attention over every earlier position, with the global rotary table.
    Full,
}

/// Config for [`GemmaModel`]: `transformers`' `GemmaConfig`,
/// `Gemma2Config` and `Gemma3TextConfig`, as far as a decode needs them.
/// One config covers the three generations; their choices are its fields:
///
/// - Gemma 1: pre-norms only, one rotary table, no window, no caps;
/// - Gemma 2: [`post_norms`](Self::post_norms) (the sandwich), a sliding window
///   on alternate layers, the attention and final soft-caps, and a
///   `query_pre_attn_scalar`;
/// - Gemma 3: Gemma 2's sandwich and window (five sliding layers to one full
///   one), QK-norm, and a local rotary table for the sliding layers beside the
///   global one, which alone a linear factor scales.
///
/// Every generation: `1 + w` RMS norms (loaded as `1 + w`, the norms applying
/// the gain as stored), a `GeGLU`, the embedding scaled by `√d_model`, the
/// head tied to it, attention's softmax and the logits in f32.
///
/// Implements [`GemmaMeta`].
#[derive(Config, Debug)]
pub struct GemmaConfig {
    /// The token count.
    pub n_vocab: usize,

    /// The model width.
    pub d_model: usize,

    /// The layer count.
    pub n_layers: usize,

    /// The query head count.
    pub n_heads: usize,

    /// The key/value head count.
    pub n_kv_heads: usize,

    /// The head width.
    pub head_dim: usize,

    /// The feed-forward hidden width.
    pub d_ff: usize,

    /// The RMS normalization epsilon.
    #[config(default = "1e-6")]
    pub rms_eps: f64,

    /// The scores are scaled by `query_pre_attn_scalar^-0.5`; `None`, by the
    /// head width's.
    #[config(default = "None")]
    pub query_pre_attn_scalar: Option<f64>,

    /// Whether each sublayer's output is normalized before the residual add
    /// (Gemma 2 and 3).
    #[config(default = "false")]
    pub post_norms: bool,

    /// Whether the queries and keys are RMS normalized, over the head width,
    /// before the rotary embedding (Gemma 3).
    #[config(default = "false")]
    pub qk_norm: bool,

    /// Each layer's attention kind; empty, every layer full.
    #[config(default = "Vec::new()")]
    pub layer_kinds: Vec<GemmaLayerKind>,

    /// The sliding layers' window.
    #[config(default = "None")]
    pub sliding_window: Option<usize>,

    /// The global rotary table's base: the full layers', and every layer's
    /// without a local one.
    #[config(default = "10000.0")]
    pub rope_theta: f64,

    /// The global rotary table's scaling.
    #[config(default = "RopeScaling::None")]
    pub rope_scaling: RopeScaling,

    /// The local rotary table's base, the sliding layers' (Gemma 3), never
    /// scaled; `None`, they take the global table.
    #[config(default = "None")]
    pub rope_local_theta: Option<f64>,

    /// The soft-cap on the attention scores (Gemma 2's 50).
    #[config(default = "None")]
    pub attn_softcap: Option<f64>,

    /// The soft-cap on the logits (Gemma 2's 30).
    #[config(default = "None")]
    pub final_softcap: Option<f64>,

    /// The dtype attention's softmax runs in: f32, as `transformers` runs it.
    #[config(default = "Some(FloatDType::F32)")]
    pub softmax_dtype: Option<FloatDType>,

    /// The dtype the logits leave the model in: f32, as `transformers`
    /// generates from.
    #[config(default = "Some(FloatDType::F32)")]
    pub logits_dtype: Option<FloatDType>,
}

impl GemmaMeta for GemmaConfig {
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

impl GemmaConfig {
    /// Layer `i`'s attention kind.
    pub fn layer_kind(
        &self,
        i: usize,
    ) -> GemmaLayerKind {
        self.layer_kinds
            .get(i)
            .copied()
            .unwrap_or(GemmaLayerKind::Full)
    }

    /// The token embedding: scaled by `√d_model`.
    pub fn embedding(&self) -> TokenEmbeddingConfig {
        TokenEmbeddingConfig::new(self.n_vocab, self.d_model)
            .with_scale(Some((self.d_model as f64).sqrt()))
    }

    /// The head: tied to the embedding, its logits soft-capped by
    /// `final_softcap` and in `logits_dtype`.
    pub fn head(&self) -> LmHeadConfig {
        LmHeadConfig::new(self.d_model, self.n_vocab)
            .with_softcap(self.final_softcap)
            .with_logits_dtype(self.logits_dtype)
    }

    /// A layer of `kind`.
    fn layer(
        &self,
        kind: GemmaLayerKind,
    ) -> CausalDecoderBlockConfig {
        let norm = |n| NormConfig::Rms(RmsNormConfig::new(n).with_epsilon(self.rms_eps));
        let window = match kind {
            GemmaLayerKind::Sliding => self.sliding_window,
            GemmaLayerKind::Full => None,
        };
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(self.n_heads, self.n_kv_heads, self.d_model)
                .with_head_dim(Some(self.head_dim))
                .with_norm(self.qk_norm.then(|| norm(self.head_dim)))
                .with_qk_norm(QkNormPlacement::BeforeRope)
                .with_sliding_window(window)
                .with_score_scale(self.query_pre_attn_scalar.map(|s| s.powf(-0.5)))
                .with_attn_softcap(self.attn_softcap)
                .with_softmax_dtype(self.softmax_dtype),
            GatedMlpConfig::new(self.d_model, self.d_ff)
                .with_activation(ActivationConfig::GeluApproximate)
                .into(),
        )
        .with_norm(norm(0))
        .with_post_norms(self.post_norms)
    }

    /// A rotary table of base `theta`, scaled by `scaling`, in the
    /// `rotate_half` convention.
    fn rotary(
        &self,
        theta: f64,
        scaling: RopeScaling,
    ) -> RotaryEmbeddingConfig {
        RotaryEmbeddingConfig::new(0, self.head_dim)
            .with_base(theta as usize)
            .with_scaling(scaling)
            .with_direction(RotaryDirection::Forward)
    }

    /// The decoder stack: two kinds of layer, sliding and full, each with
    /// its rotary table; RMS norms over the stack's output.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for `layer_kinds` of another length than
    /// the layers, or a sliding layer without a window.
    pub fn stack(&self) -> BunsenResult<PatternedStackConfig> {
        if !self.layer_kinds.is_empty() && self.layer_kinds.len() != self.n_layers {
            return Err(ConstraintError::new(
                "GemmaConfig",
                "layer_kinds",
                Rule::Relation {
                    lhs: ("layer_kinds.len".into(), self.layer_kinds.len().to_string()),
                    op: "==",
                    rhs: ("n_layers".into(), self.n_layers.to_string()),
                },
            )
            .into());
        }
        let sliding = self.layer_kinds.contains(&GemmaLayerKind::Sliding);
        if sliding && self.sliding_window.is_none() {
            return Err(ConstraintError::zero_or_empty("GemmaConfig", "sliding_window").into());
        }
        let global = self.rotary(self.rope_theta, self.rope_scaling);
        let (alts, sliding_rotary) = match self.rope_local_theta {
            Some(theta) => (vec![self.rotary(theta, RopeScaling::None)], Some(0)),
            None => (Vec::new(), None),
        };
        let norm = NormConfig::Rms(RmsNormConfig::new(0).with_epsilon(self.rms_eps));
        Ok(PatternedStackConfig::new(
            vec![
                self.layer(GemmaLayerKind::Full),
                self.layer(GemmaLayerKind::Sliding),
            ],
            (0..self.n_layers)
                .map(|i| match self.layer_kind(i) {
                    GemmaLayerKind::Full => 0,
                    GemmaLayerKind::Sliding => 1,
                })
                .collect(),
            norm,
            global,
        )
        .with_rotary_alts(alts)
        .with_kind_rotary(vec![None, sliding_rotary]))
    }
}

impl ModuleInit<GemmaModel> for GemmaConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<GemmaModel> {
        Ok(GemmaModel {
            embed_tokens: self.embedding().try_init(device)?,
            lm_head: self.head().try_init(device)?,
            stack: self.stack()?.try_init(device)?,
        })
    }
}

/// Google's Gemma decoder-only language model, generations 1 to 3 (text):
/// a [`DecoderStack`] of `GeGLU` layers with `1 + w` RMS norms, sliding and
/// full attention by layer ([`GemmaConfig::stack`]); the embedding scaled by
/// `√d_model`; the head tied to it.
///
/// Its embedding carries the name `transformers`' `GemmaModel` gives it;
/// [`hf_rekey`](crate::kits::gpts::gemma::weights::hf_rekey) maps the
/// layers' names onto the stack's, and loads each norm's `w` as `1 + w`.
///
/// Built by [`GemmaConfig`].
/// Implements [`GemmaMeta`].
#[derive(Module, Debug)]
pub struct GemmaModel {
    /// The token embedding, scaled; and the head's table.
    pub embed_tokens: TokenEmbedding,

    /// The output head, tied to the embedding.
    pub lm_head: LmHead,

    /// The decoder: its layers, final norm and rotary tables.
    pub stack: DecoderStack,
}

impl GemmaMeta for GemmaModel {
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

impl KVCacheModel for GemmaModel {
    fn kv_cache_config(&self) -> KVCacheConfig {
        self.stack.kv_cache_config(GEMMA_KV_CACHE_PRE_SIZE)
    }
}

impl GemmaModel {
    /// The rotary tables for positions `0..seq_len`.
    pub fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotarySet {
        self.stack.rotary(seq_len, device)
    }

    /// Embeds tokens, scaled.
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

    /// Runs the decoder over embeddings at positions `start..start + seq`,
    /// through the cache when there is one.
    ///
    /// # Arguments
    /// * `embeds` - `[batch, seq, d_model]`.
    /// * `r_emb` - rotary tables covering `start + seq` positions.
    /// * `kv_cache` - the cache, at position `start`; `None` means `start` is 0
    ///   and nothing is kept.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, normalized: the head's input.
    pub fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.stack.forward_hidden(embeds, r_emb, kv_cache)
    }

    /// The head: logits over the vocabulary.
    ///
    /// # Arguments
    /// * `hidden` - `[batch, seq, d_model]`, from
    ///   [`forward_hidden`](Self::forward_hidden).
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]`, soft-capped as configured, in the config's
    /// `logits_dtype`.
    pub fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.lm_head.forward(hidden, &self.embed_tokens.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    fn config() -> GemmaConfig {
        GemmaConfig::new(40, 12, 3, 2, 1, 8, 24)
            .with_post_norms(true)
            .with_qk_norm(true)
            .with_layer_kinds(vec![
                GemmaLayerKind::Sliding,
                GemmaLayerKind::Full,
                GemmaLayerKind::Sliding,
            ])
            .with_sliding_window(Some(4))
            .with_rope_theta(1_000_000.0)
            .with_rope_scaling(RopeScaling::Linear { factor: 2.0 })
            .with_rope_local_theta(Some(10_000.0))
            .with_query_pre_attn_scalar(Some(16.0))
    }

    /// The config's choices reach the stack: the kinds' windows and tables,
    /// the score scale, the norms; the model answers the meta as the config
    /// does.
    #[test]
    fn test_config_and_meta() {
        let config = config();
        let stack = config.stack().unwrap();
        assert_eq!(stack.layer_kinds, [1, 0, 1]);
        assert_eq!(stack.kinds[0].attn.sliding_window, None);
        assert_eq!(stack.kinds[1].attn.sliding_window, Some(4));
        assert_eq!(stack.kinds[0].attn.score_scale, Some(0.25));
        assert!(stack.kinds[0].attn.norm.is_some() && stack.kinds[0].post_norms);
        assert_eq!(stack.rotary.base, 1_000_000);
        assert_eq!(stack.rotary.scaling, RopeScaling::Linear { factor: 2.0 });
        assert_eq!(stack.rotary_alts[0].base, 10_000);
        assert_eq!(stack.kind_rotary, [None, Some(0)]);

        let device = cpu_device();
        let model: GemmaModel = config.init(&device);
        assert_eq!(
            (
                model.n_vocab(),
                model.d_model(),
                model.n_layers(),
                model.n_kv_heads(),
                model.head_dim()
            ),
            (40, 12, 3, 1, 8)
        );
        assert_eq!(model.embed_tokens.scale, Some(12f64.sqrt()));
        assert!(model.lm_head.proj.is_none(), "tied");
        assert_eq!(
            model.kv_cache_config().layer_windows,
            [Some(4), None, Some(4)]
        );
    }

    /// A config whose kinds do not cover the layers, or whose sliding layers
    /// have no window, is refused.
    #[test]
    fn test_refusals() {
        use crate::errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        };
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("layer_kinds")
            .assert_err(
                &config()
                    .with_layer_kinds(vec![GemmaLayerKind::Full])
                    .stack(),
            );
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("sliding_window")
            .assert_err(&config().with_sliding_window(None).stack());
    }
}
