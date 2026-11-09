//! The Gemma decoder.
use burn::{
    config::Config,
    module::Module,
    nn::norm::RmsNormConfig,
    tensor::FloatDType,
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        norm::BunsenNormalizationConfig,
        transformers::{
            attention::csa::{
                CausalSelfAttentionConfig,
                QkNormPlacement,
            },
            decoders::{
                causal::CausalDecoderBlockConfig,
                stack::PatternedStackConfig,
            },
            embedding::{
                RopeScaling,
                RotaryDirection,
                RotaryEmbeddingConfig,
                TokenEmbeddingConfig,
            },
            head::LmHeadConfig,
            mlp::TransformerGatedMlpConfig,
            models::language_decoder::{
                LmAttentionOnlyKind,
                LmDecoderMeta,
                LmDecoderModel,
                LmDecoderVariantConfig,
                LmDecoderWrapper,
            },
        },
    },
    burner::module::ToStructureConfig,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// The positions a Gemma cache allocates up front by default: one growth
/// chunk.
pub const GEMMA_KV_CACHE_PRE_SIZE: usize = 1024;

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
/// Implements [`LmDecoderMeta`].
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

impl LmDecoderMeta for GemmaConfig {
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
    /// `n` layers, every `full_every`-th full and the rest sliding:
    /// `transformers`' pattern, from layer `full_every - 1`.
    pub fn every_nth_full(
        n: usize,
        full_every: usize,
    ) -> Vec<GemmaLayerKind> {
        (0..n)
            .map(|i| {
                if (i + 1) % full_every == 0 {
                    GemmaLayerKind::Full
                } else {
                    GemmaLayerKind::Sliding
                }
            })
            .collect()
    }

    /// A Gemma 3: the sandwich, QK-norm, a `query_pre_attn_scalar` of 256,
    /// five sliding layers to one full one over a window of `window`, the
    /// local table at 10^4 and the global one at 10^6.
    fn gemma3(
        n_vocab: usize,
        d_model: usize,
        n_layers: usize,
        heads: [usize; 3],
        d_ff: usize,
        window: usize,
    ) -> Self {
        let [n_heads, n_kv_heads, head_dim] = heads;
        Self::new(
            n_vocab, d_model, n_layers, n_heads, n_kv_heads, head_dim, d_ff,
        )
        .with_post_norms(true)
        .with_qk_norm(true)
        .with_query_pre_attn_scalar(Some(256.0))
        .with_layer_kinds(Self::every_nth_full(n_layers, 6))
        .with_sliding_window(Some(window))
        .with_rope_theta(1_000_000.0)
        .with_rope_local_theta(Some(10_000.0))
    }

    /// A Gemma 2: the sandwich, a `query_pre_attn_scalar` of 256, sliding
    /// and full layers alternating over a window of 4096, one table at
    /// 10^4, soft-caps of 50 on the scores and 30 on the logits.
    fn gemma2(
        d_model: usize,
        n_layers: usize,
        heads: [usize; 3],
        d_ff: usize,
    ) -> Self {
        let [n_heads, n_kv_heads, head_dim] = heads;
        Self::new(
            256_000, d_model, n_layers, n_heads, n_kv_heads, head_dim, d_ff,
        )
        .with_post_norms(true)
        .with_query_pre_attn_scalar(Some(256.0))
        .with_layer_kinds(Self::every_nth_full(n_layers, 2))
        .with_sliding_window(Some(4096))
        .with_attn_softcap(Some(50.0))
        .with_final_softcap(Some(30.0))
    }

    /// `gemma-3-270m`'s geometry: 18 layers of width 640, 4 query heads and
    /// one key/value head of width 256, a window of 512.
    pub fn gemma_3_270m() -> Self {
        Self::gemma3(262_144, 640, 18, [4, 1, 256], 2048, 512)
    }

    /// `gemma-3-1b`'s geometry: 26 layers of width 1152, 4 query heads and
    /// one key/value head of width 256, a window of 512.
    pub fn gemma_3_1b() -> Self {
        Self::gemma3(262_144, 1152, 26, [4, 1, 256], 6912, 512)
    }

    /// `gemma-3-4b`'s text model: 34 layers of width 2560, 8 query and 4
    /// key/value heads of width 256, a window of 1024, its global table
    /// scaled linearly by 8, and 262208 tokens (the image tokens' rows
    /// among them).
    pub fn gemma_3_4b() -> Self {
        Self::gemma3(262_208, 2560, 34, [8, 4, 256], 10240, 1024)
            .with_rope_scaling(RopeScaling::Linear { factor: 8.0 })
    }

    /// `gemma-2-2b`'s geometry: 26 layers of width 2304, 8 query and 4
    /// key/value heads of width 256.
    pub fn gemma_2_2b() -> Self {
        Self::gemma2(2304, 26, [8, 4, 256], 9216)
    }

    /// `gemma-2-9b`'s geometry: 42 layers of width 3584, 16 query and 8
    /// key/value heads of width 256.
    pub fn gemma_2_9b() -> Self {
        Self::gemma2(3584, 42, [16, 8, 256], 14336)
    }

    /// `gemma-2b`'s geometry (Gemma 1): 18 layers of width 2048, 8 query
    /// heads and one key/value head of width 256.
    pub fn gemma_2b() -> Self {
        Self::new(256_000, 2048, 18, 8, 1, 256, 16384)
    }

    /// `gemma-7b`'s geometry (Gemma 1): 28 layers of width 3072, 16 heads of
    /// width 256.
    pub fn gemma_7b() -> Self {
        Self::new(256_000, 3072, 28, 16, 16, 256, 24576)
    }

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
        let norm =
            |n| BunsenNormalizationConfig::Rms(RmsNormConfig::new(n).with_epsilon(self.rms_eps));
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
                .with_softmax_dtype(self.softmax_dtype)
                .into(),
            TransformerGatedMlpConfig::new(self.d_model, self.d_ff)
                .with_activation(BunsenActivationConfig::GeluApproximate)
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
        let norm = BunsenNormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(self.rms_eps));
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

/// The language model: [`embedding`](GemmaConfig::embedding),
/// [`head`](GemmaConfig::head) and [`stack`](GemmaConfig::stack), its KV
/// cache [`GEMMA_KV_CACHE_PRE_SIZE`] positions up front.
///
/// # Errors
/// As [`stack`](GemmaConfig::stack).
impl ToStructureConfig for GemmaConfig {
    type Structure = LmDecoderVariantConfig;

    fn try_to_structure(&self) -> BunsenResult<LmDecoderVariantConfig> {
        Ok(
            LmDecoderVariantConfig::new(self.embedding(), self.head(), self.stack()?.into())
                .with_kv_cache_pre_size(GEMMA_KV_CACHE_PRE_SIZE),
        )
    }
}

/// Google's Gemma decoder-only language model, generations 1 to 3 (text):
/// an [`LmDecoderModel`] of `GeGLU` layers with `1 + w` RMS norms, sliding
/// and full attention by layer ([`GemmaConfig::stack`]); the embedding scaled
/// by `√d_model`; the head tied to it.
///
/// [`transformers_rekey`](crate::kits::language::gemma::weights::transformers_rekey)
/// maps `transformers`' `GemmaModel` names onto the [`LmDecoderModel`]'s, and
/// loads each norm's `w` as `1 + w`; the load is into [`lm`](Self::lm).
///
/// An [`LmAttentionOnlyKind`] [`LmDecoderWrapper`]: it decodes through its
/// stack's
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache)
/// ([`StackedLm`](crate::kits::support::decoding::StackedLm)), or a flat
/// [`KVCache`](crate::ops::transformers::attention::KVCache)
/// ([`LmKvForwardHidden`](crate::blocks::transformers::models::language_decoder::LmKvForwardHidden),
/// [`KVCacheModel`](crate::ops::transformers::attention::KVCacheModel)).
///
/// Built by [`GemmaConfig`].
/// Implements [`LmDecoderMeta`].
#[derive(Module, Debug)]
pub struct GemmaModel {
    /// The language model.
    pub lm: LmDecoderModel,
}

impl LmDecoderWrapper for GemmaModel {
    type Kind = LmAttentionOnlyKind;

    fn lm(&self) -> &LmDecoderModel {
        &self.lm
    }

    fn lm_mut(&mut self) -> &mut LmDecoderModel {
        &mut self.lm
    }

    fn from_lm(lm: LmDecoderModel) -> Self {
        Self { lm }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        ops::transformers::attention::KVCacheModel,
        support::testing::cpu_device,
    };

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
        assert_eq!(stack.kinds[0].attn.attention().sliding_window, None);
        assert_eq!(stack.kinds[1].attn.attention().sliding_window, Some(4));
        assert_eq!(stack.kinds[0].attn.attention().score_scale, Some(0.25));
        assert!(stack.kinds[0].attn.attention().norm.is_some() && stack.kinds[0].post_norms);
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
        assert_eq!(model.lm.embedding.scale, Some(12f64.sqrt()));
        assert!(model.lm.head.proj.is_none(), "tied");
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
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("sliding_window")
            .assert_err(&config().with_sliding_window(None).try_to_structure());
    }
}
