//! The generic decoder-only language model.
use alloc::{
    format,
    string::String,
    vec::Vec,
};
use core::{
    fmt::Debug,
    marker::PhantomData,
    ops::Range,
};

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::Int,
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        decoders::stack::{
            DecoderStack,
            DecoderStackMeta,
            DecoderStackStructureConfig,
            DecoderStackVariantConfig,
        },
        embedding::{
            RotarySet,
            TokenEmbedding,
            TokenEmbeddingConfig,
            TokenEmbeddingMeta,
        },
        head::{
            LmHead,
            LmHeadConfig,
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
        ResultContext,
    },
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
};

/// The positions an [`LmDecoderModel`]'s KV cache allocates up front by
/// default: one growth chunk. A caller that knows a decode's length sizes
/// the cache to it.
pub const LM_DECODER_KV_CACHE_PRE_SIZE: usize = 1024;

/// Common meta for [`LmDecoderModel`] and [`LmDecoderStructureConfig`].
pub trait LmDecoderMeta {
    /// The token count: the embedding's rows.
    fn n_vocab(&self) -> usize;

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The first attention layer's key/value head count.
    fn n_kv_heads(&self) -> usize;

    /// The first attention layer's head width.
    fn head_dim(&self) -> usize;
}

/// How an [`LmDecoderModel`]'s stack decodes, as a type: which caches can
/// hold its state, and so which forward API it has.
///
/// A zero-size marker; building a model of a kind refuses a stack that is
/// not of it.
pub trait LmDecodeKind:
    Clone + Copy + Debug + Default + PartialEq + Eq + Send + Sync + 'static
{
    /// The kind's name, in errors.
    const NAME: &'static str;

    /// Why a stack whose layers attend as `layer_attends` says (`true` for
    /// attention, `false` for linear attention) is not of this kind; `None`
    /// when it is.
    fn refusal(layer_attends: &[bool]) -> Option<String>;
}

/// Every layer attends: a flat [`KVCache`] holds the decode's state.
///
/// Its models implement [`LmKvForwardHidden`] and [`KVCacheModel`], and
/// decode through a
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache)
/// as every kind does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LmAttentionOnlyKind;

impl LmDecodeKind for LmAttentionOnlyKind {
    const NAME: &'static str = "LmAttentionOnlyKind";

    fn refusal(layer_attends: &[bool]) -> Option<String> {
        layer_attends
            .iter()
            .position(|attends| !attends)
            .map(|i| format!("layer {i} is linear attention; every layer must attend"))
    }
}

/// Any stack, linear-attention layers among its layers: only a
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache),
/// which carries a state for each linear layer, holds the decode's state.
///
/// Its models implement [`LmUncachedForwardHidden`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LmHybridKind;

impl LmDecodeKind for LmHybridKind {
    const NAME: &'static str = "LmHybridKind";

    fn refusal(_layer_attends: &[bool]) -> Option<String> {
        None
    }
}

/// Config for [`LmDecoderModel`], its stack in any of its configs' shapes:
/// what a named model's config builds.
///
/// Lowers to [`LmDecoderStructureConfig`], and builds the model through it.
#[derive(Config, Debug)]
pub struct LmDecoderVariantConfig {
    /// The token embedding.
    pub embedding: TokenEmbeddingConfig,

    /// The output head: tied to the embedding, or a projection of its own.
    pub head: LmHeadConfig,

    /// The decoder stack.
    pub stack: DecoderStackVariantConfig,

    /// The positions a KV cache allocates up front by default.
    #[config(default = "LM_DECODER_KV_CACHE_PRE_SIZE")]
    pub kv_cache_pre_size: usize,
}

impl ToStructureConfig for LmDecoderVariantConfig {
    type Structure = LmDecoderStructureConfig;

    /// # Errors
    /// As the stack's lowering, with a frame naming it.
    fn try_to_structure(&self) -> BunsenResult<LmDecoderStructureConfig> {
        Ok(LmDecoderStructureConfig::new(
            self.embedding.clone(),
            self.head.clone(),
            self.stack
                .try_to_structure()
                .context("lowering an LmDecoderVariantConfig's stack")?,
        )
        .with_kv_cache_pre_size(self.kv_cache_pre_size))
    }
}

/// Structure config for [`LmDecoderModel`]: the embedding, the head, and
/// the stack unrolled.
///
/// Builds a model of any [`LmDecodeKind`] its stack is of.
/// Implements [`LmDecoderMeta`] (once its stack has a layer that attends).
#[derive(Config, Debug)]
pub struct LmDecoderStructureConfig {
    /// The token embedding.
    pub embedding: TokenEmbeddingConfig,

    /// The output head: tied to the embedding, or a projection of its own.
    pub head: LmHeadConfig,

    /// The decoder stack.
    pub stack: DecoderStackStructureConfig,

    /// The positions a KV cache allocates up front by default.
    #[config(default = "LM_DECODER_KV_CACHE_PRE_SIZE")]
    pub kv_cache_pre_size: usize,
}

impl LmDecoderMeta for LmDecoderStructureConfig {
    fn n_vocab(&self) -> usize {
        self.embedding.n_vocab
    }

    fn d_model(&self) -> usize {
        self.embedding.d_model
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

impl<K: LmDecodeKind> ModuleInit<LmDecoderModel<K>> for LmDecoderStructureConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for a stack that is not of kind `K`; and
    /// as the embedding's, head's and stack's builds.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<LmDecoderModel<K>> {
        if let Some(why) = K::refusal(&self.stack.layer_attends()) {
            return Err(ConstraintError::custom(
                "LmDecoderStructureConfig",
                "stack",
                format!("not of kind {}: {why}", K::NAME),
            )
            .into());
        }
        Ok(LmDecoderModel {
            embedding: self.embedding.try_init(device)?,
            head: self.head.try_init(device)?,
            stack: self.stack.try_init(device)?,
            kv_cache_pre_size: self.kv_cache_pre_size,
            kind: PhantomData,
        })
    }
}

/// A decoder-only language model: a [`TokenEmbedding`], a [`DecoderStack`]
/// and an [`LmHead`], tied to the embedding or its own; what a named
/// model's kit wraps, with the kit's config, weights and quirks.
///
/// `K`, its [`LmDecodeKind`], says how it decodes: every kind through a
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache)
/// ([`StackedLm`](crate::kits::support::decoding::StackedLm)); an
/// [`LmAttentionOnlyKind`] model through a flat [`KVCache`] too
/// ([`LmKvForwardHidden`], [`KVCacheModel`]), an [`LmHybridKind`] one only
/// uncached besides ([`LmUncachedForwardHidden`]).
///
/// Its module paths are `embedding.`, `head.` and `stack.`: a kit's load
/// rekeys a checkpoint's names onto them, and loads into the model itself,
/// not the kit's wrapper around it.
///
/// Built by [`LmDecoderVariantConfig`] or [`LmDecoderStructureConfig`].
/// Implements [`LmDecoderMeta`].
#[derive(Module, Debug)]
pub struct LmDecoderModel<K: LmDecodeKind = LmAttentionOnlyKind> {
    /// The token embedding, and the head's table when it is tied.
    pub embedding: TokenEmbedding,

    /// The output head.
    pub head: LmHead,

    /// The decoder: its layers, final norm and rotary tables.
    pub stack: DecoderStack,

    /// The positions a KV cache allocates up front by default.
    #[module(skip)]
    kv_cache_pre_size: usize,

    /// The decode kind.
    #[module(skip)]
    kind: PhantomData<K>,
}

impl<K: LmDecodeKind> LmDecoderMeta for LmDecoderModel<K> {
    fn n_vocab(&self) -> usize {
        self.embedding.n_vocab()
    }

    fn d_model(&self) -> usize {
        self.embedding.d_model()
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

impl<K: LmDecodeKind> LmDecoderModel<K> {
    /// The positions a KV cache allocates up front by default.
    pub fn kv_cache_pre_size(&self) -> usize {
        self.kv_cache_pre_size
    }

    /// The float dtype the model's weights are held, and computed, in: its
    /// embedding's (without initializing a lazy parameter).
    pub fn float_dtype(&self) -> burn::tensor::DType {
        crate::burner::tensor::param_float_dtype(&self.embedding.weight)
    }

    /// The rotary tables for positions `0..seq_len`.
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
        #[cfg(any(debug_assertions, test))]
        let [batch, seq] =
            crate::contracts::unpack_shape_contract!(["batch", "seq"], &tokens.dims());

        let y = self.embedding.forward(tokens);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("d_model", self.d_model())],
        );

        y
    }

    /// The attention weights of an uncached pass over `embeds`, at `layers`,
    /// each cut to the queries `rows` and the keys `cols`.
    ///
    /// # Arguments
    /// * `embeds` - `[batch, seq, d_model]`, at positions `0..seq`.
    /// * `r_emb` - rotary tables covering `seq` positions.
    /// * `layers` - the layers, by index, in the order returned.
    /// * `rows`, `cols` - the query and key positions kept.
    ///
    /// # Returns
    /// For each of `layers`, `[batch, n_heads, rows.len(), cols.len()]`.
    ///
    /// # Panics
    /// If a layer index is past the last layer.
    pub fn attention_weights(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        layers: &[usize],
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<Tensor<4>> {
        self.stack
            .attention_weights(embeds, r_emb, layers, rows, cols)
    }

    /// The head: logits over the vocabulary.
    ///
    /// # Arguments
    /// * `hidden` - `[batch, seq, d_model]`, the stack's output.
    ///
    /// # Returns
    /// `[batch, seq, n_vocab]`, as the head's config shapes them (sliced,
    /// soft-capped, in its `logits_dtype`).
    pub fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &hidden.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())],
        );

        let y = self.head.forward(hidden, &self.embedding.weight);

        #[cfg(any(debug_assertions, test))]
        {
            use crate::blocks::transformers::head::LmHeadMeta;
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "seq", "n_vocab"],
                &y.dims(),
                &[
                    ("batch", batch),
                    ("seq", seq),
                    ("n_vocab", self.head.n_vocab())
                ],
            );
        }

        y
    }
}

/// The forward of a model that decodes through a flat [`KVCache`]: an
/// [`LmAttentionOnlyKind`] [`LmDecoderModel`], and what wraps one.
pub trait LmKvForwardHidden {
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
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3>;
}

impl LmKvForwardHidden for LmDecoderModel<LmAttentionOnlyKind> {
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.stack.forward_hidden(embeds, r_emb, kv_cache)
    }
}

/// The uncached forward of a model whose decode runs only through a
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache):
/// an [`LmHybridKind`] [`LmDecoderModel`], and what wraps one.
pub trait LmUncachedForwardHidden {
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
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
    ) -> Tensor<3>;
}

impl LmUncachedForwardHidden for LmDecoderModel<LmHybridKind> {
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
    ) -> Tensor<3> {
        self.stack.forward_hidden(embeds, r_emb, None)
    }
}

/// The cache's geometry is the stack's; it allocates the model's
/// [`kv_cache_pre_size`](LmDecoderModel::kv_cache_pre_size) positions up
/// front, and is capped at the stack's rotary table length where the stack
/// fixes one (a nonzero `seq_len`), else uncapped.
impl KVCacheModel for LmDecoderModel<LmAttentionOnlyKind> {
    fn kv_cache_config(&self) -> KVCacheConfig {
        let cap = self.stack.rotary.seq_len;
        self.stack
            .kv_cache_config(self.kv_cache_pre_size)
            .with_max_seq_len((cap > 0).then_some(cap))
    }
}

impl<K: LmDecodeKind> CanonicalKeys for LmDecoderModel<K> {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}

/// A named model that wraps an [`LmDecoderModel`]: a kit's model, which
/// adds its own names, config and quirks around one.
///
/// The model's decode API comes through blanket impls, by its kind: every
/// wrapper is a [`StackedLm`](crate::kits::support::decoding::StackedLm);
/// an [`LmAttentionOnlyKind`] one has [`LmKvForwardHidden`] and
/// [`KVCacheModel`], an [`LmHybridKind`] one [`LmUncachedForwardHidden`].
///
/// A wrapper loads its weights into [`lm_mut`](Self::lm_mut), under a rekey
/// onto the [`LmDecoderModel`]'s own paths.
pub trait LmDecoderWrapper {
    /// The wrapped model's decode kind.
    type Kind: LmDecodeKind;

    /// The wrapped model.
    fn lm(&self) -> &LmDecoderModel<Self::Kind>;

    /// The wrapped model, to load into.
    fn lm_mut(&mut self) -> &mut LmDecoderModel<Self::Kind>;
}

impl<W: LmDecoderWrapper<Kind = LmAttentionOnlyKind>> LmKvForwardHidden for W {
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.lm().forward_hidden(embeds, r_emb, kv_cache)
    }
}

impl<W: LmDecoderWrapper<Kind = LmHybridKind>> LmUncachedForwardHidden for W {
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
    ) -> Tensor<3> {
        self.lm().forward_hidden(embeds, r_emb)
    }
}

/// The wrapped model's cache.
impl<W: LmDecoderWrapper<Kind = LmAttentionOnlyKind>> KVCacheModel for W {
    fn kv_cache_config(&self) -> KVCacheConfig {
        self.lm().kv_cache_config()
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::norm::RmsNormConfig,
        tensor::Tolerance,
    };

    use super::*;
    use crate::{
        blocks::{
            norm::{
                BunsenNormalizationConfig,
                GatedNormConfig,
            },
            transformers::{
                attention::csa::CausalSelfAttentionConfig,
                decoders::{
                    causal::CausalDecoderBlockConfig,
                    stack::{
                        PatternedStackConfig,
                        UniformStackConfig,
                    },
                },
                embedding::RotaryEmbeddingConfig,
                mixers::GatedDeltaNetConfig,
                mlp::TransformerGatedMlpConfig,
            },
        },
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        ops::transformers::attention::KVCacheMeta,
        support::testing::{
            assert_tensors_close,
            cpu_device,
        },
    };

    fn norm() -> BunsenNormalizationConfig {
        BunsenNormalizationConfig::Rms(RmsNormConfig::new(0))
    }

    fn attention_layer() -> CausalDecoderBlockConfig {
        CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(3, 1, 12)
                .with_head_dim(Some(6))
                .into(),
            TransformerGatedMlpConfig::new(12, 20).into(),
        )
        .with_norm(norm())
    }

    fn linear_layer() -> CausalDecoderBlockConfig {
        CausalDecoderBlockConfig::new(
            GatedDeltaNetConfig::new(12, 2, 4, 4, 3)
                .with_norm(GatedNormConfig::rms(0, 1e-6))
                .into(),
            TransformerGatedMlpConfig::new(12, 20).into(),
        )
        .with_norm(norm())
    }

    fn uniform(n_layers: usize) -> LmDecoderVariantConfig {
        LmDecoderVariantConfig::new(
            TokenEmbeddingConfig::new(40, 12),
            LmHeadConfig::new(12, 40),
            UniformStackConfig::new(
                n_layers,
                attention_layer(),
                norm(),
                RotaryEmbeddingConfig::new(0, 6),
            )
            .into(),
        )
    }

    fn hybrid() -> LmDecoderVariantConfig {
        LmDecoderVariantConfig::new(
            TokenEmbeddingConfig::new(40, 12),
            LmHeadConfig::new(12, 40).with_tied(false),
            PatternedStackConfig::new(
                vec![attention_layer(), linear_layer()],
                vec![1, 0],
                norm(),
                RotaryEmbeddingConfig::new(0, 6),
            )
            .into(),
        )
    }

    /// The variant and its structure build the same model, and the meta
    /// agrees across the structure and the model; the KV cache takes the
    /// stack's geometry and the model's pre-size, uncapped where the stack
    /// builds its rotary tables to order.
    #[test]
    fn test_pathways_and_meta() {
        let device = cpu_device();
        let variant = uniform(2).with_kv_cache_pre_size(16);
        let structure = variant.to_structure();
        let a: LmDecoderModel = variant.init(&device);
        let b: LmDecoderModel = structure.init(&device);
        for meta in [&a as &dyn LmDecoderMeta, &b, &structure] {
            assert_eq!(
                (
                    meta.n_vocab(),
                    meta.d_model(),
                    meta.n_layers(),
                    meta.n_kv_heads(),
                    meta.head_dim()
                ),
                (40, 12, 2, 1, 6)
            );
        }
        assert_eq!(a.num_params(), b.num_params());
        assert_eq!(a.kv_cache_pre_size(), 16);

        let config = a.kv_cache_config();
        assert_eq!(
            (
                config.num_layers(),
                config.num_heads(),
                config.head_dim(),
                config.seq_len(),
                config.max_seq_len()
            ),
            (2, 1, 6, 16, None)
        );
        assert_eq!(
            uniform(1).kv_cache_pre_size,
            LM_DECODER_KV_CACHE_PRE_SIZE,
            "the default pre-size"
        );
    }

    /// A stack that fixes its rotary table's length caps the cache there.
    #[test]
    fn test_cache_capped_at_a_fixed_rotary_table() {
        let mut variant = uniform(1);
        if let DecoderStackVariantConfig::Uniform(stack) = &mut variant.stack {
            stack.rotary.seq_len = 24;
        }
        let model: LmDecoderModel = variant.init(&cpu_device());
        assert_eq!(model.kv_cache_config().max_seq_len(), Some(24));
    }

    /// A cached decode's hidden states are the uncached forward's; the
    /// logits cover the vocabulary.
    #[test]
    fn test_cached_matches_uncached() {
        let device = cpu_device();
        let model: LmDecoderModel = uniform(2).init(&device);
        let tokens = Tensor::<2, Int>::from_data([[1, 5, 7, 2, 9]], &device);
        let embeds = model.embed(tokens);
        let r_emb = model.rotary(8, &device);
        let whole = model.forward_hidden(embeds.clone(), &r_emb, None);
        let mut cache = model.kv_cache_config().with_seq_len(8).init();
        let head =
            model.forward_hidden(embeds.clone().slice_dim(1, 0..3), &r_emb, Some(&mut cache));
        let tail = model.forward_hidden(embeds.slice_dim(1, 3..5), &r_emb, Some(&mut cache));
        assert_tensors_close(
            &Tensor::cat(vec![head, tail], 1),
            &whole,
            Tolerance::rel_abs(1e-4, 1e-5),
        );
        assert_eq!(model.logits(whole).dims(), [1, 5, 40]);
    }

    /// A hybrid stack builds a hybrid-kind model, which runs uncached; an
    /// attention-only model refuses it, naming the linear layer.
    #[test]
    fn test_kinds() {
        let device = cpu_device();
        let model: LmDecoderModel<LmHybridKind> = hybrid().init(&device);
        assert_eq!(model.stack.layer_attends(), [false, true]);
        let tokens = Tensor::<2, Int>::from_data([[1, 5, 7]], &device);
        let hidden = model.forward_hidden(model.embed(tokens), &model.rotary(3, &device));
        assert_eq!(model.logits(hidden).dims(), [1, 3, 40]);

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("LmAttentionOnlyKind")
            .message_contains("layer 0 is linear attention")
            .assert_err(&ModuleInit::<LmDecoderModel>::try_init(&hybrid(), &device));

        let _: LmDecoderModel<LmHybridKind> = uniform(1).init(&device);
    }

    /// A wrapper of an attention-only model, as a kit's.
    #[derive(Module, Debug)]
    struct Wrapped {
        lm: LmDecoderModel,
    }

    impl LmDecoderWrapper for Wrapped {
        type Kind = LmAttentionOnlyKind;

        fn lm(&self) -> &LmDecoderModel {
            &self.lm
        }

        fn lm_mut(&mut self) -> &mut LmDecoderModel {
            &mut self.lm
        }
    }

    /// A wrapper of a hybrid model.
    #[derive(Module, Debug)]
    struct WrappedHybrid {
        lm: LmDecoderModel<LmHybridKind>,
    }

    impl LmDecoderWrapper for WrappedHybrid {
        type Kind = LmHybridKind;

        fn lm(&self) -> &LmDecoderModel<LmHybridKind> {
            &self.lm
        }

        fn lm_mut(&mut self) -> &mut LmDecoderModel<LmHybridKind> {
            &mut self.lm
        }
    }

    /// A wrapper's forward and cache are its model's, by kind.
    #[test]
    fn test_wrapper_passthrough() {
        let device = cpu_device();
        let wrapped = Wrapped {
            lm: uniform(2).with_kv_cache_pre_size(16).init(&device),
        };
        let geometry = |c: KVCacheConfig| {
            (
                c.num_layers(),
                c.num_heads(),
                c.head_dim(),
                c.seq_len(),
                c.max_seq_len(),
            )
        };
        assert_eq!(
            geometry(wrapped.kv_cache_config()),
            geometry(wrapped.lm.kv_cache_config())
        );
        let tokens = Tensor::<2, Int>::from_data([[1, 5, 7]], &device);
        let embeds = wrapped.lm().embed(tokens.clone());
        let r_emb = wrapped.lm().rotary(3, &device);
        assert_tensors_close(
            &wrapped.forward_hidden(embeds.clone(), &r_emb, None),
            &wrapped.lm.forward_hidden(embeds, &r_emb, None),
            Tolerance::default(),
        );

        let hybrid = WrappedHybrid {
            lm: hybrid().init(&device),
        };
        let embeds = hybrid.lm().embed(tokens);
        assert_tensors_close(
            &hybrid.forward_hidden(embeds.clone(), &r_emb),
            &hybrid.lm.forward_hidden(embeds, &r_emb),
            Tolerance::default(),
        );
    }

    /// A stack that does not lower is refused through the variant, with a
    /// frame naming the step.
    #[test]
    fn test_lowering_frame() {
        let mut variant = hybrid();
        if let DecoderStackVariantConfig::Patterned(stack) = &mut variant.stack {
            stack.layer_kinds = vec![0, 2];
        }
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("layer_kinds[1]")
            .frame_contains("LmDecoderVariantConfig's stack")
            .assert_err(&variant.try_to_structure());
    }
}
