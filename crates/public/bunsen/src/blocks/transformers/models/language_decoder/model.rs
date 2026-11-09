//! The model, [`LmDecoderModel`], and its forward APIs.
use alloc::vec::Vec;
use core::{
    marker::PhantomData,
    ops::Range,
};

use burn::{
    Tensor,
    module::Module,
    prelude::Int,
    tensor::Device,
};

use super::{
    LmAttentionOnlyKind,
    LmDecodeKind,
    LmHybridKind,
};
use crate::{
    blocks::transformers::{
        decoders::stack::{
            DecoderStack,
            DecoderStackMeta,
        },
        embedding::{
            RotarySet,
            TokenEmbedding,
            TokenEmbeddingMeta,
        },
        head::LmHead,
    },
    burner::module::{
        CanonicalKeys,
        Rekey,
    },
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
};

/// Common meta for [`LmDecoderModel`],
/// [`LmDecoderStructureConfig`](super::LmDecoderStructureConfig), every
/// [`LmDecoderWrapper`](super::LmDecoderWrapper) (as the model it wraps), and a
/// kit's config that builds one.
pub trait LmDecoderMeta {
    /// The token count: the embedding's rows.
    fn n_vocab(&self) -> usize;

    /// The logits' width: the head's output, which a head that slices its
    /// table (a padded vocabulary) narrows below [`n_vocab`](Self::n_vocab);
    /// by default, `n_vocab`.
    fn n_logits(&self) -> usize {
        self.n_vocab()
    }

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The first attention layer's key/value head count.
    fn n_kv_heads(&self) -> usize;

    /// The first attention layer's head width.
    fn head_dim(&self) -> usize;
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
/// Built by [`LmDecoderVariantConfig`](super::LmDecoderVariantConfig) or
/// [`LmDecoderStructureConfig`](super::LmDecoderStructureConfig). Implements
/// [`LmDecoderMeta`].
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
    pub(super) kv_cache_pre_size: usize,

    /// The decode kind.
    #[module(skip)]
    pub(super) kind: PhantomData<K>,
}

impl<K: LmDecodeKind> LmDecoderMeta for LmDecoderModel<K> {
    fn n_vocab(&self) -> usize {
        self.embedding.n_vocab()
    }

    fn n_logits(&self) -> usize {
        use crate::blocks::transformers::head::LmHeadMeta;
        self.head.n_vocab()
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
