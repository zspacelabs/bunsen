//! The decoder without its token embedding, [`LmStackDecoder`].

use alloc::{
    format,
    string::ToString,
    vec::Vec,
};
use core::{
    marker::PhantomData,
    ops::Range,
};

use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    tensor::Device,
};

use super::{
    LM_DECODER_KV_CACHE_PRE_SIZE,
    LmAttentionOnlyKind,
    LmDecodeKind,
    LmHybridKind,
    LmKvForwardHidden,
    LmUncachedForwardHidden,
};
use crate::{
    blocks::transformers::{
        decoders::stack::{
            DecoderStack,
            DecoderStackMeta,
            DecoderStackStructureConfig,
        },
        embedding::RotarySet,
        head::{
            LmHead,
            LmHeadConfig,
            LmHeadMeta,
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
    ops::transformers::attention::{
        KVCache,
        KVCacheConfig,
        KVCacheModel,
    },
};

/// Common meta for [`LmStackDecoder`] and [`LmStackDecoderConfig`].
pub trait LmStackDecoderMeta {
    /// The logits' width: the head's output.
    fn n_logits(&self) -> usize;

    /// The model width.
    fn d_model(&self) -> usize;

    /// The layer count.
    fn n_layers(&self) -> usize;

    /// The first attention layer's key/value head count.
    fn n_kv_heads(&self) -> usize;

    /// The first attention layer's head width.
    fn head_dim(&self) -> usize;
}

/// Config for [`LmStackDecoder`]: the head and the stack unrolled.
///
/// Builds a decoder of any [`LmDecodeKind`] its stack is of.
#[derive(Config, Debug)]
pub struct LmStackDecoderConfig {
    /// The output head: a projection of its own, or tied to a token table the
    /// caller hands [`logits`](LmStackDecoder::logits).
    pub head: LmHeadConfig,

    /// The decoder stack.
    pub stack: DecoderStackStructureConfig,

    /// The positions a KV cache allocates up front by default.
    #[config(default = "LM_DECODER_KV_CACHE_PRE_SIZE")]
    pub kv_cache_pre_size: usize,
}

impl LmStackDecoderMeta for LmStackDecoderConfig {
    fn n_logits(&self) -> usize {
        self.head.n_vocab()
    }

    fn d_model(&self) -> usize {
        self.stack.d_model()
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

impl<K: LmDecodeKind> ModuleInit<LmStackDecoder<K>> for LmStackDecoderConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for a stack that is not of kind `K`, or a
    /// head whose width is not the stack's; and as the head's and stack's
    /// builds.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<LmStackDecoder<K>> {
        if let Some(why) = K::refusal(&self.stack.layer_attends()) {
            return Err(ConstraintError::custom(
                "LmStackDecoderConfig",
                "stack",
                format!("not of kind {}: {why}", K::NAME),
            )
            .into());
        }
        if !self.stack.layers.is_empty() && self.head.d_model != self.stack.d_model() {
            return Err(ConstraintError::new(
                "LmStackDecoderConfig",
                "head",
                Rule::Relation {
                    lhs: ("head.d_model".into(), self.head.d_model.to_string()),
                    op: "==",
                    rhs: ("stack.d_model".into(), self.stack.d_model().to_string()),
                },
            )
            .into());
        }
        Ok(LmStackDecoder {
            head: self.head.try_init(device)?,
            stack: self.stack.try_init(device)?,
            kv_cache_pre_size: self.kv_cache_pre_size,
            kind: PhantomData,
        })
    }
}

/// A decoder-only language model's decoder, without its token embedding: a
/// [`DecoderStack`] and an [`LmHead`], reading embeddings its caller makes.
///
/// [`LmDecoderModel`](super::LmDecoderModel) is one with a
/// [`TokenEmbedding`](crate::blocks::transformers::embedding::TokenEmbedding);
/// a model whose rows are not one table's lookups (summed codebook tables,
/// replaced rows) holds one directly, with its own embedder.
///
/// `K`, its [`LmDecodeKind`], says how it decodes, as the model's does. Its
/// module paths are `head.` and `stack.`. Built by [`LmStackDecoderConfig`].
#[derive(Module, Debug)]
pub struct LmStackDecoder<K: LmDecodeKind = LmAttentionOnlyKind> {
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

impl<K: LmDecodeKind> LmStackDecoderMeta for LmStackDecoder<K> {
    fn n_logits(&self) -> usize {
        self.head.n_vocab()
    }

    fn d_model(&self) -> usize {
        self.stack.d_model()
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

impl<K: LmDecodeKind> LmStackDecoder<K> {
    /// The positions a KV cache allocates up front by default.
    pub fn kv_cache_pre_size(&self) -> usize {
        self.kv_cache_pre_size
    }

    /// The rotary tables for positions `0..seq_len`.
    pub fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotarySet {
        self.stack.rotary(seq_len, device)
    }

    /// The attention weights of an uncached pass over `embeds`, at `layers`,
    /// each cut to the queries `rows` and the keys `cols`
    /// ([`DecoderStack::attention_weights`]).
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
    /// * `table` - the token table a tied head projects onto; a head of its own
    ///   ignores it.
    ///
    /// # Returns
    /// `[batch, seq, n_logits]`, as the head's config shapes them (sliced,
    /// soft-capped, in its `logits_dtype`).
    ///
    /// # Panics
    /// If the head is tied and handed no table.
    pub fn logits(
        &self,
        hidden: Tensor<3>,
        table: Option<&Param<Tensor<2>>>,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &hidden.dims(),
            &["batch", "seq"],
            &[("d_model", self.d_model())],
        );
        let y = self.head.forward(hidden, table);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "n_vocab"],
            &y.dims(),
            &[
                ("batch", batch),
                ("seq", seq),
                ("n_vocab", self.head.n_vocab())
            ],
        );
        y
    }
}

impl LmKvForwardHidden for LmStackDecoder<LmAttentionOnlyKind> {
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
    ) -> Tensor<3> {
        self.stack.forward_hidden(embeds, r_emb, kv_cache)
    }
}

impl LmUncachedForwardHidden for LmStackDecoder<LmHybridKind> {
    fn forward_hidden(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
    ) -> Tensor<3> {
        self.stack.forward_hidden(embeds, r_emb, None)
    }
}

/// The cache's geometry is the stack's; it allocates the decoder's
/// [`kv_cache_pre_size`](LmStackDecoder::kv_cache_pre_size) positions up
/// front, and is capped at the stack's rotary table length where the stack
/// fixes one (a nonzero `seq_len`), else uncapped.
impl KVCacheModel for LmStackDecoder<LmAttentionOnlyKind> {
    fn kv_cache_config(&self) -> KVCacheConfig {
        let cap = self.stack.rotary.seq_len;
        self.stack
            .kv_cache_config(self.kv_cache_pre_size)
            .with_max_seq_len((cap > 0).then_some(cap))
    }
}

impl<K: LmDecodeKind> CanonicalKeys for LmStackDecoder<K> {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}
