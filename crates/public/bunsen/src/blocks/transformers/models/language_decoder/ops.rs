//! [`LmDecoderOps`] and [`LmPlainStackOps`]: the call APIs, by layer.
use alloc::vec::Vec;
use core::ops::Range;

use burn::{
    Tensor,
    prelude::Int,
    tensor::{
        DType,
        Device,
    },
};

use super::{
    LmDecodeKind,
    LmDecoderHolder,
    LmDecoderModel,
    LmDecoderWrapper,
};
use crate::blocks::transformers::{
    decoders::stack::DecoderStack,
    embedding::RotarySet,
};

/// The call API of a decoder-only language model that says nothing of its
/// forward: the embedding, the head, the rotary tables and the float dtype.
///
/// Every [`LmDecoderModel`], of any [`LmDecodeKind`], implements it, and
/// every [`LmDecoderHolder`] through a blanket impl, as the model it holds;
/// a model whose forward is its own (nanochat's, with inputs beside the
/// stack's) implements it by hand. The forward is [`LmPlainStackOps`]' and
/// the kind's, where it is the bare stack's.
pub trait LmDecoderOps {
    /// The float dtype the weights are held, and computed, in.
    fn float_dtype(&self) -> DType;

    /// The rotary tables for positions `0..seq_len`.
    fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotarySet;

    /// Embeds tokens: `[batch, seq]` to `[batch, seq, d_model]`; the token
    /// embedding alone, whatever the forward adds to it.
    fn embed(
        &self,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3>;

    /// The head: the stack's normalized `[batch, seq, d_model]` to logits,
    /// `[batch, seq, n_vocab]`.
    fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3>;
}

/// The API of a model whose forward is its [`DecoderStack`]'s alone, from
/// [`embed`](LmDecoderOps::embed)'s embeddings: an [`LmDecoderModel`] and
/// every [`LmDecoderWrapper`] (by a blanket impl), and nothing else, as
/// [`StackedLm`](crate::kits::support::decoding::StackedLm)'s decode, the
/// kinds' forwards and [`attention_weights`](Self::attention_weights) take
/// the bare stack to be the forward.
pub trait LmPlainStackOps: LmDecoderOps {
    /// The decoder stack.
    fn stack(&self) -> &DecoderStack;

    /// The attention weights of an uncached pass over `embeds`, at `layers`,
    /// each cut to the queries `rows` and the keys `cols`
    /// ([`DecoderStack::attention_weights`]).
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
    fn attention_weights(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        layers: &[usize],
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<Tensor<4>> {
        self.stack()
            .attention_weights(embeds, r_emb, layers, rows, cols)
    }
}

impl<K: LmDecodeKind> LmDecoderOps for LmDecoderModel<K> {
    fn float_dtype(&self) -> DType {
        LmDecoderModel::float_dtype(self)
    }

    fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotarySet {
        LmDecoderModel::rotary(self, seq_len, device)
    }

    fn embed(
        &self,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3> {
        LmDecoderModel::embed(self, tokens)
    }

    fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        LmDecoderModel::logits(self, hidden)
    }
}

impl<K: LmDecodeKind> LmPlainStackOps for LmDecoderModel<K> {
    fn stack(&self) -> &DecoderStack {
        &self.stack
    }
}

/// The held model's API.
impl<W: LmDecoderHolder> LmDecoderOps for W {
    fn float_dtype(&self) -> DType {
        self.lm().float_dtype()
    }

    fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotarySet {
        self.lm().rotary(seq_len, device)
    }

    fn embed(
        &self,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.lm().embed(tokens)
    }

    fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.lm().logits(hidden)
    }
}

/// The wrapped model's stack.
impl<W: LmDecoderWrapper> LmPlainStackOps for W {
    fn stack(&self) -> &DecoderStack {
        &self.lm().stack
    }
}
