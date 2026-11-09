//! [`LmDecoderOps`]: the kind-free API of a model and its wrappers.
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
    LmDecoderModel,
    LmDecoderWrapper,
};
use crate::blocks::transformers::{
    decoders::stack::DecoderStack,
    embedding::RotarySet,
};

/// The API every [`LmDecoderModel`] has, of any [`LmDecodeKind`], and every
/// [`LmDecoderWrapper`] through a blanket impl, as the model it wraps: the
/// stack, the rotary tables, the embedding, the head, and the attention
/// weights. The forward is by kind:
/// [`LmKvForwardHidden`](super::LmKvForwardHidden)
/// or [`LmUncachedForwardHidden`](super::LmUncachedForwardHidden).
pub trait LmDecoderOps {
    /// The decoder stack.
    fn stack(&self) -> &DecoderStack;

    /// The float dtype the weights are held, and computed, in
    /// ([`LmDecoderModel::float_dtype`]).
    fn float_dtype(&self) -> DType;

    /// The rotary tables for positions `0..seq_len`
    /// ([`LmDecoderModel::rotary`]).
    fn rotary(
        &self,
        seq_len: usize,
        device: &Device,
    ) -> RotarySet;

    /// Embeds tokens: `[batch, seq]` to `[batch, seq, d_model]`
    /// ([`LmDecoderModel::embed`]).
    fn embed(
        &self,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3>;

    /// The head: the stack's `[batch, seq, d_model]` to logits,
    /// `[batch, seq, n_vocab]` ([`LmDecoderModel::logits`]).
    fn logits(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3>;

    /// The attention weights of an uncached pass over `embeds`, at `layers`,
    /// each cut to the queries `rows` and the keys `cols`
    /// ([`LmDecoderModel::attention_weights`]).
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
    ) -> Vec<Tensor<4>>;
}

impl<K: LmDecodeKind> LmDecoderOps for LmDecoderModel<K> {
    fn stack(&self) -> &DecoderStack {
        &self.stack
    }

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

    fn attention_weights(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        layers: &[usize],
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<Tensor<4>> {
        LmDecoderModel::attention_weights(self, embeds, r_emb, layers, rows, cols)
    }
}

/// The wrapped model's API.
impl<W: LmDecoderWrapper> LmDecoderOps for W {
    fn stack(&self) -> &DecoderStack {
        &self.lm().stack
    }

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

    fn attention_weights(
        &self,
        embeds: Tensor<3>,
        r_emb: &RotarySet,
        layers: &[usize],
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<Tensor<4>> {
        self.lm()
            .attention_weights(embeds, r_emb, layers, rows, cols)
    }
}
