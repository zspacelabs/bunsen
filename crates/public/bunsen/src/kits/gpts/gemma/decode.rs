//! Cached decoding of a Gemma.
use burn::{
    Tensor,
    prelude::Int,
};

use super::GemmaModel;
use crate::{
    blocks::transformers::decoders::stack::DecoderStack,
    kits::support::decoding::StackedLm,
};

/// A Gemma decodes through its stack's
/// [`DecoderCache`](crate::blocks::transformers::decoders::stack::DecoderCache):
/// [`StackedLm::decode_state`], [`StackedLm::prefill`], then
/// [`decode_cached`](crate::kits::support::decoding::decode_cached).
impl StackedLm for GemmaModel {
    fn stack(&self) -> &DecoderStack {
        &self.stack
    }

    fn embed(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        GemmaModel::embed(self, ids)
    }

    fn head(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.logits(hidden)
    }
}
