//! Cached decoding of a Qwen3.5.
use burn::{
    Tensor,
    prelude::Int,
};

use super::Qwen3_5Model;
use crate::{
    blocks::transformers::decoders::stack::DecoderStack,
    kits::support::decoding::StackedLm,
};

/// A Qwen3.5 decodes through its stack's
/// [`DecoderStackCache`](crate::blocks::transformers::decoders::stack::DecoderStackCache),
/// a KV cache for its full-attention layers and a state for each linear one:
/// [`StackedLm::decode_state`], [`StackedLm::prefill`], then
/// [`decode_cached`](crate::kits::support::decoding::decode_cached).
impl StackedLm for Qwen3_5Model {
    fn stack(&self) -> &DecoderStack {
        &self.stack
    }

    fn embed(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        Qwen3_5Model::embed(self, ids)
    }

    fn head(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.logits(hidden)
    }
}
