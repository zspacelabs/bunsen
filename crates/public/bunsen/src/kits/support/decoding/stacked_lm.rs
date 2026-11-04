//! Cached decoding of an embedding, a decoder stack and a head.

use burn::{
    Tensor,
    prelude::Int,
    tensor::Device,
};

use crate::{
    blocks::transformers::decoders::stack::{
        DecoderCache,
        DecoderStack,
    },
    kits::support::decoding::CachedStep,
    ops::transformers::attention::KVCacheStorage,
};

/// A language model that is an embedding, a [`DecoderStack`] and a head:
/// what decodes through a [`DecoderCache`], with no state of its own.
///
/// A model gives the three pieces; the decode is the trait's:
/// [`decode_state`](Self::decode_state), [`prefill`](Self::prefill), and
/// [`CachedStep`], so [`decode_cached`](super::decode_cached) steps it.
pub trait StackedLm {
    /// The decoder stack.
    fn stack(&self) -> &DecoderStack;

    /// Embeds tokens: `[rows, seq]` to `[rows, seq, d_model]`.
    fn embed(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3>;

    /// The head: the stack's `[rows, seq, d_model]` to logits,
    /// `[rows, seq, vocab]`.
    fn head(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3>;

    /// A decode state for `rows` rows of at most `positions` positions each,
    /// under `storage` ([`DecoderStack::decode_cache`]).
    fn decode_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> DecoderCache {
        self.stack().decode_cache(rows, positions, storage, device)
    }

    /// Runs prompt embeddings through the model into `state`.
    ///
    /// # Arguments
    /// * `embeds` - `[rows, seq, d_model]`: [`embed`](Self::embed)'s tokens, or
    ///   a prompt with other frames in it.
    ///
    /// # Returns
    /// `[rows, vocab]`, the logits that follow the prompt.
    fn prefill(
        &self,
        embeds: Tensor<3>,
        state: &mut DecoderCache,
    ) -> Tensor<2> {
        let hidden = self.stack().forward_cached(embeds, state);
        let [rows, seq, _] = hidden.dims();
        let logits = self.head(hidden.slice_dim(1, seq - 1..seq));
        let vocab = logits.dims()[2];
        logits.reshape([rows, vocab])
    }
}

impl<M: StackedLm> CachedStep for M {
    type State = DecoderCache;

    fn step(
        &self,
        ids: Tensor<2, Int>,
        state: &mut DecoderCache,
    ) -> Tensor<2> {
        self.prefill(self.embed(ids), state)
    }

    fn reorder(
        &self,
        state: &mut DecoderCache,
        sources: &[usize],
    ) {
        state.reorder(sources);
    }
}
