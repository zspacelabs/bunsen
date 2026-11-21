//! [`CachedLm`]: a language model a driver decodes by token ids.
use burn::{
    Tensor,
    prelude::Int,
    tensor::Device,
};

use super::CachedHistory;
use crate::{
    blocks::transformers::{
        decoders::stack::{
            DecoderStackCache,
            DecoderStackMeta,
        },
        models::language_decoder::{
            LmDecoderOps,
            LmPlainStackOps,
        },
    },
    kits::support::decoding::{
        CachedStep,
        StackedLm,
    },
    ops::transformers::attention::KVCacheStorage,
};

/// A language model a driver decodes by token ids, whatever its forward: its
/// decode state made, a prompt prefilled into it, and the state rewound to
/// the prefix a [`CachedHistory`] shares with the next prompt; a
/// [`CachedStep`] beside, which a search steps.
///
/// Every [`StackedLm`] is one, by a blanket impl (its state a
/// [`DecoderStackCache`]); a model whose forward is its own (nanochat's, with
/// state beside the cache) implements it by hand.
pub trait CachedLm: CachedStep {
    /// A decode state for `rows` rows of at most `positions` positions each,
    /// under `storage`.
    fn decode_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> Self::State;

    /// Runs `ids`, `[rows, seq]`, through the model into `state`, after
    /// whatever it holds.
    ///
    /// # Returns
    /// `[rows, n_logits]`, the logits that follow the last position.
    fn prefill_ids(
        &self,
        ids: Tensor<2, Int>,
        state: &mut Self::State,
    ) -> Tensor<2>;

    /// Rewinds `state`, one row whose cache holds `history`, to the prefix
    /// it shares with `next` ([`CachedHistory::rewind`]), and returns the
    /// rest of `next`: what is left to prefill.
    fn rewind<'a>(
        &self,
        history: &mut CachedHistory,
        state: &mut Self::State,
        next: &'a [i64],
        device: &Device,
    ) -> &'a [i64];

    /// Whether the state rewinds to any position it holds; else only to its
    /// [snapshot](Self::snapshot), or to 0 (a stack's linear-attention
    /// states). True by default.
    fn rewinds_anywhere(&self) -> bool {
        true
    }

    /// Keeps `state` as it is now, for a later [`rewind`](Self::rewind) to
    /// go back to; a no-op by default.
    fn snapshot(
        &self,
        _state: &mut Self::State,
    ) {
    }
}

/// A stacked model's decode state is its stack's
/// ([`DecoderStack::decode_cache`](crate::blocks::transformers::decoders::stack::DecoderStack::decode_cache)),
/// its prefill [`StackedLm::prefill`] of the ids' embeddings; it rewinds
/// anywhere unless its stack has linear-attention layers.
impl<M: StackedLm> CachedLm for M {
    fn decode_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> DecoderStackCache {
        self.stack().decode_cache(rows, positions, storage, device)
    }

    fn prefill_ids(
        &self,
        ids: Tensor<2, Int>,
        state: &mut DecoderStackCache,
    ) -> Tensor<2> {
        self.prefill(LmDecoderOps::embed(self, ids), state)
    }

    fn rewind<'a>(
        &self,
        history: &mut CachedHistory,
        state: &mut DecoderStackCache,
        next: &'a [i64],
        _device: &Device,
    ) -> &'a [i64] {
        history.rewind(state, next)
    }

    fn rewinds_anywhere(&self) -> bool {
        LmPlainStackOps::stack(self)
            .layer_attends()
            .iter()
            .all(|&attends| attends)
    }

    fn snapshot(
        &self,
        state: &mut DecoderStackCache,
    ) {
        state.snapshot();
    }
}
