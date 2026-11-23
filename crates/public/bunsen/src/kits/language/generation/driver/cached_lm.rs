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
    kits::support::{
        decoding::{
            CachedStep,
            StackedLm,
        },
        timing::{
            TimedDecode,
            synthetic_ids,
        },
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

/// A token decode's phases on [`synthetic_ids`]: its state is its
/// [`decode_state`](CachedLm::decode_state), its prefill
/// [`prefill_ids`](CachedLm::prefill_ids), each step a
/// [`step`](CachedStep::step) of one fixed id per row.
impl<M: CachedLm> TimedDecode for M {
    type State = M::State;

    fn timed_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> M::State {
        self.decode_state(rows, positions, storage, device)
    }

    fn timed_prefill(
        &self,
        state: &mut M::State,
        rows: usize,
        len: usize,
        device: &Device,
    ) -> Tensor<2> {
        self.prefill_ids(synthetic_ids(rows, 0, len, device), state)
    }

    fn timed_step(
        &self,
        state: &mut M::State,
        rows: usize,
        device: &Device,
    ) -> Tensor<2> {
        self.step(synthetic_ids(rows, 0, 1, device), state)
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
            mlp::BunsenGatedMlpConfig,
            norm::BunsenNormalizationConfig,
            transformers::{
                attention::csa::CausalSelfAttentionConfig,
                decoders::{
                    causal::CausalDecoderBlockConfig,
                    stack::UniformStackConfig,
                },
                embedding::{
                    RotaryEmbeddingConfig,
                    TokenEmbeddingConfig,
                },
                head::LmHeadConfig,
                models::language_decoder::{
                    LmDecoderModel,
                    LmDecoderVariantConfig,
                },
            },
        },
        burner::module::ModuleInit,
        kits::support::timing::SYNTHETIC_ID_RANGE,
        support::testing::{
            assert_tensors_close,
            cpu_device,
        },
    };

    /// A language model as small as the synthetic ids allow.
    fn model(device: &Device) -> LmDecoderModel {
        let vocab = SYNTHETIC_ID_RANGE as usize;
        let norm = || BunsenNormalizationConfig::Rms(RmsNormConfig::new(0));
        LmDecoderVariantConfig::new(
            TokenEmbeddingConfig::new(vocab, 12),
            LmHeadConfig::new(12, vocab),
            UniformStackConfig::new(
                2,
                CausalDecoderBlockConfig::new(
                    CausalSelfAttentionConfig::new(3, 1, 12)
                        .with_head_dim(Some(6))
                        .into(),
                    BunsenGatedMlpConfig::new(12, 20).into(),
                )
                .with_norm(norm()),
                norm(),
                RotaryEmbeddingConfig::new(0, 6),
            )
            .into(),
        )
        .init(device)
    }

    /// A language model's timed phases are its decode on the synthetic ids:
    /// the prefill's and each step's logits are the ones `prefill_ids` and
    /// `step` give on them, a row each.
    #[test]
    fn test_a_cached_lm_times_its_own_decode() {
        let device = cpu_device();
        let model = model(&device);
        assert_eq!(TimedDecode::step_unit(&model), "token");
        let (rows, len) = (2, 3);
        let mut timed = model.timed_state(rows, 8, KVCacheStorage::Auto, &device);
        let mut real = model.decode_state(rows, 8, KVCacheStorage::Auto, &device);

        let prefilled = model.timed_prefill(&mut timed, rows, len, &device);
        assert_eq!(prefilled.dims(), [rows, SYNTHETIC_ID_RANGE as usize]);
        let expected = model.prefill_ids(synthetic_ids(rows, 0, len, &device), &mut real);
        assert_tensors_close(&prefilled, &expected, Tolerance::default());

        for _ in 0..2 {
            let stepped = model.timed_step(&mut timed, rows, &device);
            let expected = model.step(synthetic_ids(rows, 0, 1, &device), &mut real);
            assert_tensors_close(&stepped, &expected, Tolerance::default());
        }
    }
}
