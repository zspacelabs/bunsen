//! Cached decoding of an embedding, a decoder stack and a head.

use burn::{
    Tensor,
    prelude::Int,
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        decoders::stack::{
            DecoderStack,
            DecoderStackCache,
        },
        models::{
            LmDecodeKind,
            LmDecoderModel,
            LmDecoderWrapper,
        },
    },
    kits::support::decoding::CachedStep,
    ops::transformers::attention::KVCacheStorage,
};

/// A language model that is an embedding, a [`DecoderStack`] and a head:
/// what decodes through a [`DecoderStackCache`], with no state of its own.
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
    ) -> DecoderStackCache {
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
        state: &mut DecoderStackCache,
    ) -> Tensor<2> {
        let hidden = self.stack().forward_cached(embeds, state);
        let [rows, seq, _] = hidden.dims();
        let logits = self.head(hidden.slice_dim(1, seq - 1..seq));
        let vocab = logits.dims()[2];
        logits.reshape([rows, vocab])
    }
}

impl<M: StackedLm> CachedStep for M {
    type State = DecoderStackCache;

    fn step(
        &self,
        ids: Tensor<2, Int>,
        state: &mut DecoderStackCache,
    ) -> Tensor<2> {
        self.prefill(self.embed(ids), state)
    }

    fn reorder(
        &self,
        state: &mut DecoderStackCache,
        sources: &[usize],
    ) {
        state.reorder(sources);
    }
}

/// Every [`LmDecoderModel`], of any kind, decodes through its stack's
/// [`DecoderStackCache`].
impl<K: LmDecodeKind> StackedLm for LmDecoderModel<K> {
    fn stack(&self) -> &DecoderStack {
        &self.stack
    }

    fn embed(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        LmDecoderModel::embed(self, ids)
    }

    fn head(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.logits(hidden)
    }
}

/// A wrapper of an [`LmDecoderModel`] decodes as the model it wraps.
impl<W: LmDecoderWrapper> StackedLm for W {
    fn stack(&self) -> &DecoderStack {
        &self.lm().stack
    }

    fn embed(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.lm().embed(ids)
    }

    fn head(
        &self,
        hidden: Tensor<3>,
    ) -> Tensor<3> {
        self.lm().logits(hidden)
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
                    stack::PatternedStackConfig,
                },
                embedding::{
                    RotaryEmbeddingConfig,
                    TokenEmbeddingConfig,
                },
                head::LmHeadConfig,
                mixers::GatedDeltaNetConfig,
                mlp::TransformerGatedMlpConfig,
                models::{
                    LmDecoderVariantConfig,
                    LmHybridKind,
                    LmUncachedForwardHidden,
                },
            },
        },
        burner::module::ModuleInit,
        support::testing::{
            assert_tensors_close,
            cpu_device,
        },
    };

    /// A hybrid [`LmDecoderModel`]'s prefill and step through its decode
    /// state give the logits of the uncached forward over the whole
    /// sequence.
    #[test]
    fn test_lm_decoder_model_steps_as_uncached() {
        let device = cpu_device();
        let norm = || BunsenNormalizationConfig::Rms(RmsNormConfig::new(0));
        let mlp = || TransformerGatedMlpConfig::new(12, 20).into();
        let attention = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(3, 1, 12)
                .with_head_dim(Some(6))
                .into(),
            mlp(),
        )
        .with_norm(norm());
        let linear = CausalDecoderBlockConfig::new(
            GatedDeltaNetConfig::new(12, 2, 4, 4, 3)
                .with_norm(GatedNormConfig::rms(0, 1e-6))
                .into(),
            mlp(),
        )
        .with_norm(norm());
        let model: LmDecoderModel<LmHybridKind> = LmDecoderVariantConfig::new(
            TokenEmbeddingConfig::new(40, 12),
            LmHeadConfig::new(12, 40),
            PatternedStackConfig::new(
                vec![attention, linear],
                vec![1, 0, 1],
                norm(),
                RotaryEmbeddingConfig::new(0, 6),
            )
            .into(),
        )
        .init(&device);

        let tokens = Tensor::<2, Int>::from_data([[1, 5, 7, 2]], &device);
        let mut state = model.decode_state(1, 4, KVCacheStorage::Auto, &device);
        model.prefill(model.embed(tokens.clone().slice_dim(1, 0..3)), &mut state);
        let stepped = model.step(tokens.clone().slice_dim(1, 3..4), &mut state);

        let hidden = model.forward_hidden(model.embed(tokens), &model.rotary(4, &device));
        let whole = model.logits(hidden.slice_dim(1, 3..4)).reshape([1, 40]);
        assert_tensors_close(&stepped, &whole, Tolerance::rel_abs(1e-4, 1e-5));
    }
}
