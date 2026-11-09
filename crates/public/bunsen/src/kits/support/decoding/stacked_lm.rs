//! Cached decoding of an embedding, a decoder stack and a head.

use burn::{
    Tensor,
    prelude::Int,
};

use crate::{
    blocks::transformers::{
        decoders::stack::DecoderStackCache,
        models::language_decoder::LmPlainStackOps,
    },
    kits::support::decoding::CachedStep,
};

/// A language model that is an embedding, a
/// [`DecoderStack`](crate::blocks::transformers::decoders::stack::DecoderStack)
/// and a head whose forward is the stack's alone ([`LmPlainStackOps`]): what
/// decodes through a [`DecoderStackCache`], with no state of its own.
///
/// The decode is the trait's: [`prefill`](Self::prefill), and
/// [`CachedStep`] and
/// [`CachedLm`](crate::kits::language::driver::CachedLm) (whose
/// `decode_state` is the stack's
/// [`decode_cache`](crate::blocks::transformers::decoders::stack::DecoderStack::decode_cache)), so
/// [`decode_cached`](super::decode_cached) steps it. Every
/// [`LmPlainStackOps`] (an
/// [`LmDecoderModel`](crate::blocks::transformers::models::language_decoder::LmDecoderModel)
/// of any kind, and every wrapper of one) is one, by a blanket impl; a model
/// whose forward is its own is not, and steps by its own [`CachedStep`].
pub trait StackedLm: LmPlainStackOps {
    /// Runs prompt embeddings through the model into `state`.
    ///
    /// # Arguments
    /// * `embeds` - `[rows, seq, d_model]`:
    ///   [`embed`](crate::blocks::transformers::models::language_decoder::LmDecoderOps::embed)'
    ///   s tokens, or a prompt with other frames in it.
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
        let logits = self.logits(hidden.slice_dim(1, seq - 1..seq));
        let vocab = logits.dims()[2];
        logits.reshape([rows, vocab])
    }
}

impl<M: LmPlainStackOps> StackedLm for M {}

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
                models::language_decoder::{
                    LmDecoderModel,
                    LmDecoderVariantConfig,
                    LmHybridKind,
                    LmUncachedForwardHidden,
                },
            },
        },
        burner::module::ModuleInit,
        kits::language::driver::CachedLm,
        ops::transformers::attention::KVCacheStorage,
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
