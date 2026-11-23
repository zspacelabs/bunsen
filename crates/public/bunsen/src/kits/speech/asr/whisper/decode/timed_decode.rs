//! [`Whisper`] as a [`TimedDecode`].

use burn::{
    Tensor,
    tensor::Device,
};

use crate::{
    kits::{
        speech::asr::whisper::{
            Whisper,
            WhisperMeta,
        },
        support::timing::{
            TimedDecode,
            synthetic_ids,
        },
    },
    ops::transformers::attention::{
        KVCache,
        KVCacheStorage,
    },
};

/// What a timed Whisper decode carries: the text decoder's cache, made by
/// the encoder's pass ([`TimedDecode::timed_encode`]), and how to store it.
#[derive(Debug)]
pub struct WhisperTimedState {
    storage: KVCacheStorage,
    cache: Option<KVCache>,
}

impl WhisperTimedState {
    fn cache(&mut self) -> &mut KVCache {
        self.cache
            .as_mut()
            .expect("a timed Whisper decode is encoded before it prefills")
    }
}

/// A transcription's phases: the encoder over a 30 s window of (zero)
/// log-mels per row, the cross-attention keys and values projected into the
/// text decoder's cache, then the decoder's prefill and steps through it.
impl TimedDecode for Whisper {
    type State = WhisperTimedState;

    /// # Panics
    /// If `positions` passes the text context
    /// ([`max_text_ctx`](WhisperMeta::max_text_ctx)).
    fn timed_state(
        &self,
        _rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        _device: &Device,
    ) -> WhisperTimedState {
        assert!(
            positions <= self.max_text_ctx(),
            "a Whisper decode holds at most {} positions, not {positions}",
            self.max_text_ctx()
        );
        WhisperTimedState {
            storage,
            cache: None,
        }
    }

    fn timed_encode(
        &self,
        state: &mut WhisperTimedState,
        rows: usize,
        device: &Device,
    ) -> Option<Tensor<2>> {
        let mels = Tensor::<3>::zeros([rows, self.n_mels(), self.max_audio_ctx()], device);
        let xa = self.forward_encoder(mels);
        let [_, _, d] = xa.dims();
        state.cache = Some(self.decoder.new_cache_with(xa.clone(), 1, state.storage));
        Some(xa.slice_dim(1, 0..1).reshape([rows, d]))
    }

    fn timed_prefill(
        &self,
        state: &mut WhisperTimedState,
        rows: usize,
        len: usize,
        device: &Device,
    ) -> Tensor<2> {
        let logits = self
            .decoder
            .forward_cached(synthetic_ids(rows, 0, len, device), state.cache());
        let [_, _, vocab] = logits.dims();
        logits.slice_dim(1, len - 1..len).reshape([rows, vocab])
    }

    fn timed_reorder(
        &self,
        state: &mut WhisperTimedState,
        sources: &[usize],
    ) {
        state.cache().reorder(sources);
    }

    fn timed_step(
        &self,
        state: &mut WhisperTimedState,
        rows: usize,
        device: &Device,
    ) -> Tensor<2> {
        let logits = self
            .decoder
            .forward_cached(synthetic_ids(rows, 0, 1, device), state.cache());
        let [_, _, vocab] = logits.dims();
        logits.reshape([rows, vocab])
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::asr::whisper::WhisperApiConfig,
        support::testing::{
            assert_tensors_close,
            cpu_device,
        },
    };

    /// A timed Whisper decode is the model's: the prefill's and a step's
    /// logits are the uncached decoder's over the same tokens and the same
    /// encoder output.
    #[test]
    fn test_timed_whisper_is_its_decode() {
        let device = cpu_device();
        let model: Whisper = WhisperApiConfig::new(8, 1000, 16, 40, 2, 12, 2)
            .with_d_head(8)
            .init(&device);
        let (rows, len) = (2, 3);
        let mut state = model.timed_state(rows, len + 1, KVCacheStorage::Auto, &device);
        let encoded = model.timed_encode(&mut state, rows, &device).unwrap();
        assert_eq!(encoded.dims(), [rows, 16]);
        let prefilled = model.timed_prefill(&mut state, rows, len, &device);
        let stepped = model.timed_step(&mut state, rows, &device);

        let xa = model.forward_encoder(Tensor::zeros([rows, 8, 40], &device));
        let tokens = Tensor::cat(
            vec![
                synthetic_ids(rows, 0, len, &device),
                synthetic_ids(rows, 0, 1, &device),
            ],
            1,
        );
        let whole = model.forward_decoder(tokens, xa);
        let at = |t: usize| whole.clone().slice_dim(1, t..t + 1).reshape([rows, 1000]);
        let tolerance = Tolerance::rel_abs(1e-4, 1e-5);
        assert_tensors_close(&prefilled, &at(len - 1), tolerance);
        assert_tensors_close(&stepped, &at(len), tolerance);
    }

    /// The text context bounds a timed decode.
    #[test]
    #[should_panic(expected = "holds at most 12 positions")]
    fn test_the_text_context_bounds_it() {
        let device = cpu_device();
        let model: Whisper = WhisperApiConfig::new(8, 1000, 16, 40, 2, 12, 2)
            .with_d_head(8)
            .init(&device);
        let _ = model.timed_state(1, 13, KVCacheStorage::Auto, &device);
    }
}
