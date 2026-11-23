//! [`VoxtralRealtime`] as a [`TimedDecode`].

use burn::{
    Tensor,
    tensor::Device,
};

use super::{
    VoxtralDecodeState,
    VoxtralDecodeStep,
    VoxtralTokenLayoutConfig,
    blocks::{
        VoxtralRealtime,
        VoxtralRealtimeMeta,
    },
};
use crate::{
    blocks::transformers::models::language_decoder::LmDecoderOps,
    kits::support::{
        decoding::CachedStep,
        timing::{
            TimedDecode,
            synthetic_ids,
        },
    },
    ops::transformers::attention::KVCacheStorage,
};

/// What a timed Voxtral Realtime decode carries: its geometry until the
/// audio is encoded ([`TimedDecode::timed_encode`]), then the decode's own
/// state.
#[derive(Debug)]
pub struct VoxtralTimedState {
    positions: usize,
    storage: KVCacheStorage,
    decode: Option<VoxtralDecodeState>,
}

impl VoxtralTimedState {
    fn decode(&mut self) -> &mut VoxtralDecodeState {
        self.decode
            .as_mut()
            .expect("a timed Voxtral decode is encoded before it prefills")
    }
}

/// A transcription's phases, as [`VoxtralGreedyDecoder`](super::VoxtralGreedyDecoder)
/// runs them: the audio tower and projector over (zero) log-mels for every
/// position, then the text model's prefill and steps, each position's audio
/// added to its token, under the default layout's delay.
impl TimedDecode for VoxtralRealtime {
    type State = VoxtralTimedState;

    fn timed_state(
        &self,
        _rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        _device: &Device,
    ) -> VoxtralTimedState {
        VoxtralTimedState {
            positions,
            storage,
            decode: None,
        }
    }

    fn timed_encode(
        &self,
        state: &mut VoxtralTimedState,
        rows: usize,
        device: &Device,
    ) -> Option<Tensor<2>> {
        let positions = state.positions;
        let features = Tensor::<3>::zeros(
            [
                rows,
                self.n_mels(),
                positions * self.mel_frames_per_position(),
            ],
            device,
        );
        let audio = self.audio_embeds(features);
        let lm = &self.language_model;
        let delay = VoxtralTokenLayoutConfig::new().init().delay_positions();
        state.decode = Some(VoxtralDecodeState {
            cache: lm
                .lm
                .decoder
                .stack
                .decode_cache(rows, positions, state.storage, device),
            cond: lm.delay_condition(delay, device),
            audio: audio.clone(),
        });
        Some(audio.slice_dim(1, 0..1).reshape([rows, self.d_text()]))
    }

    fn timed_prefill(
        &self,
        state: &mut VoxtralTimedState,
        rows: usize,
        len: usize,
        device: &Device,
    ) -> Tensor<2> {
        let lm = &self.language_model;
        let decode = state.decode();
        let pos = decode.cache.cache.pos();
        let embeds = lm.embed(synthetic_ids(rows, 0, len, device))
            + decode.audio.clone().slice_dim(1, pos..pos + len);
        let hidden = lm.forward_hidden(
            embeds,
            &decode.cache.rotary,
            &decode.cond,
            Some(&mut decode.cache.cache),
        );
        let logits = lm.logits(hidden);
        let [_, _, vocab] = logits.dims();
        logits.slice_dim(1, len - 1..len).reshape([rows, vocab])
    }

    fn timed_step(
        &self,
        state: &mut VoxtralTimedState,
        rows: usize,
        device: &Device,
    ) -> Tensor<2> {
        VoxtralDecodeStep { model: self }.step(synthetic_ids(rows, 0, 1, device), state.decode())
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Int,
        tensor::Tolerance,
    };

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::asr::voxtral::blocks::{
            VoxtralAudioEncoderConfig,
            VoxtralRealtimeConfig,
            VoxtralTextConfig,
        },
        support::testing::{
            assert_tensors_close,
            cpu_device,
        },
    };

    /// A timed Voxtral decode is the model's: the prefill's and a step's
    /// logits are the uncached model's over the same ids and audio, under the
    /// default delay.
    #[test]
    fn test_timed_voxtral_is_its_decode() {
        let device = cpu_device();
        let model: VoxtralRealtime = VoxtralRealtimeConfig::new(
            VoxtralAudioEncoderConfig::new(6, 8, 2, 4, 16, 1).with_sliding_window(Some(3)),
            VoxtralTextConfig::new(1000, 12, 2, 3, 1, 4, 20).with_sliding_window(Some(5)),
        )
        .with_downsample(2)
        .init(&device);
        let (rows, len, positions) = (2, 3, 5);
        let mut state = model.timed_state(rows, positions, KVCacheStorage::Auto, &device);
        let encoded = model.timed_encode(&mut state, rows, &device).unwrap();
        assert_eq!(encoded.dims(), [rows, 12]);
        let prefilled = model.timed_prefill(&mut state, rows, len, &device);
        let stepped = model.timed_step(&mut state, rows, &device);

        let features = Tensor::<3>::zeros(
            [rows, 6, positions * model.mel_frames_per_position()],
            &device,
        );
        let audio = model.audio_embeds(features);
        let ids: Tensor<2, Int> = Tensor::cat(
            vec![
                synthetic_ids(rows, 0, len, &device),
                synthetic_ids(rows, 0, 1, &device),
            ],
            1,
        );
        let delay = VoxtralTokenLayoutConfig::new().init().delay_positions();
        let whole = model.logits(ids, audio.slice_dim(1, 0..len + 1), delay);
        let at = |t: usize| whole.clone().slice_dim(1, t..t + 1).reshape([rows, 1000]);
        let tolerance = Tolerance::rel_abs(1e-4, 1e-5);
        assert_tensors_close(&prefilled, &at(len - 1), tolerance);
        assert_tensors_close(&stepped, &at(len), tolerance);
    }
}
