//! Greedy decoding of a Voxtral Realtime utterance.
use burn::{
    Tensor,
    module::Module,
    prelude::{
        Int,
        TensorData,
    },
    tensor::Device,
};

use super::{
    VoxtralTokenLayout,
    blocks::{
        VoxtralRealtime,
        VoxtralTextMeta,
    },
};
use crate::{
    blocks::transformers::embedding::RotaryEmbedding,
    kits::decoding::{
        CachedStep,
        GreedySearch,
        decode_cached,
    },
    ops::transformers::attention::{
        KVCache,
        KVCacheModel,
        KVCacheStorage,
    },
};

/// What a Voxtral Realtime decode carries between steps: the text model's
/// cache, the rotary table and the delay condition it decodes under, and the
/// audio embeddings, one per position, that each step adds to its token's.
#[derive(Module, Debug)]
pub struct VoxtralDecodeState {
    /// The text model's cache; its position is the next step's.
    pub cache: KVCache,

    /// The rotary table, covering every position of the audio.
    pub rotary: RotaryEmbedding,

    /// `[1, 1, d_text]`: the delay condition.
    pub cond: Tensor<3>,

    /// `[rows, positions, d_text]`: the audio embeddings.
    pub audio: Tensor<3>,
}

/// A [`VoxtralRealtime`] as [`decode_cached`] steps it: each id is embedded,
/// the next position's audio added, and the text model run one position
/// through the cache.
#[derive(Clone, Copy, Debug)]
pub struct VoxtralDecodeStep<'a> {
    /// The model.
    pub model: &'a VoxtralRealtime,
}

impl CachedStep for VoxtralDecodeStep<'_> {
    type State = VoxtralDecodeState;

    fn step(
        &self,
        ids: Tensor<2, Int>,
        state: &mut VoxtralDecodeState,
    ) -> Tensor<2> {
        let lm = &self.model.language_model;
        let pos = state.cache.pos();
        let audio = state.audio.clone().slice_dim(1, pos..pos + 1);
        let hidden = lm.forward_hidden(
            lm.embed(ids) + audio,
            &state.rotary,
            &state.cond,
            Some(&mut state.cache),
        );
        last_logits(lm.logits(hidden))
    }

    fn reorder(
        &self,
        state: &mut VoxtralDecodeState,
        sources: &[usize],
    ) {
        state.cache.reorder(sources);
        let device = state.audio.device();
        let index = Tensor::<1, Int>::from_data(
            TensorData::new(
                sources.iter().map(|&s| s as i64).collect::<Vec<_>>(),
                [sources.len()],
            ),
            &device,
        );
        state.audio = state.audio.clone().select(0, index);
    }
}

/// `[rows, vocab]`: the last position's logits.
fn last_logits(logits: Tensor<3>) -> Tensor<2> {
    let [rows, seq, vocab] = logits.dims();
    logits.slice_dim(1, seq - 1..seq).reshape([rows, vocab])
}

/// Greedy decoding of one utterance, as `transformers`' `generate` decodes
/// it: the prompt's ids prefilled over the first audio positions, then one
/// greedy id per position, until the audio's positions are used or `</s>`.
#[derive(Clone, Debug)]
pub struct VoxtralGreedyDecoder {
    /// The cache storage the decode uses.
    pub storage: KVCacheStorage,
}

impl Default for VoxtralGreedyDecoder {
    fn default() -> Self {
        Self {
            storage: KVCacheStorage::Auto,
        }
    }
}

impl VoxtralGreedyDecoder {
    /// A decode state over `audio`, for one row.
    ///
    /// # Arguments
    /// * `audio` - `[1, positions, d_text]` audio embeddings.
    pub fn state(
        &self,
        model: &VoxtralRealtime,
        layout: &VoxtralTokenLayout,
        audio: Tensor<3>,
        device: &Device,
    ) -> VoxtralDecodeState {
        let lm = &model.language_model;
        let positions = audio.dims()[1];
        VoxtralDecodeState {
            cache: lm
                .kv_cache_config()
                .with_seq_len(positions)
                .with_max_seq_len(Some(positions))
                .with_storage(self.storage)
                .init(),
            rotary: lm.rotary(positions, device),
            cond: lm.delay_condition(layout.delay_positions(), device),
            audio,
        }
    }

    /// Decodes an utterance.
    ///
    /// # Arguments
    /// * `model` - the model.
    /// * `layout` - the stream's layout: its prompt and delay.
    /// * `audio` - `[1, positions, d_text]` audio embeddings of the padded
    ///   utterance ([`VoxtralRealtime::audio_embeds`]), at least the prompt's
    ///   length.
    ///
    /// # Returns
    /// The whole sequence, the prompt first: one id per position but the
    /// last (whose successor has no audio), or fewer, ending in `</s>`.
    ///
    /// # Panics
    /// If the audio is shorter than the prompt.
    pub fn decode(
        &self,
        model: &VoxtralRealtime,
        layout: &VoxtralTokenLayout,
        audio: Tensor<3>,
    ) -> Vec<i64> {
        let [positions] = crate::contracts::unpack_shape_contract!(
            [1, "positions", "d_text"],
            &audio.dims(),
            &["positions"],
            &[("d_text", model.language_model.d_model())]
        );
        let prompt = layout.prompt();
        assert!(
            positions >= prompt.len(),
            "{positions} audio positions under a {}-position prompt",
            prompt.len()
        );
        let device = audio.device();
        let mut state = self.state(model, layout, audio.clone(), &device);
        let lm = &model.language_model;
        let ids = Tensor::<2, Int>::from_data(
            TensorData::new(prompt.clone(), [1, prompt.len()]),
            &device,
        );
        let embeds = lm.embed(ids) + audio.slice_dim(1, 0..prompt.len());
        let first = last_logits(lm.logits(lm.forward_hidden(
            embeds,
            &state.rotary,
            &state.cond,
            Some(&mut state.cache),
        )));
        let eos = layout.config().eos as i64;
        let mut search = GreedySearch::new([eos], eos);
        let budget = positions - prompt.len();
        let out = decode_cached(
            &VoxtralDecodeStep { model },
            &mut state,
            vec![prompt.clone()],
            first,
            budget,
            &mut search,
            &[],
        );
        let mut ids = prompt;
        ids.extend(out[0][0].tokens_with_stop());
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::speech::voxtral::{
            VoxtralTokenLayoutConfig,
            blocks::{
                VoxtralAudioEncoderConfig,
                VoxtralRealtimeConfig,
                VoxtralTextConfig,
            },
        },
        support::testing::cpu_device,
    };

    /// The cached greedy decode chooses, at every position, the argmax an
    /// uncached pass over the same ids and audio chooses, and fills every
    /// position but the last.
    #[test]
    fn test_cached_decode_matches_uncached() {
        let device = cpu_device();
        let model: VoxtralRealtime = VoxtralRealtimeConfig::new(
            VoxtralAudioEncoderConfig::new(6, 8, 2, 4, 16, 1).with_sliding_window(Some(3)),
            VoxtralTextConfig::new(40, 12, 2, 3, 1, 4, 20).with_sliding_window(Some(5)),
        )
        .with_downsample(2)
        .init(&device);
        // Ids the toy vocabulary holds.
        let layout = VoxtralTokenLayoutConfig::new()
            .with_bos(1)
            .with_eos(39)
            .with_streaming_pad(3)
            .with_streaming_word(4)
            .with_left_pad_positions(2)
            .with_delay_ms(160)
            .init();
        let audio = crate::support::testing::seeded_tensor::<3>(
            1,
            [1, 12, 12],
            burn::tensor::Distribution::Default,
            &device,
        );
        let ids = VoxtralGreedyDecoder::default().decode(&model, &layout, audio.clone());
        let prompt = layout.prompt();
        assert_eq!(&ids[..prompt.len()], &prompt[..]);
        if ids.last() != Some(&39) {
            assert_eq!(ids.len(), 12);
        }

        for n in prompt.len()..ids.len() {
            let tokens =
                Tensor::<2, Int>::from_data(TensorData::new(ids[..n].to_vec(), [1, n]), &device);
            let logits = model.logits(tokens, audio.clone().slice_dim(1, 0..n), 2);
            let expected = last_logits(logits).argmax(1).into_scalar::<i64>();
            assert_eq!(ids[n], expected, "position {n}");
        }
    }
}
