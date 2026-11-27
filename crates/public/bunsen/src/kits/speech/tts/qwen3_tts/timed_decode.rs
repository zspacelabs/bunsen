//! [`Qwen3TtsModel`] as a [`TimedDecode`].

use burn::{
    Tensor,
    prelude::{
        Int,
        s,
    },
    tensor::{
        Device,
        TensorData,
    },
};

use super::blocks::{
    Qwen3TtsModel,
    Qwen3TtsModelMeta,
};
use crate::{
    blocks::transformers::{
        embedding::RotarySet,
        models::language_decoder::LmDecoderMeta,
    },
    burner::module::ModuleInit,
    kits::support::timing::{
        TimedDecode,
        synthetic_ids,
    },
    ops::{
        reduce::argmax_rows,
        transformers::attention::{
            KVCache,
            KVCacheModel,
            KVCacheStorage,
        },
    },
};

/// What a timed Qwen3-TTS take carries: the talker's cache and rotary
/// tables, and its last hidden state and the logits after it, which choose
/// the next frame's codebook 0.
#[derive(burn::module::Module, Debug)]
pub struct Qwen3TtsTimedState {
    cache: KVCache,
    rotary: RotarySet,
    logits: Option<Tensor<2>>,
    cache_hidden: Option<Tensor<3>>,
}

/// A take's phases, as [`run_take`](Qwen3TtsModel::run_take) runs them, one
/// row: the talker prefilled with (text-embedded) prompt rows, then per step
/// a frame: codebook 0 from the talker's logits, the code predictor's depth
/// decode of the rest from the talker's hidden state, the frame's
/// embeddings summed (and the text pad) as the talker's next row. Every code
/// is the greedy choice, read back to the host as the samplers read theirs,
/// so a frame syncs once per codebook, as a real take does.
impl TimedDecode for Qwen3TtsModel {
    type State = Qwen3TtsTimedState;

    fn step_unit(&self) -> &'static str {
        "frame"
    }

    /// # Panics
    /// For more than one row: a take is one.
    fn timed_state(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> Qwen3TtsTimedState {
        assert_eq!(rows, 1, "a Qwen3-TTS take is one row");
        Qwen3TtsTimedState {
            cache: self
                .talker
                .kv_cache_config()
                .with_storage(storage)
                .init(device),
            rotary: self.talker.rotary(positions, device),
            logits: None,
            cache_hidden: None,
        }
    }

    fn timed_prefill(
        &self,
        state: &mut Qwen3TtsTimedState,
        rows: usize,
        len: usize,
        device: &Device,
    ) -> Tensor<2> {
        let embeds = self.talker.text_embed(synthetic_ids(rows, 0, len, device));
        let hidden = self
            .talker
            .forward_hidden(embeds, &state.rotary, Some(&mut state.cache));
        self.talker_logits(state, hidden.slice(s![.., len - 1..len, ..]))
    }

    fn timed_step(
        &self,
        state: &mut Qwen3TtsTimedState,
        _rows: usize,
        device: &Device,
    ) -> Tensor<2> {
        let talker = &self.talker;
        let predictor = &self.code_predictor;
        let logits = state
            .logits
            .take()
            .expect("a timed Qwen3-TTS take is prefilled before it steps");
        let code0 =
            Tensor::<2, Int>::from_data(TensorData::new(vec![greedy(logits)], [1, 1]), device);
        let embed0 = talker.codec_embed(code0);
        let hidden = state
            .cache_hidden
            .take()
            .expect("the talker's last hidden state");
        let context = Tensor::cat(vec![hidden, embed0.clone()], 1);
        let rest = predictor.decode(context, |_, logits| {
            Tensor::from_data(TensorData::new(vec![greedy(logits)], [1]), device)
        });
        let mut row = embed0;
        for k in 1..self.code_groups() {
            row = row + predictor.embed(k, rest.clone().slice(s![.., k - 1..k]));
        }
        row = row + talker.text_embed(synthetic_ids(1, 0, 1, device));
        let hidden = talker.forward_hidden(row, &state.rotary, Some(&mut state.cache));
        self.talker_logits(state, hidden)
    }
}

impl Qwen3TtsModel {
    /// The talker's `[1, V]` logits after `hidden`, `[1, 1, D]`, kept in
    /// `state` (with `hidden`) for the next frame's choice.
    fn talker_logits(
        &self,
        state: &mut Qwen3TtsTimedState,
        hidden: Tensor<3>,
    ) -> Tensor<2> {
        let vocab = self.talker.lm.n_logits();
        let logits = self.talker.logits(hidden.clone()).reshape([1, vocab]);
        state.logits = Some(logits.clone());
        state.cache_hidden = Some(hidden);
        logits
    }
}

/// `[1, V]` logits' greedy choice, read back to the host: the argmax the
/// samplers take ([`argmax_rows`]).
fn greedy(logits: Tensor<2>) -> i64 {
    argmax_rows(logits)
        .into_data()
        .convert::<i64>()
        .try_to_vec::<i64>()
        .expect("an argmax reads back")[0]
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::{
            language::generation::qwen3::Qwen3Config,
            speech::tts::qwen3_tts::{
                blocks::{
                    Qwen3TtsCodePredictorConfig,
                    Qwen3TtsModelConfig,
                    Qwen3TtsTalkerConfig,
                    release_text_projection,
                },
                driver::Qwen3TtsChooser,
                prompt::{
                    Qwen3TtsPrompt,
                    Qwen3TtsPromptConfig,
                },
            },
        },
        support::testing::{
            REORDER,
            assert_tensors_close,
            cpu_device,
        },
    };

    /// Every code greedy, and every codebook-0 choice's logits kept.
    struct Greedy(Vec<Tensor<2>>);

    impl Qwen3TtsChooser for Greedy {
        fn codebook0(
            &mut self,
            _frame: usize,
            logits: Tensor<2>,
            _blocked: &[i64],
        ) -> i64 {
            self.0.push(logits.clone());
            greedy(logits)
        }

        fn codebook(
            &mut self,
            _frame: usize,
            _k: usize,
            logits: Tensor<2>,
        ) -> i64 {
            greedy(logits)
        }
    }

    /// A timed take is the model's: its prefill's and each frame's logits
    /// are the ones `run_take` chooses codebook 0 from, over the same prompt
    /// rows, the same pad, and greedy codes.
    #[test]
    fn test_timed_qwen3_tts_is_its_take() {
        let device = cpu_device();
        let model: Qwen3TtsModel = Qwen3TtsModelConfig::new(
            Qwen3TtsTalkerConfig::new(
                Qwen3Config::new(11, 8, 1, 2, 1, 4, 12),
                1000,
                6,
                release_text_projection(6, 8),
                4,
            ),
            Qwen3TtsCodePredictorConfig::new(Qwen3Config::new(7, 4, 1, 2, 1, 2, 6), 8, 3),
        )
        .init(&device);
        let len = 3;
        let mut state = model.timed_state(1, len + 3, KVCacheStorage::Auto, &device);
        assert_eq!(TimedDecode::step_unit(&model), "frame");
        let timed = [
            model.timed_prefill(&mut state, 1, len, &device),
            model.timed_step(&mut state, 1, &device),
            model.timed_step(&mut state, 1, &device),
        ];

        // The pad is the synthetic step id; no id is the codec's EOS.
        let pad = synthetic_ids(1, 0, 1, &device)
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap()[0] as u32;
        let ids = Qwen3TtsPromptConfig::new(
            pad,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            u32::MAX,
            0,
            0,
            0,
            0,
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let prompt = Qwen3TtsPrompt {
            embeds: model.talker.text_embed(synthetic_ids(1, 0, len, &device)),
            trailing: Tensor::zeros([1, 0, 8], &device),
        };
        let mut chooser = Greedy(Vec::new());
        model.run_take(&prompt, &ids, 3, 0, &mut chooser);
        assert_eq!(chooser.0.len(), 3);
        for (timed, taken) in timed.iter().zip(&chooser.0) {
            assert_tensors_close(timed, taken, REORDER);
        }
    }

    /// A take is one row.
    #[test]
    #[should_panic(expected = "a Qwen3-TTS take is one row")]
    fn test_a_take_is_one_row() {
        let device = cpu_device();
        let model: Qwen3TtsModel = Qwen3TtsModelConfig::new(
            Qwen3TtsTalkerConfig::new(
                Qwen3Config::new(11, 8, 1, 2, 1, 4, 12),
                1000,
                6,
                release_text_projection(6, 8),
                4,
            ),
            Qwen3TtsCodePredictorConfig::new(Qwen3Config::new(7, 4, 1, 2, 1, 2, 6), 8, 3),
        )
        .init(&device);
        let _ = model.timed_state(2, 8, KVCacheStorage::Auto, &device);
    }
}
