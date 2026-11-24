//! [`Qwen3TtsModel`]: the talker and its code predictor, and the take loop.

use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    config::Config,
    prelude::*,
};

use super::{
    Qwen3TtsCodePredictorConfig,
    Qwen3TtsTalker,
    Qwen3TtsTalkerConfig,
    Qwen3TtsTalkerMeta,
};
use crate::{
    blocks::transformers::{
        decoders::depth::{
            DepthDecoder,
            DepthDecoderMeta,
        },
        embedding::RotarySet,
        models::language_decoder::LmDecoderMeta,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::{
        speech::tts::qwen3_tts::{
            prompt::{
                Qwen3TtsPrompt,
                Qwen3TtsPromptConfig,
            },
            sampler::Qwen3TtsSamplerConfig,
        },
        support::decoding::PoolRelease,
    },
    ops::transformers::attention::{
        KVCache,
        KVCacheModel,
    },
};

/// Config for [`Qwen3TtsModel`].
#[derive(Config, Debug)]
pub struct Qwen3TtsModelConfig {
    /// The talker.
    pub talker: Qwen3TtsTalkerConfig,

    /// The code predictor.
    pub code_predictor: Qwen3TtsCodePredictorConfig,
}

/// A [`Qwen3TtsModel`]'s geometry, which its config and the model answer
/// alike.
pub trait Qwen3TtsModelMeta {
    /// The talker's width, `W`.
    fn d_model(&self) -> usize;

    /// The codebooks a frame holds, `K + 1`: the talker's first and the code
    /// predictor's rest.
    fn code_groups(&self) -> usize;

    /// The codec table's control rows, at its end, which a take never samples
    /// but EOS: the table past the code predictor's codes (the release's
    /// 1024, which upstream's `generate` fixes).
    fn control_rows(&self) -> usize;
}

impl Qwen3TtsModelMeta for Qwen3TtsModelConfig {
    fn d_model(&self) -> usize {
        self.talker.d_model()
    }

    fn code_groups(&self) -> usize {
        self.code_predictor.codebooks() + 1
    }

    fn control_rows(&self) -> usize {
        self.talker
            .codec_vocab()
            .saturating_sub(self.code_predictor.vocab())
    }
}

impl ModuleInit<Qwen3TtsModel> for Qwen3TtsModelConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsModel> {
        Ok(Qwen3TtsModel {
            talker: self.talker.try_init(device)?,
            code_predictor: self.code_predictor.try_init(device)?,
        })
    }
}

/// Config for a take ([`Qwen3TtsModel::generate_codes`]); its defaults are
/// the release's (`generation_config.json`,
/// [`take_config`](crate::kits::speech::tts::qwen3_tts::interop::qwen_tts::Qwen3TtsGenerationMirror::take_config)).
#[derive(Config, Debug, PartialEq)]
pub struct Qwen3TtsTakeConfig {
    /// The talker's sampler: codebook 0.
    #[config(default = "Qwen3TtsSamplerConfig::new().with_repetition_penalty(1.05)")]
    pub talker: Qwen3TtsSamplerConfig,

    /// The code predictor's sampler: codebooks 1 on.
    #[config(default = "Qwen3TtsSamplerConfig::new()")]
    pub code_predictor: Qwen3TtsSamplerConfig,

    /// The most frames a take holds. Upstream's `max_new_tokens` of
    /// codebook-0 tokens make one frame fewer: the last token's frame is
    /// never completed.
    #[config(default = 8191)]
    pub max_frames: usize,

    /// The frames before EOS may end the take (upstream's
    /// `min_new_tokens`).
    #[config(default = 2)]
    pub min_frames: usize,
}

/// Why a take ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3TtsStop {
    /// The talker chose EOS.
    Eos,

    /// It reached [`max_frames`](Qwen3TtsTakeConfig::max_frames).
    MaxFrames,
}

/// A take's codes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen3TtsCodes {
    /// Each frame's codebooks, `[F][K + 1]`; the EOS frame not among them.
    pub frames: Vec<Vec<u32>>,

    /// Why the take ended.
    pub stop: Qwen3TtsStop,
}

impl Qwen3TtsCodes {
    /// The frames as the codec reads them: `[1, K + 1, F]`.
    ///
    /// # Panics
    /// For a take of no frames.
    pub fn codec_input(
        &self,
        device: &Device,
    ) -> Tensor<3, Int> {
        let f = self.frames.len();
        assert!(f > 0, "a take of at least one frame");
        let g = self.frames[0].len();
        let flat: Vec<i64> = self
            .frames
            .iter()
            .flat_map(|frame| frame.iter().map(|&c| i64::from(c)))
            .collect();
        Tensor::<2, Int>::from_data(TensorData::new(flat, [f, g]), device)
            .transpose()
            .unsqueeze_dim(0)
    }
}

/// Qwen3-TTS's model (`Qwen3TTSForConditionalGeneration` without its
/// speaker encoder): the talker and its code predictor.
///
/// A take ([`generate_codes`](Self::generate_codes)) prefills the talker
/// with the prompt, then per frame: the talker's logits choose codebook 0 (EOS
/// ends the take); the code predictor, from the talker's hidden state and
/// codebook 0's embedding, chooses the rest; the frame's embeddings, summed
/// (codebook 0's through the codec table, the rest's through the
/// predictor's tables), plus the frame's trailing text, are the talker's
/// next row.
///
/// Built by [`Qwen3TtsModelConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsModel {
    /// The talker.
    pub talker: Qwen3TtsTalker,

    /// The code predictor.
    pub code_predictor: DepthDecoder,
}

impl Qwen3TtsModelMeta for Qwen3TtsModel {
    fn d_model(&self) -> usize {
        self.talker.d_model()
    }

    fn code_groups(&self) -> usize {
        self.code_predictor.codebooks() + 1
    }

    fn control_rows(&self) -> usize {
        self.talker
            .codec_vocab()
            .saturating_sub(self.code_predictor.vocab())
    }
}

impl Qwen3TtsModel {
    /// A take: the codes of `prompt`, chosen as `config` says (its samplers,
    /// [`Qwen3TtsSampler`](crate::kits::speech::tts::qwen3_tts::sampler::Qwen3TtsSampler)), ids from `ids`.
    ///
    /// The draws run on the backend's random numbers: `device.seed(..)`
    /// before the call repeats a sampled take.
    pub fn generate_codes(
        &self,
        prompt: &Qwen3TtsPrompt,
        ids: &Qwen3TtsPromptConfig,
        config: &Qwen3TtsTakeConfig,
    ) -> Qwen3TtsCodes {
        let mut chooser = self.sampled_chooser(ids, config);
        self.run_take(
            prompt,
            ids,
            config.max_frames,
            config.min_frames,
            &mut chooser,
        )
    }

    /// A take of at most `max_frames` frames, each code chosen by
    /// `chooser`, EOS blocked before `min_frames`: the loop
    /// [`generate_codes`](Self::generate_codes) runs, for a caller that
    /// chooses otherwise (a script that dictates the codes, and reads the
    /// logits).
    pub fn run_take(
        &self,
        prompt: &Qwen3TtsPrompt,
        ids: &Qwen3TtsPromptConfig,
        max_frames: usize,
        min_frames: usize,
        chooser: &mut impl Qwen3TtsChooser,
    ) -> Qwen3TtsCodes {
        let mut run = self.start_take(prompt, ids, max_frames, min_frames);
        loop {
            if let Some(stop) = self.step_take(&mut run, chooser) {
                return Qwen3TtsCodes {
                    frames: run.frames,
                    stop,
                };
            }
        }
    }

    /// A take started: the talker prefilled with `prompt`, nothing chosen
    /// yet. [`step_take`](Self::step_take) then runs it a frame at a time;
    /// [`run_take`](Self::run_take) is that loop.
    pub fn start_take(
        &self,
        prompt: &Qwen3TtsPrompt,
        ids: &Qwen3TtsPromptConfig,
        max_frames: usize,
        min_frames: usize,
    ) -> Qwen3TtsTakeRun {
        let device = prompt.embeds.device();
        let talker = &self.talker;
        let l = prompt.embeds.dims()[1];
        let r_emb = talker.rotary(l + max_frames, &device);
        let mut cache = talker.kv_cache_config().init(&device);
        let pad = talker.text_embed(Tensor::from_data(
            TensorData::new(vec![i64::from(ids.tts_pad)], [1, 1]),
            &device,
        ));
        let hidden = talker.forward_hidden(prompt.embeds.clone(), &r_emb, Some(&mut cache));
        let t = hidden.dims()[1];
        Qwen3TtsTakeRun {
            r_emb,
            cache,
            pad,
            trailing: prompt.trailing.clone(),
            hidden: hidden.slice(s![.., t - 1..t, ..]),
            frames: Vec::new(),
            max_frames,
            min_frames,
            eos: i64::from(ids.codec_eos),
            pool: PoolRelease::default(),
            stop: None,
        }
    }

    /// The take's next frame, each code chosen by `chooser`, EOS blocked
    /// before its `min_frames`: the frame pushed onto
    /// [`frames`](Qwen3TtsTakeRun::frames), or why the take stopped (no
    /// frame pushed).
    ///
    /// # Panics
    /// For a take that stopped already.
    pub fn step_take(
        &self,
        run: &mut Qwen3TtsTakeRun,
        chooser: &mut impl Qwen3TtsChooser,
    ) -> Option<Qwen3TtsStop> {
        assert!(run.stop.is_none(), "a step of a take that stopped");
        let device = run.hidden.device();
        let talker = &self.talker;
        let predictor = &self.code_predictor;
        let vocab = talker.lm.n_logits();
        let f = run.frames.len();
        if f == run.max_frames {
            run.stop = Some(Qwen3TtsStop::MaxFrames);
            return run.stop;
        }
        let logits = talker.logits(run.hidden.clone()).reshape([1, vocab]);
        let eos = run.eos;
        let blocked: &[i64] = if f < run.min_frames { &[eos] } else { &[] };
        let first = chooser.codebook0(f, logits, blocked);
        if first == eos {
            run.stop = Some(Qwen3TtsStop::Eos);
            return run.stop;
        }

        let code0 = Tensor::<2, Int>::from_data(TensorData::new(vec![first], [1, 1]), &device);
        let embed0 = talker.codec_embed(code0);
        let context = Tensor::cat(vec![run.hidden.clone(), embed0.clone()], 1);
        let mut frame = vec![first as u32];
        let rest = predictor.decode(context, |k, logits| {
            let code = chooser.codebook(f, k, logits);
            frame.push(code as u32);
            Tensor::from_data(TensorData::new(vec![code], [1]), &device)
        });
        let mut row = embed0;
        for k in 1..self.code_groups() {
            row = row + predictor.embed(k, rest.clone().slice(s![.., k - 1..k]));
        }
        let trailing = run.trailing.dims()[1];
        row = row
            + if f < trailing {
                run.trailing.clone().slice(s![.., f..=f, ..])
            } else {
                run.pad.clone()
            };
        run.frames.push(frame);
        run.pool.after_step(f, &device);
        run.hidden = talker.forward_hidden(row, &run.r_emb, Some(&mut run.cache));
        None
    }

    /// The samplers' chooser of a take under `config`
    /// ([`generate_codes`](Self::generate_codes)'s), for a caller that steps
    /// the take itself ([`step_take`](Self::step_take)).
    pub fn sampled_chooser(
        &self,
        ids: &Qwen3TtsPromptConfig,
        config: &Qwen3TtsTakeConfig,
    ) -> Qwen3TtsSampledChooser {
        let vocab = self.talker.lm.n_logits();
        let eos = i64::from(ids.codec_eos);
        let control = vocab.saturating_sub(self.control_rows()) as i64..vocab as i64;
        Qwen3TtsSampledChooser {
            talker: config.talker.init(control.filter(|&id| id != eos)),
            code_predictor: config.code_predictor.init([]),
            history: Vec::new(),
            frame: Vec::new(),
        }
    }
}

/// A take in progress ([`Qwen3TtsModel::start_take`]): the talker's cache
/// and its last hidden row, and the frames chosen so far.
///
/// A `Module` over bare tensors: moved with `to_device`, never recorded or
/// visited.
#[derive(Module, Debug)]
pub struct Qwen3TtsTakeRun {
    /// The talker's rotary tables, over the prompt and the most frames.
    pub r_emb: RotarySet,

    /// The talker's KV cache.
    pub cache: KVCache,

    /// `[1, 1, W]`: the text pad's row, past the trailing rows.
    pub pad: Tensor<3>,

    /// `[1, t, W]`: what the first frames add, in order.
    pub trailing: Tensor<3>,

    /// `[1, 1, W]`: the talker's last hidden row.
    pub hidden: Tensor<3>,

    /// The frames so far, `[F][K + 1]`.
    pub frames: Vec<Vec<u32>>,

    /// The most frames.
    pub max_frames: usize,

    /// The frames before EOS may end the take.
    pub min_frames: usize,

    /// The codec's EOS id.
    pub eos: i64,

    /// When the device pool's free memory is released.
    #[module(skip)]
    pub pool: PoolRelease,

    /// Why the take stopped; `None` while it runs.
    #[module(skip)]
    pub stop: Option<Qwen3TtsStop>,
}

/// What chooses a take's codes from its logits
/// ([`Qwen3TtsModel::run_take`]): the samplers, or a script.
pub trait Qwen3TtsChooser {
    /// Frame `frame`'s codebook 0, from the talker's logits, `[1, V]`; never
    /// one of `blocked` (EOS, before the take's first frames). EOS ends the
    /// take.
    fn codebook0(
        &mut self,
        frame: usize,
        logits: Tensor<2>,
        blocked: &[i64],
    ) -> i64;

    /// Frame `frame`'s codebook `k`, `1..=K`, from the code predictor's
    /// logits, `[1, V_k]`.
    fn codebook(
        &mut self,
        frame: usize,
        k: usize,
        logits: Tensor<2>,
    ) -> i64;
}

/// [`Qwen3TtsModel::generate_codes`]'s chooser
/// ([`sampled_chooser`](Qwen3TtsModel::sampled_chooser)): the samplers, the
/// talker's repetition penalty over the take's codebook-0 history, the
/// predictor's over the frame's codes.
#[derive(Debug)]
pub struct Qwen3TtsSampledChooser {
    talker: crate::kits::speech::tts::qwen3_tts::sampler::Qwen3TtsSampler,

    code_predictor: crate::kits::speech::tts::qwen3_tts::sampler::Qwen3TtsSampler,

    history: Vec<i64>,
    frame: Vec<i64>,
}

impl Qwen3TtsChooser for Qwen3TtsSampledChooser {
    fn codebook0(
        &mut self,
        _frame: usize,
        logits: Tensor<2>,
        blocked: &[i64],
    ) -> i64 {
        let code = first_id(self.talker.choose(
            logits,
            core::slice::from_ref(&self.history),
            blocked,
        ));
        self.history.push(code);
        self.frame.clear();
        code
    }

    fn codebook(
        &mut self,
        _frame: usize,
        _k: usize,
        logits: Tensor<2>,
    ) -> i64 {
        let code = first_id(self.code_predictor.choose(
            logits,
            core::slice::from_ref(&self.frame),
            &[],
        ));
        self.frame.push(code);
        code
    }
}

/// The first id of `ids`.
fn first_id(ids: Tensor<1, Int>) -> i64 {
    ids.into_data().iter::<i64>().next().expect("an id")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kits::{
            language::generation::qwen3::Qwen3Config,
            speech::tts::qwen3_tts::blocks::release_text_projection,
        },
        support::testing::cpu_device,
    };

    /// The config and the model built from it answer the meta alike: the
    /// control rows are the codec table's 11 rows past the predictor's 7
    /// codes.
    #[test]
    fn test_model_meta() {
        let config = Qwen3TtsModelConfig::new(
            Qwen3TtsTalkerConfig::new(
                Qwen3Config::new(11, 8, 1, 2, 1, 4, 12),
                13,
                6,
                release_text_projection(6, 8),
                4,
            ),
            Qwen3TtsCodePredictorConfig::new(Qwen3Config::new(7, 4, 1, 2, 1, 2, 6), 8, 3),
        );
        let model: Qwen3TtsModel = config.init(&cpu_device());
        for meta in [&config as &dyn Qwen3TtsModelMeta, &model] {
            assert_eq!(
                [meta.d_model(), meta.code_groups(), meta.control_rows()],
                [8, 4, 4]
            );
        }
    }
}
