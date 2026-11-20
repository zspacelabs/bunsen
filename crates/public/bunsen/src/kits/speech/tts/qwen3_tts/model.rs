//! The model: the talker and its code predictor, and the take loop.

use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    config::Config,
    prelude::*,
};

use super::{
    code_predictor::Qwen3TtsCodePredictorConfig,
    prompt::{
        Qwen3TtsPrompt,
        Qwen3TtsPromptConfig,
    },
    sampler::{
        Qwen3TtsGenerationJson,
        Qwen3TtsSamplerConfig,
    },
    talker::{
        Qwen3TtsTalker,
        Qwen3TtsTalkerConfig,
        Qwen3TtsTalkerMeta,
    },
};
use crate::{
    blocks::transformers::{
        decoders::depth::{
            DepthDecoder,
            DepthDecoderMeta,
        },
        models::language_decoder::LmDecoderMeta,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::support::model_config::TransformersConfigJson,
    ops::transformers::attention::KVCacheModel,
};

/// The codec table's control rows, at its end: `generate` never samples
/// them but EOS.
pub const QWEN3_TTS_CONTROL_ROWS: usize = 1024;

/// Config for [`Qwen3TtsModel`].
#[derive(Config, Debug)]
pub struct Qwen3TtsModelConfig {
    /// The talker.
    pub talker: Qwen3TtsTalkerConfig,

    /// The code predictor.
    pub code_predictor: Qwen3TtsCodePredictorConfig,
}

impl Qwen3TtsModelConfig {
    /// The config of a model `config.json`: its `talker_config`, and that
    /// section's `code_predictor_config`.
    ///
    /// # Errors
    /// As [`Qwen3TtsTalkerConfig::from_transformers_section`] and
    /// [`Qwen3TtsCodePredictorConfig::from_transformers_section`].
    pub fn from_transformers(json: &TransformersConfigJson) -> BunsenResult<Self> {
        let section = json.section("talker_config")?;
        let talker = Qwen3TtsTalkerConfig::from_transformers_section(&section)?;
        let code_predictor = Qwen3TtsCodePredictorConfig::from_transformers_section(
            &section.section("code_predictor_config")?,
            talker.lm.d_model,
        )?;
        Ok(Self::new(talker, code_predictor))
    }
}

/// A [`Qwen3TtsModel`]'s geometry, which its config and the model answer
/// alike.
pub trait Qwen3TtsModelMeta {
    /// The talker's width, `W`.
    fn d_model(&self) -> usize;

    /// The codebooks a frame holds, `K + 1`: the talker's first and the code
    /// predictor's rest.
    fn code_groups(&self) -> usize;
}

impl Qwen3TtsModelMeta for Qwen3TtsModelConfig {
    fn d_model(&self) -> usize {
        self.talker.d_model()
    }

    fn code_groups(&self) -> usize {
        self.code_predictor.codebooks() + 1
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
/// [`from_generation`](Self::from_generation)).
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

impl Qwen3TtsTakeConfig {
    /// The settings of a `generation_config.json`.
    pub fn from_generation(generation: &Qwen3TtsGenerationJson) -> Self {
        Self::new()
            .with_talker(Qwen3TtsSamplerConfig::talker(generation))
            .with_code_predictor(Qwen3TtsSamplerConfig::code_predictor(generation))
            .with_max_frames(generation.max_new_tokens.saturating_sub(1))
    }
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
}

impl Qwen3TtsModel {
    /// A take: the codes of `prompt`, chosen as `config` says (its samplers,
    /// [`Qwen3TtsSampler`](super::sampler::Qwen3TtsSampler)), ids from `ids`.
    ///
    /// The draws run on the backend's random numbers: `device.seed(..)`
    /// before the call repeats a sampled take.
    pub fn generate_codes(
        &self,
        prompt: &Qwen3TtsPrompt,
        ids: &Qwen3TtsPromptConfig,
        config: &Qwen3TtsTakeConfig,
    ) -> Qwen3TtsCodes {
        let vocab = self.talker.lm.n_logits();
        let eos = i64::from(ids.codec_eos);
        let control = vocab.saturating_sub(QWEN3_TTS_CONTROL_ROWS) as i64..vocab as i64;
        let mut chooser = SampledChoices {
            talker: config.talker.init(control.filter(|&id| id != eos)),
            code_predictor: config.code_predictor.init([]),
            history: Vec::new(),
            frame: Vec::new(),
        };
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
        let device = prompt.embeds.device();
        let talker = &self.talker;
        let predictor = &self.code_predictor;
        let vocab = talker.lm.n_logits();
        let eos = i64::from(ids.codec_eos);

        let l = prompt.embeds.dims()[1];
        let r_emb = talker.rotary(l + max_frames, &device);
        let mut cache = talker.kv_cache_config().init(&device);
        let pad = talker.text_embed(Tensor::from_data(
            TensorData::new(vec![i64::from(ids.tts_pad)], [1, 1]),
            &device,
        ));
        let trailing = prompt.trailing.dims()[1];

        let last = |hidden: Tensor<3>| {
            let t = hidden.dims()[1];
            hidden.slice(s![.., t - 1..t, ..])
        };
        let mut hidden =
            last(talker.forward_hidden(prompt.embeds.clone(), &r_emb, Some(&mut cache)));
        let mut frames = Vec::new();
        loop {
            let f = frames.len();
            if f == max_frames {
                return Qwen3TtsCodes {
                    frames,
                    stop: Qwen3TtsStop::MaxFrames,
                };
            }
            let logits = talker.logits(hidden.clone()).reshape([1, vocab]);
            let blocked: &[i64] = if f < min_frames { &[eos] } else { &[] };
            let first = chooser.codebook0(f, logits, blocked);
            if first == eos {
                return Qwen3TtsCodes {
                    frames,
                    stop: Qwen3TtsStop::Eos,
                };
            }

            let code0 = Tensor::<2, Int>::from_data(TensorData::new(vec![first], [1, 1]), &device);
            let embed0 = talker.codec_embed(code0);
            let context = Tensor::cat(vec![hidden, embed0.clone()], 1);
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
            row = row
                + if f < trailing {
                    prompt.trailing.clone().slice(s![.., f..=f, ..])
                } else {
                    pad.clone()
                };
            frames.push(frame);
            hidden = talker.forward_hidden(row, &r_emb, Some(&mut cache));
        }
    }
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

/// [`Qwen3TtsModel::generate_codes`]'s chooser: the samplers, the talker's
/// repetition penalty over the take's codebook-0 history, the predictor's
/// over the frame's codes.
struct SampledChoices {
    talker: super::sampler::Qwen3TtsSampler,
    code_predictor: super::sampler::Qwen3TtsSampler,
    history: Vec<i64>,
    frame: Vec<i64>,
}

impl Qwen3TtsChooser for SampledChoices {
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
        kits::language::qwen3::Qwen3Config,
        support::testing::cpu_device,
    };

    /// The config and the model built from it answer the meta alike.
    #[test]
    fn test_model_meta() {
        let config = Qwen3TtsModelConfig::new(
            Qwen3TtsTalkerConfig::new(Qwen3Config::new(11, 8, 1, 2, 1, 4, 12), 13, 6, 4),
            Qwen3TtsCodePredictorConfig::new(Qwen3Config::new(7, 4, 1, 2, 1, 2, 6), 8, 3),
        );
        let model: Qwen3TtsModel = config.init(&cpu_device());
        for meta in [&config as &dyn Qwen3TtsModelMeta, &model] {
            assert_eq!([meta.d_model(), meta.code_groups()], [8, 4]);
        }
    }
}
