//! The toy model's takes against `qwen_tts`' own.
//!
//! `testdata/qwen3_tts/toy.safetensors` holds, per prompt mode, the
//! talker's prefill input, its trailing text, and the codes of upstream's
//! greedy take of at most 6 codebook-0 tokens
//! (`tools/gen_qwen3_tts_toy_fixtures.py`): 5 frames, the last token's frame
//! never completed. A greedy take here, from the same prompt and with the same
//! processors, is those codes.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    Qwen3TtsChooser,
    Qwen3TtsCodes,
    Qwen3TtsModel,
    Qwen3TtsModelConfig,
    Qwen3TtsStop,
    Qwen3TtsTakeConfig,
    prompt::{
        Qwen3TtsPrompt,
        Qwen3TtsPromptConfig,
        Qwen3TtsPromptJson,
    },
    sampler::Qwen3TtsSamplerConfig,
    weights::load_qwen3_tts_model_tensors,
};
use crate::{
    burner::module::ModuleInit,
    kits::support::interop::transformers::TransformersConfigJson,
    support::testing::{
        ToyFixture,
        assert_io_close,
        performance_device,
    },
};

const MODES: [&str; 6] = [
    "english",
    "english_upfront",
    "auto",
    "dialect",
    "instructed",
    "described",
];

struct Toy {
    model: Qwen3TtsModel,
    ids: Qwen3TtsPromptConfig,
    fixture: ToyFixture,
}

impl Toy {
    fn load() -> Self {
        let device = performance_device();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/qwen3_tts");
        let text = std::fs::read_to_string(dir.join("toy.config.json")).expect("read");
        let model_json = TransformersConfigJson::parse(&text)
            .and_then(|c| c.section("model"))
            .expect("the toy's model config");
        let config = Qwen3TtsModelConfig::from_transformers(&model_json).expect("the toy");
        let value: serde_json::Value = serde_json::from_str(&text).expect("parse");
        let ids: Qwen3TtsPromptJson =
            serde_json::from_value(value["model"].clone()).expect("the toy's ids");
        let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
        let mut model: Qwen3TtsModel = config.init(&device);
        load_qwen3_tts_model_tensors(&mut model, fixture.take_weights(), None)
            .expect("load the toy");
        Self {
            model,
            ids: Qwen3TtsPromptConfig::from_json(&ids).expect("the toy's prompt ids"),
            fixture,
        }
    }

    fn prompt(
        &self,
        mode: &str,
    ) -> Qwen3TtsPrompt {
        Qwen3TtsPrompt {
            embeds: self.fixture.io(&format!("{mode}.prompt_embeds")),
            trailing: self.fixture.io(&format!("{mode}.trailing_text")),
        }
    }

    fn take(
        &self,
        mode: &str,
        config: &Qwen3TtsTakeConfig,
    ) -> Qwen3TtsCodes {
        self.model
            .generate_codes(&self.prompt(mode), &self.ids, config)
    }
}

fn greedy() -> Qwen3TtsTakeConfig {
    Qwen3TtsTakeConfig::new()
        .with_talker(
            Qwen3TtsSamplerConfig::new()
                .with_sample(false)
                .with_repetition_penalty(1.05),
        )
        .with_code_predictor(Qwen3TtsSamplerConfig::new().with_sample(false))
        .with_max_frames(5)
}

/// Every mode's greedy take is upstream's, code for code.
#[test]
fn test_toy_greedy_takes() {
    let toy = Toy::load();
    for mode in MODES {
        let take = toy.take(mode, &greedy());
        let expected: Vec<Vec<u32>> = toy
            .fixture
            .io::<2>(&format!("{mode}.codes"))
            .iter_dim(0)
            .map(|row| row.into_data().iter::<f32>().map(|v| v as u32).collect())
            .collect();
        assert_eq!(take.frames, expected, "{mode}");
        assert_eq!(take.stop, Qwen3TtsStop::MaxFrames, "{mode}");
        assert_eq!(
            take.codec_input(toy.fixture.device()).dims(),
            [1, 4, expected.len()]
        );
    }
}

/// A sampled take stays in the codebooks' codes and its length, and EOS
/// cannot end it before its first frames.
#[test]
fn test_toy_sampled_take() {
    let toy = Toy::load();
    let config = Qwen3TtsTakeConfig::new()
        .with_max_frames(12)
        .with_min_frames(3);
    let take = toy.take("english", &config);
    assert!(take.frames.len() >= 3 && take.frames.len() <= 12);
    for frame in &take.frames {
        assert_eq!(frame.len(), 4);
        assert!(frame.iter().all(|&c| c < 32), "{frame:?}");
    }
    if take.frames.len() < 12 {
        assert_eq!(take.stop, Qwen3TtsStop::Eos);
    }
}

/// Dictates a take's codes, and keeps the logits it was shown.
struct Script {
    codes: Vec<Vec<u32>>,
    talker: Vec<Tensor<2>>,
    predictor: Vec<Tensor<2>>,
}

impl Qwen3TtsChooser for Script {
    fn codebook0(
        &mut self,
        frame: usize,
        logits: Tensor<2>,
        _blocked: &[i64],
    ) -> i64 {
        self.talker.push(logits);
        i64::from(self.codes[frame][0])
    }

    fn codebook(
        &mut self,
        frame: usize,
        k: usize,
        logits: Tensor<2>,
    ) -> i64 {
        self.predictor.push(logits);
        i64::from(self.codes[frame][k])
    }
}

/// A scripted take, dictated upstream's codes, sees upstream's logits:
/// the talker's at the prefill and each step, the predictor's every frame.
#[test]
fn test_toy_scripted_take() {
    let toy = Toy::load();
    for mode in MODES {
        let codes: Vec<Vec<u32>> = toy
            .fixture
            .io::<2>(&format!("{mode}.codes"))
            .iter_dim(0)
            .map(|row| row.into_data().iter::<f32>().map(|v| v as u32).collect())
            .collect();
        let mut script = Script {
            codes: codes.clone(),
            talker: Vec::new(),
            predictor: Vec::new(),
        };
        let take = toy
            .model
            .run_take(&toy.prompt(mode), &toy.ids, codes.len(), 2, &mut script);
        assert_eq!(take.frames, codes, "{mode}");
        let talker: Tensor<2> = toy.fixture.io(&format!("{mode}.talker_logits"));
        let frames = codes.len();
        assert_io_close(
            &format!("{mode}: talker"),
            Tensor::cat(script.talker, 0),
            talker.slice(s![0..frames, ..]),
        );
        let predictor: Tensor<3> = toy.fixture.io(&format!("{mode}.predictor_logits"));
        let [_, k, v] = predictor.dims();
        assert_io_close(
            &format!("{mode}: predictor"),
            Tensor::cat(script.predictor, 0),
            predictor
                .slice(s![0..frames, .., ..])
                .reshape([frames * k, v]),
        );
    }
}
