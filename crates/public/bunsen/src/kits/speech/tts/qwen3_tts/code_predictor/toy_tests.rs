//! The toy code predictor against `qwen_tts`' own, teacher-forced.
//!
//! `testdata/qwen3_tts/toy.safetensors` holds, per prompt mode, a greedy
//! take's codes and the predictor's raw logits for codebooks 1-3 of each
//! frame (`tools/gen_qwen3_tts_toy_fixtures.py`). Each frame here starts
//! from the talker's hidden state at the row that chose its codebook 0 and
//! that codebook's embedding, as upstream's does, and is dictated the take's
//! codes; its logits are upstream's. The geometry comes from
//! `toy.config.json`'s `model.talker_config`.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    Qwen3TtsCodePredictorConfig,
    weights::load_qwen3_tts_code_predictor_tensors,
};
use crate::{
    blocks::transformers::decoders::depth::DepthDecoder,
    burner::module::ModuleInit,
    kits::{
        speech::tts::qwen3_tts::talker::{
            Qwen3TtsTalker,
            Qwen3TtsTalkerConfig,
            Qwen3TtsTalkerMeta,
            weights::load_qwen3_tts_talker_tensors,
        },
        support::interop::transformers::TransformersConfigJson,
    },
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

fn load() -> (Qwen3TtsTalker, DepthDecoder, ToyFixture) {
    let device = performance_device();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/qwen3_tts");
    let json = TransformersConfigJson::parse(
        &std::fs::read_to_string(dir.join("toy.config.json")).expect("read the toy config"),
    )
    .expect("parse the toy config");
    let section = json
        .section("model")
        .and_then(|model| model.section("talker_config"))
        .expect("the toy's talker_config");
    let talker_config =
        Qwen3TtsTalkerConfig::from_transformers_section(&section).expect("the toy's talker");
    let predictor_config = Qwen3TtsCodePredictorConfig::from_transformers_section(
        &section
            .section("code_predictor_config")
            .expect("the toy's code_predictor_config"),
        talker_config.lm.d_model,
    )
    .expect("the toy's code predictor");
    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    let weights = fixture.take_weights();
    let mut talker: Qwen3TtsTalker = talker_config.init(&device);
    load_qwen3_tts_talker_tensors(&mut talker, weights.clone(), None).expect("the talker");
    let mut predictor: DepthDecoder = predictor_config.init(&device);
    load_qwen3_tts_code_predictor_tensors(&mut predictor, weights, None).expect("the predictor");
    (talker, predictor, fixture)
}

/// The toy predictor is narrower than its talker, so it projects (1.7B's
/// shape), and predicts the frame's codebooks after the first.
#[test]
fn test_toy_geometry() {
    let (talker, predictor, _) = load();
    assert!(predictor.input_proj.is_some());
    assert_eq!(predictor.tables[0].weight.dims()[1], talker.d_model());
    assert_eq!(predictor.heads.len(), 3);
}

/// Every mode's frames, teacher-forced: each codebook's logits are
/// upstream's.
#[test]
fn test_toy_frames_teacher_forced() {
    let (talker, predictor, fixture) = load();
    let device = fixture.device().clone();
    for mode in MODES {
        let io = |name: &str| format!("{mode}.{name}");
        let prompt = fixture.io::<3>(&io("prompt_embeds"));
        let steps = fixture.io::<2>(&io("step_embeds")).unsqueeze_dim::<3>(0);
        let codes = fixture.io::<2>(&io("codes")).int();
        let expected = fixture.io::<3>(&io("predictor_logits"));
        let [frames, groups] = codes.dims();
        let l = prompt.dims()[1];
        let rows = Tensor::cat(vec![prompt, steps], 1);
        let r_emb = talker.rotary(rows.dims()[1], &device);
        let hidden = talker.forward_hidden(rows, &r_emb, None);
        for f in 0..frames {
            let at = l - 1 + f;
            let code0 = codes.clone().slice(s![f..=f, 0..1]);
            let context = Tensor::cat(
                vec![
                    hidden.clone().slice(s![.., at..=at, ..]),
                    talker.codec_embed(code0),
                ],
                1,
            );
            let mut logits = Vec::new();
            let chosen = predictor.decode(context, |k, row| {
                logits.push(row);
                codes.clone().slice(s![f..=f, k..=k]).reshape([1])
            });
            assert_eq!(
                chosen.into_data(),
                codes.clone().slice(s![f..=f, 1..groups]).into_data()
            );
            assert_io_close(
                &format!("{mode}, frame {f}"),
                Tensor::cat(logits, 0),
                expected.clone().slice(s![f..=f, .., ..]).squeeze_dim(0),
            );
        }
    }
}
