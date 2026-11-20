//! The toy talker against `qwen_tts`' own.
//!
//! `testdata/qwen3_tts/toy.safetensors` holds a tiny
//! `Qwen3TTSForConditionalGeneration` under the model file's names and, per
//! prompt mode, the talker's prefill input, each decode step's input and its
//! raw logits at the prefill and at each step, computed in float64
//! (`tools/gen_qwen3_tts_toy_fixtures.py`); and the first mode's prefill
//! through the stack. The geometry comes from `toy.config.json`'s
//! `model.talker_config`, and the weights load through
//! [`load_talker_tensors`].

use std::path::PathBuf;

use burn::prelude::*;

use super::Qwen3TtsTalker;
use crate::{
    burner::module::ModuleInit,
    kits::{
        speech::tts::qwen3_tts::interop::qwen_tts::*,
        support::interop::transformers::TransformersConfigJson,
    },
    ops::transformers::attention::{
        KVCacheModel,
        testing::KV_CACHE_STORAGES,
    },
    support::testing::{
        ToyFixture,
        assert_io_close,
        performance_device,
    },
};

/// The fixture's prompt modes.
pub(crate) const MODES: [&str; 6] = [
    "english",
    "english_upfront",
    "auto",
    "dialect",
    "instructed",
    "described",
];

fn load() -> (Qwen3TtsTalker, ToyFixture) {
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
    let config = talker_config(&section).expect("the toy's config");
    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    let mut model: Qwen3TtsTalker = config.init(&device);
    load_talker_tensors(&mut model, fixture.take_weights(), None).expect("load the toy weights");
    (model, fixture)
}

/// The first mode's prefill, through the stack and its final norm.
#[test]
fn test_toy_prefill_hidden() {
    let (talker, fixture) = load();
    let embeds = fixture.io::<3>("english.prompt_embeds");
    let r_emb = talker.rotary(embeds.dims()[1], fixture.device());
    assert_io_close(
        "prefill hidden",
        talker.forward_hidden(embeds, &r_emb, None),
        fixture.io("prefill.norm"),
    );
}

/// Every mode's take: the prefill's last logits and each step's, through
/// a KV cache of every storage, and uncached over the whole take.
#[test]
fn test_toy_take_logits() {
    let (talker, fixture) = load();
    for mode in MODES {
        let io = |name: &str| format!("{mode}.{name}");
        let prompt = fixture.io::<3>(&io("prompt_embeds"));
        let steps = fixture.io::<2>(&io("step_embeds")).unsqueeze_dim::<3>(0);
        let expected = fixture.io::<2>(&io("talker_logits"));
        let [_, l, _] = prompt.dims();
        let n = steps.dims()[1];
        let rows = Tensor::cat(vec![prompt.clone(), steps.clone()], 1);
        let r_emb = talker.rotary(l + n, fixture.device());

        let uncached = talker.logits(talker.forward_hidden(rows, &r_emb, None));
        assert_io_close(
            &format!("{mode}: uncached"),
            uncached.slice(s![0..1, l - 1..l + n, ..]).squeeze_dim(0),
            expected.clone(),
        );

        for storage in KV_CACHE_STORAGES {
            let mut cache = talker
                .kv_cache_config()
                .with_seq_len(2)
                .with_chunk_size(2)
                .with_storage(storage)
                .init(&prompt.device());
            let prefill =
                talker.logits(talker.forward_hidden(prompt.clone(), &r_emb, Some(&mut cache)));
            let mut parts = vec![prefill.slice(s![.., l - 1..l, ..])];
            for k in 0..n {
                let step = steps.clone().slice(s![.., k..=k, ..]);
                parts.push(talker.logits(talker.forward_hidden(step, &r_emb, Some(&mut cache))));
            }
            assert_eq!(
                cache.pos(),
                l + n,
                "{mode}, {storage:?}: the cache's length"
            );
            assert_io_close(
                &format!("{mode}, {storage:?}: cached"),
                Tensor::cat(parts, 1).squeeze_dim(0),
                expected.clone(),
            );
        }
    }
}
