//! The toy model against upstream's own `DecisionModel`.
//!
//! `testdata/laya/` holds a tiny seeded Laya's weights and its answers,
//! computed in float64 (`tools/gen_laya_toy_fixtures.py`, upstream's
//! `DecisionModel` over a tiny `ModernBertModel`): a 3-option choice and a
//! yes/no question, padded into one batch, whose references are each prompt
//! run alone. The encoder's config comes from its `config.json`, and the
//! weights, every projection fused as upstream stores them, load through
//! [`load_laya_tensors`].

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    LayaAgentConfig,
    LayaConfig,
    LayaModel,
    LayaQuestion,
    LayaSpecials,
    weights::load_laya_tensors,
};
use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::{
        support::tokens::{
            Detokenizer,
            Tokenizer,
        },
        text::encoders::modernbert::weights::modernbert_config_from_transformers,
    },
    support::testing::{
        ToyFixture,
        assert_io_close,
        performance_device,
    },
};

struct Toy {
    model: LayaModel,
    fixture: ToyFixture,
    lengths: Vec<usize>,
}

impl Toy {
    fn load(device: &Device) -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/laya");
        let json = std::fs::read_to_string(dir.join("laya_toy.encoder.config.json"))
            .expect("read the toy encoder config");
        let encoder = modernbert_config_from_transformers(&json).expect("parse the toy config");
        let mut fixture = ToyFixture::load(dir.join("laya_toy.safetensors"), device);
        let mut model: LayaModel = LayaConfig::new(encoder).init(device);
        let applied =
            load_laya_tensors(&mut model, fixture.take_weights(), None).expect("load the toy");
        assert_eq!(applied.unused, ["temperature"]);
        let lengths: Tensor<1> = fixture.io("lengths");
        let lengths = lengths
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(|n| n as usize)
            .collect();
        Self {
            model,
            fixture,
            lengths,
        }
    }

    fn int<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D, Int> {
        let x: Tensor<D> = self.fixture.io(key);
        x.int()
    }
}

/// The head's states at each prompt's tokens, each question's option
/// logits (`-1e4` past its options), and the act logits are upstream's.
#[test]
fn test_toy_forward() {
    let toy = Toy::load(&performance_device());
    let ids = toy.fixture.ids();
    let kinds: Tensor<1, Int> = toy.int("kinds");
    let markers: Tensor<2, Int> = toy.int("markers");
    let options: Tensor<2> = toy.fixture.io("options");
    let options = options.greater_elem(0.5);

    let head = toy
        .model
        .head_states(ids.clone(), &toy.lengths, kinds.clone());
    let expected: Tensor<3> = toy.fixture.io("head");
    for (row, &n) in toy.lengths.iter().enumerate() {
        assert_io_close(
            &format!("prompt {row} head"),
            head.clone().slice(s![row..row + 1, 0..n]),
            expected.clone().slice(s![row..row + 1, 0..n]),
        );
    }

    let out = toy
        .model
        .forward(ids, &toy.lengths, kinds, markers, options);
    assert_io_close("logits", out.logits, toy.fixture.io("logits"));
    assert_io_close("act logits", out.act_logits, toy.fixture.io("act_logits"));
}

/// A tokenizer within the toy's 40 ids: `[PAD]` 0, `[CLS]` 1, `[SEP]` 2,
/// `[MASK]` 3, and each other byte `4 + byte % 36`, a special token's
/// spelling included. Ids render as their numbers: the tests read no text
/// back.
#[derive(Debug)]
struct Bytes;

impl Detokenizer for Bytes {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        Ok(ids.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
    }
}

impl Tokenizer for Bytes {
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Ok(text.bytes().map(|b| 4 + i64::from(b) % 36).collect())
    }

    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        self.encode(text)
    }

    fn contains(
        &self,
        id: i64,
    ) -> bool {
        (0..40).contains(&id)
    }

    fn len(&self) -> usize {
        40
    }

    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64> {
        ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
            .iter()
            .position(|s| *s == content)
            .map(|i| i as i64)
    }
}

/// Questions of every kind, decided together in one padded batch, are each
/// answered as when asked alone; each answer is a distribution over its
/// options, keyed as its kind keys them.
#[test]
fn test_decide_batches_alike() {
    let device = performance_device();
    let toy = Toy::load(&device);
    let specials = LayaSpecials::from_tokenizer_config(
        r#"{"cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]",
            "pad_token": "[PAD]"}"#,
        &Bytes,
    )
    .unwrap();
    let agent = LayaAgentConfig::parse(
        r#"{"head_layers": 2, "max_len": 60, "head_max_len": 40,
            "temperature": [1.5, 1.25, 2.0], "temperature_by_options": {"choice:3-5": 0.5}}"#,
    )
    .unwrap();
    let questions: Vec<LayaQuestion> = serde_json::from_str(
        r#"[
            {"type": "choice", "question": "which next?", "options": ["retry", "escalate", "close"]},
            {"type": "noul", "question": "is it done?"},
            {"type": "score", "question": "how bad?", "options": ["fine", "bad", "worse", "worst"]}
        ]"#,
    )
    .unwrap();
    let state = "the call failed twice";
    let together = toy
        .model
        .decide(&Bytes, &specials, &agent, state, &questions, &device)
        .unwrap();
    assert_eq!(together.len(), 3);
    for (q, answer) in questions.iter().zip(&together) {
        let alone = toy
            .model
            .decide(
                &Bytes,
                &specials,
                &agent,
                state,
                core::slice::from_ref(q),
                &device,
            )
            .unwrap();
        let keys: Vec<&str> = answer
            .probabilities
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys, q.option_keys());
        let sum: f64 = answer.probabilities.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-9, "{sum}");
        for ((_, p), (_, a)) in answer.probabilities.iter().zip(&alone[0].probabilities) {
            assert!((p - a).abs() < 1e-4, "{}: {p} alone {a}", q.question);
        }
        assert!((answer.act_probability - alone[0].act_probability).abs() < 1e-4);
    }
}
