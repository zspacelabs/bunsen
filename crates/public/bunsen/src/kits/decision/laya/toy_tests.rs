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
    LayaConfig,
    LayaModel,
    weights::load_laya_tensors,
};
use crate::{
    burner::module::ModuleInit,
    kits::text::encoders::modernbert::weights::modernbert_config_from_transformers,
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
