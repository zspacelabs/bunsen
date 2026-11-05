//! The toy model against `transformers`' own forward.
//!
//! `testdata/modernbert/` holds a tiny seeded `ModernBertModel`'s weights and
//! its answers, computed in float64 (`tools/gen_modernbert_toy_fixtures.py`):
//! local layers among global ones with a two-sided window of ±2 against 12
//! tokens, two rotary bases, random norm gains, and a padded batch, whose
//! references are each item run alone, unpadded: what padding must not
//! change. Its geometry comes from its `config.json` through
//! [`modernbert_config_from_transformers`], and its weights, `Wqkv` and `Wi`
//! fused, load through [`load_transformers_tensors`], so the reader, the
//! splits and the loader are under test too.

use std::path::PathBuf;

use burn::prelude::*;

use super::{
    ModernBertModel,
    weights::{
        load_transformers_tensors,
        modernbert_config_from_transformers,
    },
};
use crate::{
    burner::module::ModuleInit,
    support::testing::{
        ToyFixture,
        assert_io_close,
        performance_device,
    },
};

struct Toy {
    model: ModernBertModel,
    fixture: ToyFixture,
    lengths: Vec<usize>,
}

impl Toy {
    fn load(device: &Device) -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/modernbert");
        let json = std::fs::read_to_string(dir.join("modernbert_toy.config.json"))
            .expect("read the toy config");
        let config = modernbert_config_from_transformers(&json).expect("parse the toy config");
        let mut fixture = ToyFixture::load(dir.join("modernbert_toy.safetensors"), device);
        let mut model: ModernBertModel = config.init(device);
        let applied = load_transformers_tensors(&mut model, fixture.take_weights(), None)
            .expect("load the toy weights");
        assert!(applied.unused.is_empty(), "{:?}", applied.unused);
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

    /// `x`, and the fixture's `key`, cut to item `row`'s valid positions.
    fn valid(
        &self,
        row: usize,
        x: Tensor<3>,
    ) -> Tensor<3> {
        x.slice(s![row..row + 1, 0..self.lengths[row]])
    }
}

/// The encoder is `transformers`' forward: the first two layers (the first,
/// global and without an attention norm; then a local one) and the final
/// hidden states, at every item's valid positions.
#[test]
fn test_toy_forward() {
    let toy = Toy::load(&performance_device());
    let model = &toy.model;
    let ids = toy.fixture.ids();
    let [_, seq] = ids.dims();
    let device = ids.device();

    let embeds = model.embeddings.forward(ids.clone());
    let r_emb = model.stack.rotary(seq, &device);
    let valid = crate::ops::transformers::attention::length_mask(&toy.lengths, seq, &device);
    let sides = || crate::blocks::transformers::decoders::causal::DecoderSideInputs {
        key_valid: Some(valid.clone()),
        ..Default::default()
    };
    let layer0 = model.stack.layers[0].forward_with(embeds, r_emb.for_layer(0), &mut None, sides());
    let layer1 =
        model.stack.layers[1].forward_with(layer0.clone(), r_emb.for_layer(1), &mut None, sides());
    let hidden = model.forward(ids, Some(&toy.lengths));
    for row in 0..toy.lengths.len() {
        for (name, ours) in [
            ("layer0", &layer0),
            ("layer1", &layer1),
            ("hidden", &hidden),
        ] {
            assert_io_close(
                &format!("item {row} {name}"),
                toy.valid(row, ours.clone()),
                toy.valid(row, toy.fixture.io(name)),
            );
        }
    }
}

/// An item alone, unpadded, encodes as it does in the padded batch: the
/// padding reaches no valid position.
#[test]
fn test_toy_padding_is_invisible() {
    let toy = Toy::load(&performance_device());
    let ids = toy.fixture.ids();
    let n = toy.lengths[1];
    let batched = toy.valid(1, toy.model.forward(ids.clone(), Some(&toy.lengths)));
    let alone = toy.model.forward(ids.slice(s![1..2, 0..n]), None);
    assert!(
        !batched.clone().is_nan().any().into_scalar::<bool>(),
        "batched: NaN"
    );
    assert!(
        !alone.clone().is_nan().any().into_scalar::<bool>(),
        "alone: NaN"
    );
    assert_io_close("item 1 alone", alone, batched);
}
