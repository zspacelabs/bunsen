//! The toy model against `transformers`' own forward.
//!
//! `testdata/parakeet/toy.safetensors` holds a tiny seeded `ParakeetForTDT`'s
//! weights and its answers, computed in float64 by `transformers`
//! (`tools/gen_parakeet_toy_fixtures.py`). The weights load through
//! [`load_parakeet_tdt_tensors`], so the loader is under test too.

use burn::prelude::*;

use super::{
    blocks::{
        ParakeetTdt,
        ParakeetTdtConfig,
    },
    interop::transformers::load_parakeet_tdt_tensors,
};
use crate::{
    blocks::{
        signals::DwStridingSubsamplingConfig,
        transformers::encoders::{
            EncoderTransformer,
            conformer::{
                ConformerBlockConfig,
                ConformerEncoderConfig,
            },
        },
    },
    burner::module::ModuleInit,
    ops::transformers::attention::rel_position_table,
    support::testing::{
        TOY,
        ToyFixture,
        assert_io_close,
        expect_variant,
        matching::TensorMatcher,
        performance_device,
        testdata,
    },
};

/// `toy.json`'s config, as a [`ParakeetTdtConfig`].
fn toy_config() -> ParakeetTdtConfig {
    ParakeetTdtConfig::new(
        ConformerEncoderConfig::new(
            DwStridingSubsamplingConfig::new(24, 4, 16),
            ConformerBlockConfig::new(16, 2, 32, 5)
                .with_bias(false)
                .with_conv_bias(false),
            2,
        ),
        11,
        12,
    )
}

/// The input lengths, in frames.
const LENGTHS: [usize; 2] = [53, 37];

/// The encoder's output lengths.
const OUT_LENGTHS: [usize; 2] = [7, 5];

struct Toy {
    model: ParakeetTdt,
    fixture: ToyFixture,
}

impl Toy {
    fn load() -> Self {
        let mut fixture = ToyFixture::load_prefixed(
            testdata!("parakeet/toy.safetensors"),
            "model.",
            &performance_device(),
        );
        let mut model: ParakeetTdt = toy_config().init(fixture.device());
        let applied = load_parakeet_tdt_tensors(&mut model, fixture.take_weights())
            .expect("load the toy weights");
        assert!(
            applied
                .unused
                .iter()
                .all(|n| n.ends_with("num_batches_tracked")),
            "unused: {:?}",
            applied.unused
        );
        Self { model, fixture }
    }

    fn io<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D> {
        self.fixture.io(key)
    }
}

/// `[batch, time, n_mels]` (the fixture's layout) to `[batch, n_mels, time]`.
fn features(toy: &Toy) -> Tensor<3> {
    toy.io::<3>("features").swap_dims(1, 2)
}

#[test]
fn test_toy_positions() {
    let toy = Toy::load();
    let expected: Tensor<3> = toy.io("positions");
    let actual = rel_position_table(OUT_LENGTHS[0], 16, toy.fixture.device());
    assert_io_close("positions", actual, expected.slice_dim(0, 0..1));
}

#[test]
fn test_toy_subsampling() {
    let toy = Toy::load();
    let (x, lengths) = toy
        .model
        .encoder
        .subsampling
        .forward(features(&toy), Some(&LENGTHS));
    assert_eq!(lengths, OUT_LENGTHS);
    assert_io_close("subsampling", x, toy.io("subsampling"));
}

/// Each module of the first block, on the subsampled frames, unmasked.
#[test]
fn test_toy_block_modules() {
    let toy = Toy::load();
    let x: Tensor<3> = toy.io("subsampling");
    expect_variant!(
        let EncoderTransformer::Conformer(stack) = &toy.model.encoder.transformer,
        "a FastConformer's stack is a Conformer's",
    );
    let layer = &stack.layers[0];
    let positions = rel_position_table(OUT_LENGTHS[0], 16, toy.fixture.device());

    let ff1 = layer
        .feed_forward1
        .branch
        .forward(layer.feed_forward1.pre.as_ref().unwrap().forward(x.clone()));
    assert_io_close("feed_forward1", ff1, toy.io("block.ff1"));

    let attn = layer.attn.branch.forward(
        layer.attn.pre.as_ref().unwrap().forward(x.clone()),
        positions,
        None,
    );
    assert_io_close("self_attn", attn, toy.io("block.attn"));

    let conv = layer
        .conv
        .branch
        .forward(layer.conv.pre.as_ref().unwrap().forward(x), None);
    assert_io_close("conv", conv, toy.io("block.conv"));
}

/// The encoder, layer by layer and whole, on a padded batch: every valid
/// frame matches.
#[test]
fn test_toy_encoder() {
    let toy = Toy::load();
    let (encoded, lengths) = toy.model.encoder.forward(features(&toy), Some(&LENGTHS));
    assert_eq!(lengths, OUT_LENGTHS);
    let expected: Tensor<3> = toy.io("encoder");
    TensorMatcher::close_to(&expected, TOY)
        .valid_prefix(1, &OUT_LENGTHS)
        .named("encoder")
        .assert(&encoded);

    // Item 1 alone, unpadded, encodes as it does in the batch.
    let alone = features(&toy)
        .slice_dim(0, 1..2)
        .slice_dim(2, 0..LENGTHS[1]);
    let (alone, _) = toy.model.encoder.forward(alone, None);
    assert_io_close(
        "encoder item 1 alone",
        alone,
        expected.slice_dim(0, 1..2).slice_dim(1, 0..OUT_LENGTHS[1]),
    );
}

/// The projection, the prediction network, and the joint over the full
/// lattice.
#[test]
fn test_toy_transducer() {
    let toy = Toy::load();
    let (projected, _) = toy.model.encode(features(&toy), Some(&LENGTHS));
    let expected: Tensor<3> = toy.io("projected");
    TensorMatcher::close_to(&expected, TOY)
        .valid_prefix(1, &OUT_LENGTHS)
        .named("projected")
        .assert(&projected);

    let tokens = toy.io::<2>("tokens").int();
    let decoded = toy.model.decoder.forward(tokens);
    assert_io_close("decoder", decoded.clone(), toy.io("decoder"));

    let logits = toy.model.joint.forward(
        projected.unsqueeze_dim::<4>(2),
        decoded.unsqueeze_dim::<4>(1),
    );
    let expected: Tensor<4> = toy.io("logits");
    TensorMatcher::close_to(&expected, TOY)
        .valid_prefix(1, &OUT_LENGTHS)
        .named("logits")
        .assert(&logits);
}
