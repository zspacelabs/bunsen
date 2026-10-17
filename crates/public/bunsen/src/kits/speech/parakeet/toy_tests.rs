//! The toy model against `transformers`' own forward.
//!
//! `testdata/parakeet/toy.safetensors` holds a tiny seeded `ParakeetForTDT`'s
//! weights and its answers, computed in float64 by `transformers`
//! (`tools/gen_parakeet_toy_fixtures.py`). The weights load through
//! [`load_hf_tensors`], so the loader is under test too.

use alloc::collections::BTreeMap;

use burn::{
    prelude::*,
    store::{
        ModuleStore,
        SafetensorsStore,
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::Tolerance,
};

use super::{
    blocks::{
        ParakeetTdt,
        ParakeetTdtConfig,
    },
    weights::load_hf_tensors,
};
use crate::{
    blocks::conformer::{
        ConformerBlockConfig,
        ConformerEncoderConfig,
        DwStridingSubsamplingConfig,
    },
    burner::module::ModuleInit,
    ops::transformers::attention::rel_position_table,
    support::testing::performance_device,
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
    io: BTreeMap<String, PackTensor>,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/parakeet/toy.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
        let all = store
            .get_all_tensors()
            .expect("read the toy fixture")
            .clone();

        let mut weights = Vec::new();
        let mut io = BTreeMap::new();
        for (name, tensor) in all {
            if let Some(param) = name.strip_prefix("model.") {
                let mut tensor = tensor;
                tensor.name = param.to_string();
                weights.push(tensor);
            } else if let Some(key) = name.strip_prefix("io.") {
                io.insert(key.to_string(), tensor);
            }
        }

        let device = performance_device();
        let mut model: ParakeetTdt = toy_config().init(&device);
        let applied = load_hf_tensors(&mut model, weights).expect("load the toy weights");
        assert!(
            applied
                .unused
                .iter()
                .all(|n| n.ends_with("num_batches_tracked")),
            "unused: {:?}",
            applied.unused
        );
        Self { model, io, device }
    }

    fn io<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D> {
        let data = bridge::to_data(&self.io[key]).expect("read an io tensor");
        Tensor::from_data(data, &self.device)
    }
}

fn assert_close<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
) {
    assert_eq!(actual.dims(), expected.dims(), "{what}: shape");
    let tolerance = Tolerance::<f32>::rel_abs(1e-3, 1e-4);
    eprintln!("checking {what}");
    actual
        .to_data_as::<f32>()
        .assert_approx_eq::<f32>(&expected.to_data_as::<f32>(), tolerance);
}

/// The valid frames of `[batch, time, ...]` item `b`.
fn valid<const D: usize>(
    x: Tensor<D>,
    b: usize,
    len: usize,
) -> Tensor<D> {
    x.slice_dim(0, b..b + 1).slice_dim(1, 0..len)
}

/// `[batch, time, n_mels]` (the fixture's layout) to `[batch, n_mels, time]`.
fn features(toy: &Toy) -> Tensor<3> {
    toy.io::<3>("features").swap_dims(1, 2)
}

#[test]
fn test_toy_positions() {
    let toy = Toy::load();
    let expected: Tensor<3> = toy.io("positions");
    let actual = rel_position_table(OUT_LENGTHS[0], 16, &toy.device);
    assert_close("positions", actual, expected.slice_dim(0, 0..1));
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
    assert_close("subsampling", x, toy.io("subsampling"));
}

/// Each module of the first block, on the subsampled frames, unmasked.
#[test]
fn test_toy_block_modules() {
    let toy = Toy::load();
    let x: Tensor<3> = toy.io("subsampling");
    let layer = &toy.model.encoder.layers[0];
    let positions = rel_position_table(OUT_LENGTHS[0], 16, &toy.device);

    let ff1 = layer
        .feed_forward1
        .forward(layer.norm_feed_forward1.forward(x.clone()));
    assert_close("feed_forward1", ff1, toy.io("block.ff1"));

    let attn = layer
        .self_attn
        .forward(layer.norm_self_att.forward(x.clone()), positions, None);
    assert_close("self_attn", attn, toy.io("block.attn"));

    let conv = layer.conv.forward(layer.norm_conv.forward(x), None);
    assert_close("conv", conv, toy.io("block.conv"));
}

/// The encoder, layer by layer and whole, on a padded batch: every valid
/// frame matches.
#[test]
fn test_toy_encoder() {
    let toy = Toy::load();
    let (encoded, lengths) = toy.model.encoder.forward(features(&toy), Some(&LENGTHS));
    assert_eq!(lengths, OUT_LENGTHS);
    let expected: Tensor<3> = toy.io("encoder");
    for (b, &len) in OUT_LENGTHS.iter().enumerate() {
        assert_close(
            &format!("encoder item {b}"),
            valid(encoded.clone(), b, len),
            valid(expected.clone(), b, len),
        );
    }

    // Item 1 alone, unpadded, encodes as it does in the batch.
    let alone = features(&toy)
        .slice_dim(0, 1..2)
        .slice_dim(2, 0..LENGTHS[1]);
    let (alone, _) = toy.model.encoder.forward(alone, None);
    assert_close(
        "encoder item 1 alone",
        alone,
        valid(expected, 1, OUT_LENGTHS[1]),
    );
}

/// The projection, the prediction network, and the joint over the full
/// lattice.
#[test]
fn test_toy_transducer() {
    let toy = Toy::load();
    let (projected, _) = toy.model.encode(features(&toy), Some(&LENGTHS));
    let expected: Tensor<3> = toy.io("projected");
    for (b, &len) in OUT_LENGTHS.iter().enumerate() {
        assert_close(
            &format!("projected item {b}"),
            valid(projected.clone(), b, len),
            valid(expected.clone(), b, len),
        );
    }

    let tokens = toy.io::<2>("tokens").int();
    let decoded = toy.model.decoder.forward(tokens);
    assert_close("decoder", decoded.clone(), toy.io("decoder"));

    let logits = toy.model.joint.forward(
        projected.unsqueeze_dim::<4>(2),
        decoded.unsqueeze_dim::<4>(1),
    );
    let expected: Tensor<4> = toy.io("logits");
    for (b, &len) in OUT_LENGTHS.iter().enumerate() {
        assert_close(
            &format!("logits item {b}"),
            valid(logits.clone(), b, len),
            valid(expected.clone(), b, len),
        );
    }
}
