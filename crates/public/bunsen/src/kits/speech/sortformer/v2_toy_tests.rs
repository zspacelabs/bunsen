//! The toy Sortformer v2 against `NeMo`'s own forward.
//!
//! `testdata/sortformer/v2_toy.safetensors` holds a tiny seeded
//! `SortformerEncLabelModel` in Sortformer v2's layout, its weights under
//! `NeMo`'s names, and its answers stage by stage, computed in float64 by
//! `NeMo` (`tools/gen_sortformer_v2_toy_fixtures.py`). The weights load
//! through [`load_sortformer_v2_nemo_tensors`], so the loader is under test
//! too.

use std::collections::BTreeMap;

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
use serde::Deserialize;

use super::{
    blocks::{
        Sortformer,
        SortformerV2Config,
    },
    weights::load_sortformer_v2_nemo_tensors,
};
use crate::{
    burner::module::ModuleInit,
    support::testing::performance_device,
};

/// The fixture's toy, as a [`SortformerV2Config`].
fn toy_config() -> SortformerV2Config {
    SortformerV2Config::new()
        .with_n_mels(24)
        .with_d_model(16)
        .with_n_layers(2)
        .with_n_heads(2)
        .with_ff_expansion(2)
        .with_subsampling_channels(4)
        .with_conv_kernel_size(3)
        .with_d_hidden(8)
        .with_transformer_layers(2)
        .with_transformer_heads(2)
        .with_transformer_ff_expansion(2)
        .with_n_speakers(3)
}

/// `v2_toy.json`'s lengths of one case.
#[derive(Debug, Deserialize)]
struct Case {
    mel_lengths: Vec<usize>,
    encoder_lengths: Vec<usize>,
}

struct Toy {
    model: Sortformer,
    io: BTreeMap<String, PackTensor>,
    cases: BTreeMap<String, Case>,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/sortformer");
        let mut store = SafetensorsStore::from_file(format!("{dir}/v2_toy.safetensors"));
        let all = store
            .get_all_tensors()
            .expect("read the toy fixture")
            .clone();

        let mut weights = Vec::new();
        let mut io = BTreeMap::new();
        for (name, tensor) in all {
            if let Some(rest) = name.strip_prefix("weights.") {
                let mut tensor = tensor;
                tensor.name = rest.to_string();
                weights.push(tensor);
            } else if let Some(rest) = name.strip_prefix("io.") {
                io.insert(rest.to_string(), tensor);
            }
        }

        let device = performance_device();
        let mut model: Sortformer = toy_config().init(&device);
        let applied =
            load_sortformer_v2_nemo_tensors(&mut model, weights).expect("load the toy weights");
        // What the loader drops, and nothing else.
        for name in &applied.unused {
            assert!(
                name.ends_with(".num_batches_tracked")
                    || name.starts_with("preprocessor.featurizer.")
                    || name.starts_with("sortformer_modules.hidden_to_spks."),
                "unused: {name}"
            );
        }

        #[derive(Deserialize)]
        struct Meta {
            cases: BTreeMap<String, Case>,
        }
        let meta = std::fs::read_to_string(format!("{dir}/v2_toy.json")).expect("read v2_toy.json");
        let cases = serde_json::from_str::<Meta>(&meta)
            .expect("parse v2_toy.json")
            .cases;
        Self {
            model,
            io,
            cases,
            device,
        }
    }

    fn get<const D: usize>(
        &self,
        name: &str,
    ) -> Tensor<D> {
        let data = bridge::to_data(&self.io[name]).expect("read a fixture tensor");
        Tensor::from_data(data, &self.device)
    }
}

/// `got` against `want` on each item's first `lengths[b]` frames: past them
/// a padded batch's frames mean nothing.
fn assert_close(
    what: &str,
    got: Tensor<3>,
    want: Tensor<3>,
    lengths: &[usize],
) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let [_, _, width] = got.dims();
    for (b, &n) in lengths.iter().enumerate() {
        got.clone()
            .slice([b..b + 1, 0..n, 0..width])
            .into_data()
            .assert_approx_eq::<f32>(
                &want.clone().slice([b..b + 1, 0..n, 0..width]).into_data(),
                Tolerance::rel_abs(1e-4, 1e-4),
            );
    }
}

/// The toy's geometry is the fixture's.
#[test]
fn test_toy_config() {
    use super::SortformerMeta;
    let config = toy_config();
    assert_eq!(config.n_mels(), 24);
    assert_eq!(config.output_upsampling(), 1);
    let v2 = SortformerV2Config::diar_streaming_sortformer_4spk_v2();
    assert_eq!(
        (v2.n_layers, v2.d_model, v2.transformer_layers, v2.d_hidden),
        (17, 512, 18, 192)
    );
}

/// Every stage, at a length the subsampling fills, one it pads, and a padded
/// batch of two, is `NeMo`'s; the probabilities are zero past each length.
#[test]
#[serial_test::serial]
fn test_stages_match_nemo() {
    let toy = Toy::load();
    let model = &toy.model;
    for (case, lengths) in &toy.cases {
        let io = |stage: &str| format!("{case}.{stage}");
        let mels: Tensor<3> = toy.get(&io("mels"));
        let padded = lengths.mel_lengths.len() > 1;
        let mel_lengths = padded.then_some(lengths.mel_lengths.as_slice());
        let enc = &lengths.encoder_lengths;

        let (pre, pre_lengths) = model.pre_encode(mels.clone(), mel_lengths);
        assert_eq!(&pre_lengths, enc, "{case}: encoder lengths");
        assert_close(
            &io("pre_encoded"),
            pre.clone(),
            toy.get(&io("pre_encoded")),
            enc,
        );

        let enc_lengths = padded.then_some(enc.as_slice());
        let encoded = model.encoder.forward_pre_encoded(pre.clone(), enc_lengths);
        assert_close(
            &io("encoded"),
            encoded.clone(),
            toy.get(&io("encoded")),
            enc,
        );

        let projected = model.projection.forward(encoded);
        assert_close(
            &io("projected"),
            projected.clone(),
            toy.get(&io("projected")),
            enc,
        );

        let transformed = model
            .transformer
            .as_ref()
            .expect("v2 has a transformer")
            .forward(projected, enc_lengths);
        assert_close(
            &io("transformed"),
            transformed,
            toy.get(&io("transformed")),
            enc,
        );

        let probs = model.forward(mels, mel_lengths).output;
        let frames = probs.dims()[1];
        let full = vec![frames; enc.len()];
        assert_close(&io("probs"), probs, toy.get(&io("probs")), &full);
    }
}
