//! The toy model against `transformers`' own forward.
//!
//! `testdata/sortformer/nemotron_toy.safetensors` holds a tiny seeded
//! `Nemotron3DiarizationForAudioFrameClassification`'s weights and its
//! answers, stage by stage, computed in float64 by `transformers`
//! (`tools/gen_nemotron_diar_toy_fixtures.py`). The weights load through
//! [`load_nemotron_hf_tensors`], so the loader is under test too.

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

use super::{
    blocks::{
        NemotronDiarizationConfig,
        Sortformer,
    },
    weights::load_nemotron_hf_tensors,
};
use crate::{
    burner::module::ModuleInit,
    support::testing::performance_device,
};

/// `nemotron_toy.json`'s config, as a [`NemotronDiarizationConfig`].
fn toy_config() -> NemotronDiarizationConfig {
    NemotronDiarizationConfig::new()
        .with_n_mels(24)
        .with_d_model(16)
        .with_n_layers(2)
        .with_n_heads(2)
        .with_ff_expansion(2)
        .with_d_hidden(8)
        .with_n_speakers(3)
}

struct Toy {
    model: Sortformer,
    io: BTreeMap<String, PackTensor>,
    device: Device,
}

impl Toy {
    fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/sortformer/nemotron_toy.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
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
        let applied = load_nemotron_hf_tensors(&mut model, weights).expect("load the toy weights");
        assert!(applied.unused.is_empty(), "{:?}", applied.unused);
        Self { model, io, device }
    }

    fn get<const D: usize>(
        &self,
        name: &str,
    ) -> Tensor<D> {
        let data = bridge::to_data(&self.io[name]).expect("read a fixture tensor");
        Tensor::from_data(data, &self.device)
    }
}

fn assert_close<const D: usize>(
    what: &str,
    got: Tensor<D>,
    want: Tensor<D>,
) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    got.into_data()
        .assert_approx_eq::<f32>(&want.into_data(), Tolerance::rel_abs(1e-4, 1e-4));
}

/// Every stage, at a length that fills its last stack and one that the
/// stacking pads, is the reference's.
#[test]
#[serial_test::serial]
fn test_stages_match_transformers() {
    let toy = Toy::load();
    let model = &toy.model;
    for case in ["even", "ragged"] {
        let io = |stage: &str| format!("{case}.{stage}");
        let mels: Tensor<3> = toy.get(&io("mels"));

        let (stacked, _) = model.pre_encode(mels.clone(), None);
        assert_close(&io("stacked"), stacked.clone(), toy.get(&io("stacked")));

        let encoded = model.encoder.forward_pre_encoded(stacked.clone(), None);
        assert_close(&io("encoded"), encoded.clone(), toy.get(&io("encoded")));

        let projected = model.projection.forward(encoded);
        assert_close(
            &io("projected"),
            projected.clone(),
            toy.get(&io("projected")),
        );

        let upsampled = model.upsample.as_ref().unwrap().forward(projected);
        assert_close(&io("upsampled"), upsampled, toy.get(&io("upsampled")));

        let probs = model.forward(mels, None);
        assert_close(&io("probs"), probs.output, toy.get(&io("probs")));
        assert_close(&io("pooled"), probs.encoder_rate, toy.get(&io("pooled")));
    }
}

/// The learned silence embedding loads, as the speaker cache will read it.
#[test]
#[serial_test::serial]
fn test_silence_embedding_loads() {
    let toy = Toy::load();
    let silence = toy.model.silence.as_ref().expect("a learned silence").val();
    assert_eq!(silence.dims(), [16]);
    assert!(silence.abs().sum().into_scalar::<f32>() > 0.0);
}
