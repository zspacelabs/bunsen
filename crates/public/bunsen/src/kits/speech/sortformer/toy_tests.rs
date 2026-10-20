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
    SortformerMelStream,
    SortformerStreamingConfig,
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

/// Streamed through the speaker cache, at a geometry that compresses it five
/// times, in pieces of any size: every output frame is what `transformers`'
/// offline mode (`NeMo`'s sync streaming) gives the whole input.
#[test]
#[serial_test::serial]
fn test_streaming_matches_transformers() {
    let toy = Toy::load();
    let model = std::sync::Arc::new(toy.model.clone());
    let mels: Tensor<3> = toy.get("stream.mels");
    let want: Tensor<3> = toy.get("stream.probs");
    let [_, frames, n_mels] = mels.dims();
    let mels: Tensor<2> = mels.reshape([frames, n_mels]);
    let config = SortformerStreamingConfig::new(3, 1, 6, 12, 4);

    for pieces in [&[frames][..], &[1][..], &[5, 13, 2][..], &[24][..]] {
        let mut stream = SortformerMelStream::new(model.clone(), config.clone()).unwrap();
        let mut got = Vec::new();
        let mut at = 0;
        for &n in pieces.iter().cycle() {
            if at >= frames {
                break;
            }
            let end = (at + n).min(frames);
            got.extend(stream.push(mels.clone().slice_dim(0, at..end)).unwrap());
            at = end;
        }
        got.extend(stream.finish().unwrap());
        assert_eq!(stream.cache().compressions(), 5, "{pieces:?}");

        let got = Tensor::<3>::from_data(TensorData::new(got, [1, frames, 3]), &toy.device);
        assert_close(&format!("streamed in {pieces:?}"), got, want.clone());
    }
}

/// Over samples, through `NeMo`'s mel front end streamed, the activity is the
/// one-shot front end's mels through the mel stream, however the samples are
/// split, and its frames are 10 ms, contiguous from sample 0.
#[test]
#[serial_test::serial]
fn test_sample_stream_is_the_mel_stream() {
    use crate::kits::speech::{
        diarization::SpeakerActivityStream,
        nemo::NemoMelFrontEndConfig,
    };

    let toy = Toy::load();
    let model = std::sync::Arc::new(toy.model.clone());
    let config = SortformerStreamingConfig::new(3, 1, 6, 12, 4);
    let front_end: crate::kits::speech::nemo::NemoMelFrontEnd =
        NemoMelFrontEndConfig::nemotron_diarization()
            .with_n_mels(24)
            .init(&toy.device);

    let samples: Vec<f32> = (0..16_000 * 2 + 777)
        .map(|n| {
            let t = n as f32 / 16_000.0;
            0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                + 0.05 * (((n * 7919) % 1009) as f32 / 504.5 - 1.0)
        })
        .collect();

    let (mels, lengths) = front_end.forward(
        Tensor::from_data(
            TensorData::new(samples.clone(), [1, samples.len()]),
            &toy.device,
        ),
        None,
    );
    let mels: Tensor<2> = mels
        .swap_dims(1, 2)
        .slice_dim(1, 0..lengths[0])
        .squeeze_dim(0);
    let mut one_shot = SortformerMelStream::new(model.clone(), config.clone()).unwrap();
    let mut want = one_shot.push(mels).unwrap();
    want.extend(one_shot.finish().unwrap());
    assert_eq!(want.len(), lengths[0] * 3);

    for piece in [samples.len(), 160, 1234] {
        let mut stream =
            super::SortformerStream::new(model.clone(), front_end.clone(), config.clone()).unwrap();
        let mut activity = crate::kits::speech::diarization::SpeakerActivity::empty(0, 160, 3);
        for chunk in samples.chunks(piece) {
            activity.append(stream.push(chunk).unwrap()).unwrap();
        }
        activity.append(stream.finish().unwrap()).unwrap();
        assert_eq!(activity.start, 0);
        assert_eq!(activity.frames(), lengths[0], "{piece}");
        let got = Tensor::<1>::from_floats(activity.probs.as_slice(), &toy.device);
        let want = Tensor::<1>::from_floats(want.as_slice(), &toy.device);
        assert_close(&format!("pieces of {piece}"), got, want);
    }
}
