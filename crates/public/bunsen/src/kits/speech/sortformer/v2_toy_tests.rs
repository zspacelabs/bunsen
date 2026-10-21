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
    SilenceFill,
    SortformerMelStream,
    SortformerStreamingConfig,
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

/// `v2_toy.json`'s streamed case.
#[derive(Debug, Deserialize)]
struct Stream {
    mel_frames: usize,
    geometry: Geometry,
    compressions: usize,
    silent_frames: usize,
}

/// `NeMo`'s `sortformer_modules` geometry.
#[derive(Debug, Deserialize)]
struct Geometry {
    spkcache_len: usize,
    fifo_len: usize,
    chunk_len: usize,
    spkcache_update_period: usize,
    spkcache_sil_frames_per_spk: usize,
    chunk_left_context: usize,
    chunk_right_context: usize,
}

impl Geometry {
    fn config(&self) -> SortformerStreamingConfig {
        SortformerStreamingConfig::new(
            self.chunk_len,
            self.chunk_right_context,
            self.fifo_len,
            self.spkcache_len,
            self.spkcache_update_period,
        )
        .with_left_context(self.chunk_left_context)
        .with_silence_frames_per_speaker(self.spkcache_sil_frames_per_spk)
    }
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
    stream: Stream,
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
            stream: Stream,
        }
        let meta = std::fs::read_to_string(format!("{dir}/v2_toy.json")).expect("read v2_toy.json");
        let meta = serde_json::from_str::<Meta>(&meta).expect("parse v2_toy.json");
        Self {
            model,
            io,
            cases: meta.cases,
            stream: meta.stream,
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

/// The model streamed through the speaker cache is `NeMo`'s sync streaming,
/// however the mels are split: the probabilities, the compressions, and the
/// silent frames the running-mean silence fill took in.
#[test]
#[serial_test::serial]
fn test_streaming_matches_nemo() {
    let toy = Toy::load();
    let model = std::sync::Arc::new(toy.model);
    let mels: Tensor<3> = {
        let data = bridge::to_data(&toy.io["stream.mels"]).unwrap();
        Tensor::from_data(data, &toy.device)
    };
    let want: Tensor<3> = {
        let data = bridge::to_data(&toy.io["stream.probs"]).unwrap();
        Tensor::from_data(data, &toy.device)
    };
    let frames = toy.stream.mel_frames;
    assert_eq!(mels.dims(), [1, frames, 24]);
    let mels = mels.squeeze_dim::<2>(0);

    for piece in [frames, 37, 1] {
        let mut stream =
            SortformerMelStream::new(model.clone(), toy.stream.geometry.config()).unwrap();
        let mut out = Vec::new();
        let mut at = 0;
        while at < frames {
            let next = (at + piece).min(frames);
            out.extend(stream.push(mels.clone().slice_dim(0, at..next)).unwrap());
            at = next;
        }
        out.extend(stream.finish().unwrap());

        let got = Tensor::<1>::from_data(out.as_slice(), &toy.device).reshape(want.dims());
        got.into_data()
            .assert_approx_eq::<f32>(&want.clone().into_data(), Tolerance::rel_abs(1e-4, 1e-4));
        assert_eq!(
            stream.cache().compressions(),
            toy.stream.compressions,
            "{piece}-frame pieces: compressions"
        );
        let SilenceFill::RunningMean { frames: silent, .. } = stream.cache().silence() else {
            panic!("v2 fills its silence slots with a running mean")
        };
        assert_eq!(
            *silent, toy.stream.silent_frames,
            "{piece}-frame pieces: silent frames"
        );
    }
}
