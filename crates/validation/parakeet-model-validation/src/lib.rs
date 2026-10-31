//! # Parakeet TDT cross-check against a pretrained ONNX reference.
//!
//! Validates [`bunsen`]'s Parakeet TDT implementation against an independent
//! one, generated from `istupakov/parakeet-tdt-0.6b-v3-onnx` (`NeMo`'s own
//! export of NVIDIA's `parakeet-tdt-0.6b-v3`) by `burn_onnx::ModelGen`.
//!
//! ## This crate hosts nothing, and generates nothing
//!
//! The generated reference comes from `bind-parakeet-onnx`'s `onnx_gen`
//! feature, which fetches ~2.5 GB of assets, pins each to a SHA-256, and
//! caches them — all under the **`download`** feature, which is off by default
//! so an ordinary `cargo build --workspace` never reaches the network.
//!
//! ```sh
//! cargo test --release -p parakeet-model-validation \
//!   --features download,gpu-tests,wgpu
//! ```
//!
//! `gpu-tests` compiles every test that loads a model, `download` fetches
//! what they compare against, and a backend feature says what they run on.
//! None is inferred from the others. Without both `download` and `gpu-tests`
//! the suite is the fixture-integrity checks, which load no model.
//!
//! `PARAKEET_ONNX_ENCODER`, `PARAKEET_ONNX_ENCODER_DATA` and
//! `PARAKEET_ONNX_DECODER_JOINT` point the build at local files instead; they
//! are read by `bind-parakeet-onnx`'s build script, not this crate's.
//!
//! ## The reference is itself checked
//!
//! `testdata/onnx_reference.json` records what onnxruntime, the export's own
//! runtime, computes for each graph on deterministic synthetic input
//! (`tools/gen_parakeet_fixtures.py` writes it). The `reference` tests hold
//! the generated models to it, so a defect in the ONNX import cannot pass for
//! a defect in bunsen, or hide one.

#![cfg_attr(not(feature = "download"), allow(unused))]

use bunsen::support::testing::TensorRecord;
use burn::prelude::*;
use serde::Deserialize;

/// The reference models, generated from the `istupakov` export.
#[cfg(feature = "download")]
pub mod reference {
    pub use bind_parakeet_onnx::onnx_gen::*;
}

/// The mel channel count of `parakeet-tdt-0.6b-v3`.
pub const N_MELS: usize = 128;

/// The encoder's model width.
pub const D_ENC: usize = 1024;

/// The prediction network's width, and the LSTM state width.
pub const D_PRED: usize = 640;

/// The blank token: one past the 8192 vocabulary tokens.
pub const BLANK: usize = 8192;

/// The TDT durations the joint scores, after the blank.
pub const N_DURATIONS: usize = 5;

/// The synthetic feature length, in mel frames: 4 s of 10 ms hops.
pub const N_FRAMES: usize = 400;

/// The fixture onnxruntime wrote: see the crate docs.
pub fn onnx_reference() -> OnnxReference {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/onnx_reference.json");
    let text = std::fs::read_to_string(path).expect("read the onnx reference fixture");
    serde_json::from_str(&text).expect("parse the onnx reference fixture")
}

/// `testdata/onnx_reference.json`.
#[derive(Debug, Deserialize)]
pub struct OnnxReference {
    /// The export and revision the fixture was made from.
    pub source: String,

    /// The feature-extractor graph on [`synthetic_waveform`].
    pub preprocessor: PreprocessorRecord,

    /// The encoder graph on [`synthetic_features`].
    pub encoder: EncoderRecord,

    /// The prediction-and-joint graph, one step per target.
    pub decoder_joint: DecoderJointRecord,

    /// A greedy decode of real speech.
    pub speech: SpeechRecord,
}

/// The feature extractor's record.
#[derive(Debug, Deserialize)]
pub struct PreprocessorRecord {
    /// The synthetic waveform's length, in samples.
    pub n_samples: usize,

    /// `features [1, 128, T]`.
    pub features: TensorRecord,

    /// `features_lens [1]`.
    pub features_lens: Vec<i64>,
}

/// The encoder's record.
#[derive(Debug, Deserialize)]
pub struct EncoderRecord {
    /// The synthetic input's length, in mel frames.
    pub n_frames: usize,

    /// `outputs [1, 1024, T/8]`.
    pub outputs: TensorRecord,

    /// `encoded_lengths [1]`.
    pub encoded_lengths: Vec<i64>,
}

/// The prediction-and-joint graph's record.
#[derive(Debug, Deserialize)]
pub struct DecoderJointRecord {
    /// One step per target, each from its own synthetic state.
    pub steps: Vec<DecoderJointStep>,
}

/// One prediction-and-joint step.
#[derive(Debug, Deserialize)]
pub struct DecoderJointStep {
    /// The previous token fed to the prediction network.
    pub target: i64,

    /// `outputs [1, 1, 1, 8198]`.
    pub logits: TensorRecord,

    /// The argmax over the 8193 token logits.
    pub token_argmax: usize,

    /// The argmax over the 5 duration logits.
    pub duration_argmax: usize,

    /// The first LSTM state out, `[2, 1, 640]`.
    pub output_states_1: TensorRecord,

    /// The second LSTM state out, `[2, 1, 640]`.
    pub output_states_2: TensorRecord,
}

/// A greedy TDT decode of a clip.
#[derive(Debug, Deserialize)]
pub struct SpeechRecord {
    /// The clip's file name in `bunsen-test-audio`.
    pub clip: String,

    /// The clip's length, in samples.
    pub n_samples: usize,

    /// The encoder's output length.
    pub encoded_length: usize,

    /// The emitted tokens, blanks dropped.
    pub ids: Vec<usize>,

    /// The encoder frame each token was emitted at.
    pub frames: Vec<usize>,
}

/// A deterministic `[1, N_MELS, N_FRAMES]` stand-in for real features: the
/// formula `tools/gen_parakeet_fixtures.py` uses.
pub fn synthetic_features(device: &Device) -> Tensor<3> {
    let data: Vec<f32> = (0..N_MELS * N_FRAMES)
        .map(|k| {
            let (m, f) = (k / N_FRAMES, k % N_FRAMES);
            (((m * 7 + f * 13) % 211) as f64 / 105.0 - 1.0) as f32
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [1, N_MELS, N_FRAMES]), device)
}

/// A deterministic `[1, D_ENC, 1]` encoder frame.
pub fn synthetic_encoder_frame(device: &Device) -> Tensor<3> {
    let data: Vec<f32> = (0..D_ENC)
        .map(|c| (((c * 5) % 197) as f64 / 98.0 - 1.0) as f32 * 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [1, D_ENC, 1]), device)
}

/// A deterministic `[2, 1, D_PRED]` LSTM state; `salt` varies it.
pub fn synthetic_state(
    salt: usize,
    device: &Device,
) -> Tensor<3> {
    let data: Vec<f32> = (0..2 * D_PRED)
        .map(|k| ((((k * 3 + salt) % 101) as f64 / 50.0 - 1.0) * 0.25) as f32)
        .collect();
    Tensor::from_data(TensorData::new(data, [2, 1, D_PRED]), device)
}

/// A deterministic `[1, n_samples]` waveform: two tones and a ramp of noise.
pub fn synthetic_waveform(
    n_samples: usize,
    device: &Device,
) -> Tensor<2> {
    use std::f64::consts::PI;
    let data: Vec<f32> = (0..n_samples)
        .map(|n| {
            let n = n as f64;
            (0.30 * (2.0 * PI * 440.0 * n / 16000.0).sin()
                + 0.15 * (2.0 * PI * 1234.5 * n / 16000.0).sin()
                + 0.05 * (((n as u64 * 7919) % 1009) as f64 / 504.5 - 1.0)) as f32
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [1, n_samples]), device)
}

/// The pretrained cache the checkpoints come through.
///
/// Under `download`, it is rooted at `bind-parakeet-onnx`'s directory,
/// which its build laid out as one, and offline: nothing is fetched at run
/// time. Routing the checkpoints through the library's own factory is also
/// what this crate is validating. Without `download` it is the default
/// cache, which fetches on first use.
pub fn weights_cache() -> bunsen::data::pretrained::PretrainedCache {
    use bunsen::data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    };
    #[cfg(feature = "download")]
    let options = PretrainedCacheOptions::default()
        .with_disk(
            bunsen::data::cache::BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(bind_parakeet_onnx::cache_dir().to_path_buf())),
        )
        .with_offline(true);
    #[cfg(not(feature = "download"))]
    let options = PretrainedCacheOptions::default();
    PretrainedCache::new(options).expect("open the pretrained cache")
}

/// A Parakeet pretrained by name, through bunsen's factory and
/// [`weights_cache`].
pub fn load_bundle(
    name: &str,
    device: &Device,
) -> std::sync::Arc<bunsen::kits::speech::parakeet::ParakeetBundle> {
    bunsen::kits::speech::parakeet::pretrained::default_parakeet_factory()
        .expect("the parakeet factory")
        .load_bundle(name, &weights_cache(), device)
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// A clip from `bunsen-test-audio`, by file name, at 16 kHz.
pub fn speech_clip(name: &str) -> Vec<f32> {
    let path = bunsen_test_audio::by_file(name).path();
    bunsen::support::audio::load_audio_mono_sr(&path, 16000)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

/// The ground-truth transcript of a clip from `bunsen-test-audio`, by file
/// name.
///
/// # Panics
/// If the clip has no transcript.
pub fn speech_transcript(name: &str) -> String {
    bunsen_test_audio::by_file(name)
        .transcript()
        .unwrap_or_else(|| panic!("{name} has no transcript"))
}

/// `testdata/phonon2_jfk_moon.json`: Phonon-2's own reference path
/// (`transformers`) on `jfk_moon.mp3`, written by
/// `tools/gen_phonon2_fixture.py`.
#[derive(Debug, Deserialize)]
pub struct Phonon2Reference {
    /// The clip.
    pub clip: String,

    /// The transcript `transformers` decodes.
    pub text: String,

    /// The emitted tokens, blanks dropped.
    pub ids: Vec<usize>,

    /// The encoder frame each token was emitted at.
    pub frames: Vec<usize>,
}

/// Reads `testdata/phonon2_jfk_moon.json`.
pub fn phonon2_reference() -> Phonon2Reference {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/phonon2_jfk_moon.json"
    );
    let text = std::fs::read_to_string(path).expect("read the phonon-2 fixture");
    serde_json::from_str(&text).expect("parse the phonon-2 fixture")
}

/// The generated reference models, held to the onnxruntime fixture.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod reference_tests;

/// bunsen's Parakeet, stage by stage, against the generated reference.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod staged;

/// Transcriptions of real speech: bunsen against the references and the
/// ground truth.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod speech;

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture is self-consistent: every record's samples cover its
    /// shape, the graphs' shapes are the model's, and the speech decode's
    /// tokens are in the vocabulary at frames inside the encoder output.
    #[test]
    fn test_onnx_reference_fixture_integrity() {
        let fixture = onnx_reference();

        let mut records = vec![&fixture.preprocessor.features, &fixture.encoder.outputs];
        for step in &fixture.decoder_joint.steps {
            records.extend([&step.logits, &step.output_states_1, &step.output_states_2]);
        }
        for record in records {
            assert_eq!(record.samples.len(), record.expected_samples());
            assert!(record.abs_sum >= record.sum.abs());
        }

        assert_eq!(fixture.preprocessor.features.shape[..2], [1, N_MELS]);
        assert_eq!(fixture.encoder.n_frames, N_FRAMES);
        assert_eq!(fixture.encoder.outputs.shape, [1, D_ENC, N_FRAMES / 8]);
        assert_eq!(fixture.encoder.encoded_lengths, [(N_FRAMES / 8) as i64]);
        for step in &fixture.decoder_joint.steps {
            assert_eq!(step.logits.shape, [1, 1, 1, BLANK + 1 + N_DURATIONS]);
            assert!(step.token_argmax <= BLANK);
            assert!(step.duration_argmax < N_DURATIONS);
            assert_eq!(step.output_states_1.shape, [2, 1, D_PRED]);
        }

        let phonon = phonon2_reference();
        assert_eq!(phonon.clip, fixture.speech.clip);
        assert_eq!(phonon.ids.len(), phonon.frames.len());
        assert!(phonon.ids.iter().all(|&id| id < BLANK));
        assert!(phonon.frames.windows(2).all(|w| w[0] <= w[1]));
        assert!(phonon.text.starts_with("the moon."));

        let speech = &fixture.speech;
        assert!(!speech.ids.is_empty());
        assert_eq!(speech.ids.len(), speech.frames.len());
        assert!(speech.ids.iter().all(|&id| id < BLANK));
        assert!(speech.frames.windows(2).all(|w| w[0] <= w[1]));
        assert!(speech.frames.iter().all(|&f| f < speech.encoded_length));
    }
}
