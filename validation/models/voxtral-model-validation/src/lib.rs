//! # Voxtral Realtime cross-check against pretrained references.
//!
//! Validates [`bunsen`]'s Voxtral Realtime (`kits::speech::voxtral`) against
//! an independent implementation, generated from
//! `onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX` (an export of
//! `mistralai/Voxtral-Mini-4B-Realtime-2602`) by `burn_onnx::ModelGen`, and
//! against `transformers`' own greedy decodes of real speech.
//!
//! ## This crate hosts nothing, and generates nothing
//!
//! The generated reference comes from `bind-voxtral-onnx`'s `onnx_gen`
//! feature, and the checkpoint from its `checkpoint` feature, which fetch
//! their assets, pin each to a SHA-256, and cache them, all under the
//! **`download`** feature, off by default so an ordinary
//! `cargo build --workspace` never reaches the network.
//!
//! ```sh
//! cargo test --release -p voxtral-model-validation \
//!   --features download,gpu-tests,wgpu
//! ```
//!
//! `gpu-tests` compiles every test that loads a model, `download` fetches
//! what they compare against, and a backend feature says what they run on.
//! Without both `download` and `gpu-tests` the suite is the fixture-integrity
//! checks, which load no model.
//!
//! ## The reference is itself checked
//!
//! The export uses onnxruntime-only ops that `burn-onnx` cannot import, so
//! `bind-voxtral-onnx` rewrites them to standard ONNX first.
//! `testdata/onnx_reference.json` records what onnxruntime computes for each
//! *published* graph on deterministic synthetic input, and
//! `testdata/jfk_moon.json` what `transformers` decodes for real speech
//! (`validation/tools/gen_voxtral_fixtures.py` writes both). The generated
//! models are held to the first, so neither a defect in the rewrite nor one in
//! the ONNX import can pass for a defect in bunsen; bunsen is held to both.
//!
//! ## Memory
//!
//! The fp32 decoder graph is 13.8 GB, and bunsen's model 17.6 GB in fp32:
//! never both on one 24 GB card. A test loads one model, frees it
//! (`DeviceMemoryGuard`) before the next one loads, and the tests run one at
//! a time (`#[serial]`); bunsen is compared with the reference through
//! onnxruntime's recorded run, not side by side.

#![cfg_attr(not(feature = "download"), allow(unused))]

use bunsen::support::testing::TensorRecord;
use burn::prelude::*;
use serde::Deserialize;

/// The reference models, generated from the rewritten export.
#[cfg(feature = "download")]
pub mod reference {
    pub use bind_voxtral_onnx::onnx_gen::*;
}

#[cfg(feature = "download")]
mod reference_decode;
#[cfg(feature = "download")]
pub use reference_decode::*;

/// The mel channel count.
pub const N_MELS: usize = 128;

/// The text model's width, and the audio embeddings'.
pub const D_TEXT: usize = 3072;

/// The vocabulary size.
pub const N_VOCAB: usize = 131_072;

/// Mel frames per decoder position (80 ms).
pub const MEL_FRAMES_PER_TOKEN: usize = 8;

/// The audio encoder's layer count.
pub const ENCODER_LAYERS: usize = 32;

/// The audio encoder's heads (as many key/value heads as query heads).
pub const ENCODER_HEADS: usize = 32;

/// The audio encoder's head width.
pub const ENCODER_HEAD_DIM: usize = 64;

/// The packed conv carry the encoder graph reads and returns, `[1, 1408, 2]`:
/// `conv1`'s 128 input channels and `conv2`'s 1280, two frames each.
pub const PADDING_CACHE: [usize; 3] = [1, 1408, 2];

/// The text decoder's layer count.
pub const DECODER_LAYERS: usize = 26;

/// The text decoder's key/value heads.
pub const DECODER_KV_HEADS: usize = 8;

/// The text decoder's head width.
pub const DECODER_HEAD_DIM: usize = 128;

/// `testdata/onnx_reference.json`: onnxruntime's run of each published
/// graph.
#[derive(Debug, Deserialize)]
pub struct OnnxReference {
    /// The export and revision the fixture was made from.
    pub source: String,
    /// The audio encoder's run.
    pub encoder: EncoderReference,
    /// The token embedding's run.
    pub embed: EmbedReference,
    /// The text decoder's run.
    pub decoder: DecoderReference,
}

/// The audio encoder over `frames` synthetic mel frames from empty caches,
/// then over `step_frames` more through the caches it returned.
#[derive(Debug, Deserialize)]
pub struct EncoderReference {
    /// The first call's mel frames.
    pub frames: usize,
    /// The second call's mel frames, after the first's.
    pub step_frames: usize,
    /// `[1, frames/8, D_TEXT]`.
    pub embeds: TensorRecord,
    /// The carry after the first call, [`PADDING_CACHE`].
    pub padding_cache: TensorRecord,
    /// Layer 0's keys after the first call, `[1, 32, frames/2, 64]`.
    pub key_0: TensorRecord,
    /// Layer 31's values after the first call.
    pub value_31: TensorRecord,
    /// `[1, step_frames/8, D_TEXT]`.
    pub step_embeds: TensorRecord,
    /// The carry after the second call.
    pub step_padding_cache: TensorRecord,
    /// Layer 31's keys after the second call,
    /// `[1, 32, (frames+step_frames)/2, 64]`.
    pub step_key_31: TensorRecord,
}

/// The token embedding over fixed ids.
#[derive(Debug, Deserialize)]
pub struct EmbedReference {
    /// The ids.
    pub ids: Vec<i64>,
    /// `[1, ids, D_TEXT]`.
    pub embeds: TensorRecord,
}

/// The text decoder's prefill of `prefill` synthetic positions from an empty
/// cache, then one cached step.
#[derive(Debug, Deserialize)]
pub struct DecoderReference {
    /// The prefill's positions.
    pub prefill: usize,
    /// `[1, prefill, N_VOCAB]`.
    pub logits: TensorRecord,
    /// The prefill's greedy token at each position.
    pub argmax: Vec<i64>,
    /// Layer 0's keys after the prefill, `[1, 8, prefill, 128]`.
    pub key_0: TensorRecord,
    /// Layer 25's values after the prefill.
    pub value_25: TensorRecord,
    /// `[1, 1, N_VOCAB]`.
    pub step_logits: TensorRecord,
    /// The step's greedy token.
    pub step_argmax: i64,
    /// Layer 25's keys after the step, `[1, 8, prefill + 1, 128]`.
    pub step_key_25: TensorRecord,
}

/// Reads `testdata/onnx_reference.json`.
pub fn onnx_reference() -> OnnxReference {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/onnx_reference.json");
    let text = std::fs::read_to_string(path).expect("read the onnx reference fixture");
    serde_json::from_str(&text).expect("parse the onnx reference fixture")
}

/// A deterministic `[1, N_MELS, frames]` stand-in for log-mels, in the
/// extractor's range, from mel frame `offset` on: the formula
/// `validation/tools/gen_voxtral_fixtures.py` uses.
pub fn synthetic_features(
    frames: usize,
    offset: usize,
    device: &Device,
) -> Tensor<3> {
    let data: Vec<f32> = (0..N_MELS * frames)
        .map(|k| {
            let (m, f) = (k / frames, offset + k % frames);
            ((((m * 7 + f * 13) % 211) as f64 / 105.0 - 1.0) * 0.8 + 0.3) as f32
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [1, N_MELS, frames]), device)
}

/// A deterministic, broadband `[n]` waveform: the integer sawtooth
/// `validation/tools/gen_voxtral_fixtures.py` uses, exact in `f32`.
pub fn synthetic_wave(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| ((((k * 7919 + 13) % 2003) as f64 / 1001.5 - 1.0) * 0.25) as f32)
        .collect()
}

/// A deterministic `[1, seq, D_TEXT]` stand-in for the decoder's input
/// embeddings: the formula `validation/tools/gen_voxtral_fixtures.py` uses.
pub fn synthetic_embeds(
    seq: usize,
    salt: usize,
    device: &Device,
) -> Tensor<3> {
    let data: Vec<f32> = (0..seq * D_TEXT)
        .map(|k| ((((k * 5 + salt) % 197) as f64 / 98.0 - 1.0) * 0.05) as f32)
        .collect();
    Tensor::from_data(TensorData::new(data, [1, seq, D_TEXT]), device)
}

/// The top-1 minus top-2 logit under which a greedy step is a near-tie: two
/// float32 implementations may pick either token, and their decodes part
/// there.
pub const NEAR_TIE: f64 = 0.05;

/// One window's entry in `testdata/jfk_moon.json`: `transformers`' greedy
/// decode, in float32.
#[derive(Debug, Deserialize)]
pub struct WindowDecode {
    /// The window's first second.
    pub start: usize,
    /// One past its last second.
    pub end: usize,
    /// The processor's prompt.
    pub prompt_ids: Vec<i64>,
    /// The whole sequence, the prompt first.
    pub ids: Vec<i64>,
    /// Each generated id's top-1 minus top-2 logit.
    pub margins: Vec<f64>,
    /// The decoded text, special tokens skipped.
    pub text: String,
}

/// `testdata/jfk_moon.json`'s `front_end`: the processor and the audio
/// encoder on [`synthetic_wave`].
#[derive(Debug, Deserialize)]
pub struct FrontEndReference {
    /// The waveform's length.
    pub samples: usize,
    /// The processor's `[1, 128, frames]` features of it, padded as an
    /// utterance.
    pub input_features: TensorRecord,
    /// The model's `[1, positions, 3072]` audio embeddings of them.
    pub audio_embeds: TensorRecord,
}

/// `testdata/jfk_moon.json`.
#[derive(Debug, Deserialize)]
pub struct SpeechReference {
    /// The clip, in `bunsen-test-audio`.
    pub clip: String,
    /// The front end and the audio encoder on a synthetic waveform.
    pub front_end: FrontEndReference,
    /// The windows, in order.
    pub windows: Vec<WindowDecode>,
}

/// Reads `testdata/jfk_moon.json`.
pub fn speech_reference() -> SpeechReference {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/jfk_moon.json");
    let text = std::fs::read_to_string(path).expect("read the speech fixture");
    serde_json::from_str(&text).expect("parse the speech fixture")
}

/// A pretrained cache over the bundled checkpoint, offline.
///
/// Without `download` there is none: an ordinary cache, which the tests
/// that need it are not compiled without.
pub fn weights_cache() -> bunsen::data::pretrained::PretrainedCache {
    use bunsen::data::pretrained::{
        PretrainedCache,
        PretrainedCacheOptions,
    };
    #[cfg(feature = "download")]
    let options = PretrainedCacheOptions::default()
        .with_disk(
            bunsen::data::cache::BunsenDiskCacheOptions::default()
                .with_cache_dir(Some(bind_voxtral_onnx::cache_dir().to_path_buf())),
        )
        .with_offline(true);
    #[cfg(not(feature = "download"))]
    let options = PretrainedCacheOptions::default();
    PretrainedCache::new(options).expect("open the pretrained cache")
}

/// The published model, through bunsen's factory and [`weights_cache`].
pub fn load_bundle(
    device: &Device
) -> std::sync::Arc<bunsen::kits::speech::voxtral::VoxtralBundle> {
    use bunsen::kits::speech::voxtral::pretrained::{
        DEFAULT_VOXTRAL_MODEL,
        default_voxtral_factory,
    };
    default_voxtral_factory()
        .expect("the voxtral factory")
        .load_bundle(DEFAULT_VOXTRAL_MODEL, &weights_cache(), device)
        .unwrap_or_else(|e| panic!("load {DEFAULT_VOXTRAL_MODEL}: {e}"))
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

/// The generated models against onnxruntime's run of the published graphs.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod reference_tests;

/// bunsen's model against onnxruntime's run of the published graphs.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod cross;

/// bunsen's front end, audio embeddings and decodes of real speech against
/// `transformers`'.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod speech;

/// bunsen's vocabulary and transcript parser on `transformers`' decodes.
/// Reads the bundled `tekken.json`; loads no model.
#[cfg(all(test, feature = "download"))]
mod vocabulary;

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture is self-consistent: every record's samples cover its
    /// shape, and the shapes are the model's.
    #[test]
    fn test_fixture_integrity() {
        let fixture = onnx_reference();
        assert!(
            fixture
                .source
                .starts_with("onnx-community/Voxtral-Mini-4B-Realtime-2602-ONNX@"),
            "{}",
            fixture.source
        );

        let e = &fixture.encoder;
        let d = &fixture.decoder;
        for record in [
            &e.embeds,
            &e.padding_cache,
            &e.key_0,
            &e.value_31,
            &e.step_embeds,
            &e.step_padding_cache,
            &e.step_key_31,
            &fixture.embed.embeds,
            &d.logits,
            &d.key_0,
            &d.value_25,
            &d.step_logits,
            &d.step_key_25,
        ] {
            assert_eq!(record.samples.len(), record.expected_samples());
        }

        assert_eq!(e.frames % MEL_FRAMES_PER_TOKEN, 0);
        assert_eq!(e.step_frames % MEL_FRAMES_PER_TOKEN, 0);
        assert_eq!(e.embeds.shape, [1, e.frames / MEL_FRAMES_PER_TOKEN, D_TEXT]);
        assert_eq!(e.padding_cache.shape, PADDING_CACHE);
        assert_eq!(
            e.key_0.shape,
            [1, ENCODER_HEADS, e.frames / 2, ENCODER_HEAD_DIM]
        );
        assert_eq!(
            e.step_key_31.shape,
            [
                1,
                ENCODER_HEADS,
                (e.frames + e.step_frames) / 2,
                ENCODER_HEAD_DIM
            ]
        );
        // The first call reaches past the encoder's 750-frame window.
        assert!(e.frames / 2 > 750);

        assert_eq!(
            fixture.embed.embeds.shape,
            [1, fixture.embed.ids.len(), D_TEXT]
        );

        assert_eq!(d.logits.shape, [1, d.prefill, N_VOCAB]);
        assert_eq!(d.argmax.len(), d.prefill);
        assert_eq!(
            d.key_0.shape,
            [1, DECODER_KV_HEADS, d.prefill, DECODER_HEAD_DIM]
        );
        assert_eq!(d.step_logits.shape, [1, 1, N_VOCAB]);
        assert_eq!(
            d.step_key_25.shape,
            [1, DECODER_KV_HEADS, d.prefill + 1, DECODER_HEAD_DIM]
        );

        let speech = speech_reference();
        assert_eq!(speech.clip, "jfk_moon.mp3");
        assert!(!speech.windows.is_empty());
        for (k, w) in speech.windows.iter().enumerate() {
            assert!(w.ids.starts_with(&w.prompt_ids), "window {k}");
            assert_eq!(
                w.prompt_ids.len() + w.margins.len(),
                w.ids.len(),
                "window {k}"
            );
        }
        let f = &speech.front_end;
        for record in [&f.input_features, &f.audio_embeds] {
            assert_eq!(record.samples.len(), record.expected_samples());
        }
        assert_eq!(f.input_features.shape[..2], [1, N_MELS]);
        assert_eq!(
            f.audio_embeds.shape[1] * MEL_FRAMES_PER_TOKEN,
            f.input_features.shape[2]
        );
    }
}
