//! # Qwen3-ASR cross-check against pretrained references.
//!
//! Validates [`bunsen`]'s Qwen3-ASR implementation against an independent
//! one, generated from `OpenVoiceOS/qwen3-asr-0.6b-onnx` (an export of
//! `Qwen/Qwen3-ASR-0.6B-hf`) by `burn_onnx::ModelGen`, and against
//! `transformers`' own decode of real speech.
//!
//! ## This crate hosts nothing, and generates nothing
//!
//! The generated reference comes from `bunsen-bundled-qwen3-asr`'s `onnx_gen`
//! feature, which fetches its assets, pins each to a SHA-256, and caches them
//! — all under the **`download`** feature, off by default so an ordinary
//! `cargo build --workspace` never reaches the network.
//!
//! ```sh
//! cargo test --release -p qwen3-asr-model-validation \
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
//! `testdata/onnx_reference.json` records what onnxruntime, the export's own
//! runtime, computes for each graph on deterministic synthetic input, and
//! `testdata/jfk_moon.json` what `transformers` decodes for real speech
//! (`tools/gen_qwen3_asr_fixtures.py` writes both). The generated models are
//! held to the first, so a defect in the ONNX import cannot pass for a defect
//! in bunsen.

#![cfg_attr(not(feature = "download"), allow(unused))]

use bunsen::support::testing::TensorRecord;
use burn::prelude::*;
use serde::Deserialize;

/// The reference models, generated from the `OpenVoiceOS` export.
#[cfg(feature = "download")]
pub mod reference {
    pub use bunsen_bundled_qwen3_asr::onnx_gen::*;
}

#[cfg(feature = "download")]
mod reference_decode;
#[cfg(feature = "download")]
pub use reference_decode::*;

/// The mel channel count.
pub const N_MELS: usize = 128;

/// Mel frames per encoder chunk.
pub const CHUNK: usize = 100;

/// Encoder frames per full chunk.
pub const FRAMES_PER_CHUNK: usize = 13;

/// Chunks per attention window.
pub const WINDOW_CHUNKS: usize = 8;

/// The text model's width.
pub const D_TEXT: usize = 1024;

/// The text model's layer count.
pub const N_LAYERS: usize = 28;

/// The text model's key/value heads.
pub const N_KV_HEADS: usize = 8;

/// The text model's head width.
pub const HEAD_DIM: usize = 128;

/// The vocabulary size.
pub const N_VOCAB: usize = 151_936;

/// `testdata/onnx_reference.json`.
#[derive(Debug, Deserialize)]
pub struct OnnxReference {
    /// The export and revision the fixture was made from.
    pub source: String,
    /// The encoder on [`synthetic_features`].
    pub encoder: EncoderRecord,
    /// The embedding on a few ids.
    pub embed: EmbedRecord,
    /// The decoder's prefill, then one step.
    pub decoder: DecoderRecord,
}

/// The encoder's record.
#[derive(Debug, Deserialize)]
pub struct EncoderRecord {
    /// The synthetic input's chunk count.
    pub chunks: usize,
    /// `audio_embeds [1, chunks * 13, 1024]`.
    pub audio_embeds: TensorRecord,
}

/// The embedding's record.
#[derive(Debug, Deserialize)]
pub struct EmbedRecord {
    /// The embedded ids.
    pub ids: Vec<i64>,
    /// `inputs_embeds [1, ids, 1024]`.
    pub inputs_embeds: TensorRecord,
}

/// The decoder's record.
#[derive(Debug, Deserialize)]
pub struct DecoderRecord {
    /// The prefill length.
    pub seq: usize,
    /// The prefill's logits.
    pub prefill_logits: TensorRecord,
    /// Layer 0's key cache after the prefill.
    pub prefill_key_0: TensorRecord,
    /// Layer 27's value cache after the prefill.
    pub prefill_value_27: TensorRecord,
    /// The step's logits.
    pub step_logits: TensorRecord,
    /// Layer 27's key cache after the step.
    pub step_key_27: TensorRecord,
}

/// Reads `testdata/onnx_reference.json`.
pub fn onnx_reference() -> OnnxReference {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/onnx_reference.json");
    let text = std::fs::read_to_string(path).expect("read the onnx reference fixture");
    serde_json::from_str(&text).expect("parse the onnx reference fixture")
}

/// `testdata/jfk_moon.json`: `transformers`' greedy decodes of `jfk_moon.mp3`.
#[derive(Debug, Deserialize)]
pub struct SpeechReference {
    /// The clip, under `whisper-model-validation/testdata/`.
    pub clip: String,
    /// Qwen's base model.
    pub base: Vec<WindowDecode>,
    /// Phonon-1.
    pub audio6: Vec<WindowDecode>,
    /// Phonon-1 Micro.
    pub micro: Vec<WindowDecode>,
    /// Phonon-1 Big.
    pub parity: Vec<WindowDecode>,
}

/// One window's decode.
#[derive(Debug, Deserialize)]
pub struct WindowDecode {
    /// The window's start, in seconds.
    pub start: usize,
    /// The window's end, in seconds.
    pub end: usize,
    /// The prompt, audio placeholders included.
    pub prompt_ids: Vec<i64>,
    /// The generated ids.
    pub ids: Vec<i64>,
    /// The generated text, its `language ...<asr_text>` prefix included.
    pub text: String,
}

/// Reads `testdata/jfk_moon.json`.
pub fn speech_reference() -> SpeechReference {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/jfk_moon.json");
    let text = std::fs::read_to_string(path).expect("read the speech fixture");
    serde_json::from_str(&text).expect("parse the speech fixture")
}

/// A deterministic `[1, N_MELS, frames]` stand-in for log-mels: the formula
/// `tools/gen_qwen3_asr_fixtures.py` uses.
pub fn synthetic_features(
    frames: usize,
    device: &Device,
) -> Tensor<3> {
    let data: Vec<f32> = (0..N_MELS * frames)
        .map(|k| {
            let (m, f) = (k / frames, k % frames);
            ((((m * 7 + f * 13) % 211) as f64 / 105.0 - 1.0) * 0.8) as f32
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [1, N_MELS, frames]), device)
}

/// A deterministic `[1, seq, D_TEXT]` stand-in for input embeddings.
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

/// `[1, 1, n, n]`: 0 within an attention window of `WINDOW_CHUNKS` chunks,
/// `-inf` across windows.
pub fn window_bias(
    n: usize,
    device: &Device,
) -> Tensor<4> {
    let window = FRAMES_PER_CHUNK * WINDOW_CHUNKS;
    let data: Vec<f32> = (0..n * n)
        .map(|k| {
            if (k / n) / window == (k % n) / window {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    Tensor::from_data(TensorData::new(data, [1, 1, n, n]), device)
}

/// The generated reference models, held to the onnxruntime fixture.
#[cfg(all(test, feature = "download", feature = "gpu-tests"))]
mod reference_tests;

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures are self-consistent: every record's samples cover its
    /// shape, the shapes are the model's, and every decode's prompt is the
    /// chat template's, around one placeholder per audio frame.
    #[test]
    fn test_fixture_integrity() {
        let fixture = onnx_reference();
        let decoder = &fixture.decoder;
        for record in [
            &fixture.encoder.audio_embeds,
            &fixture.embed.inputs_embeds,
            &decoder.prefill_logits,
            &decoder.prefill_key_0,
            &decoder.prefill_value_27,
            &decoder.step_logits,
            &decoder.step_key_27,
        ] {
            assert_eq!(record.samples.len(), record.expected_samples());
        }
        assert_eq!(
            fixture.encoder.audio_embeds.shape,
            [1, fixture.encoder.chunks * FRAMES_PER_CHUNK, D_TEXT]
        );
        assert_eq!(decoder.prefill_logits.shape, [1, decoder.seq, N_VOCAB]);
        assert_eq!(
            decoder.step_key_27.shape,
            [1, N_KV_HEADS, decoder.seq + 1, HEAD_DIM]
        );

        let speech = speech_reference();
        assert_eq!(speech.clip, "jfk_moon.mp3");
        for decodes in [&speech.base, &speech.audio6, &speech.micro, &speech.parity] {
            assert_eq!(decodes.len(), 2);
            for d in decodes {
                assert_eq!(
                    &d.prompt_ids[..9],
                    [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
                );
                assert_eq!(
                    &d.prompt_ids[d.prompt_ids.len() - 6..],
                    [151670, 151645, 198, 151644, 77091, 198]
                );
                assert!(
                    d.prompt_ids[9..d.prompt_ids.len() - 6]
                        .iter()
                        .all(|&t| t == 151676)
                );
                assert!(d.text.starts_with("language English<asr_text>"));
                assert!(matches!(d.ids.last(), Some(151645) | Some(151643)));
            }
        }
    }
}
