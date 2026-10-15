//! Transcriptions of `jfk_moon.mp3`, in the two 30 s windows
//! `testdata/jfk_moon.json` decodes.
//!
//! Token ids are compared exactly and their differences printed, but the gates
//! are on text: a greedy argmax over 151936 logits can flip on a backend's
//! last digits and the text barely move.

use bunsen::{
    kits::speech::qwen3_asr::Qwen3AsrFrontEndMeta,
    support::testing::{
        DeviceMemoryGuard,
        asr::{
            normalize_transcript,
            word_error_rate,
        },
        performance_device,
    },
};

use super::*;

/// The clip every test here transcribes.
const CLIP: &str = "jfk_moon.mp3";

/// The token that ends a reply's header and begins its transcript.
const ASR_TEXT: &str = "<asr_text>";

fn wer(
    hypothesis: &str,
    reference: &str,
) -> f64 {
    word_error_rate(
        &normalize_transcript(hypothesis),
        &normalize_transcript(reference),
    )
}

/// What each window of `fixture` says after its header.
fn reference_text(fixture: &[WindowDecode]) -> String {
    fixture
        .iter()
        .map(|w| {
            w.text
                .split_once(ASR_TEXT)
                .map_or(w.text.as_str(), |(_, t)| t)
                .trim()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Transcribes the fixture's windows with the pretrained `name`, reports
/// where its ids part from `fixture`'s, and returns the WERs against the
/// fixture's text and against the clip's transcript.
fn transcribe_windows(
    name: &str,
    fixture: &[WindowDecode],
) -> (f64, f64) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let bundle = load_bundle(name, &device);
    let rate = bundle.front_end.sample_rate();
    let samples = speech_clip(CLIP);

    let mut texts = Vec::new();
    for window in fixture {
        let end = (window.end * rate).min(samples.len());
        let transcript = bundle
            .transcribe(&samples[window.start * rate..end], &device)
            .expect("transcribe");
        let theirs: Vec<usize> = window.ids.iter().map(|&t| t as usize).collect();
        match transcript
            .tokens
            .iter()
            .zip(&theirs)
            .position(|(a, b)| a != b)
        {
            None if transcript.tokens.len() == theirs.len() => {
                eprintln!("{name} {}s: ids identical ({})", window.start, theirs.len())
            }
            first => eprintln!(
                "{name} {}s: ids differ at {first:?} (ours {}, theirs {})",
                window.start,
                transcript.tokens.len(),
                theirs.len()
            ),
        }
        assert_eq!(transcript.language.as_deref(), Some("English"), "{name}");
        texts.push(transcript.text);
    }
    let text = texts.join(" ");
    let against_reference = wer(&text, &reference_text(fixture));
    let against_truth = wer(&text, &speech_transcript(CLIP));
    eprintln!("{name}: {text}");
    eprintln!("WER vs transformers {against_reference:.4}, vs transcript {against_truth:.4}");
    (against_reference, against_truth)
}

/// bunsen on Qwen's base weights decodes what `transformers` decodes.
#[test]
#[serial_test::serial]
fn test_bunsen_base_transcribes_like_transformers() {
    let (reference, truth) = transcribe_windows("qwen/qwen3-asr-0.6b", &speech_reference().base);
    // Measured on wgpu: 0.0 (ids identical) and 0.0085.
    assert!(reference <= 0.0, "WER vs transformers {reference}");
    assert!(truth <= 0.02, "WER vs the transcript {truth}");
}

/// bunsen's Phonon-1 decodes what `transformers` decodes on the same dense
/// weights. Its transcript is further from the truth than the base model's:
/// it repeats a sentence, as `transformers` does on these weights.
#[test]
#[serial_test::serial]
fn test_bunsen_phonon1_transcribes_like_transformers() {
    let (reference, truth) = transcribe_windows("fermion/phonon-1", &speech_reference().audio6);
    // Measured on wgpu: 0.0 (ids identical) and 0.1026.
    assert!(reference <= 0.0, "WER vs transformers {reference}");
    assert!(truth <= 0.12, "WER vs the transcript {truth}");
}

/// bunsen's Phonon-1-Micro decodes what `transformers` decodes.
#[test]
#[serial_test::serial]
fn test_bunsen_phonon1_micro_transcribes_like_transformers() {
    let (reference, truth) =
        transcribe_windows("fermion/phonon-1-micro", &speech_reference().micro);
    // Measured on wgpu: 0.0 (ids identical) and 0.1111.
    assert!(reference <= 0.0, "WER vs transformers {reference}");
    assert!(truth <= 0.13, "WER vs the transcript {truth}");
}

/// bunsen's Phonon-1-Big decodes what `transformers` decodes.
#[test]
#[serial_test::serial]
fn test_bunsen_phonon1_big_transcribes_like_transformers() {
    let (reference, truth) = transcribe_windows("fermion/phonon-1-big", &speech_reference().parity);
    // Measured on wgpu: 0.0 (ids identical) and 0.0513.
    assert!(reference <= 0.0, "WER vs transformers {reference}");
    assert!(truth <= 0.07, "WER vs the transcript {truth}");
}
