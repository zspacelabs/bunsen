//! bunsen against `transformers` on `jfk_moon.mp3`, in the windows
//! `testdata/jfk_moon.json` decodes.
//!
//! Token ids are compared exactly, up to the first step `transformers` itself
//! calls a near-tie; the gates past it are on text: a greedy argmax over
//! 131072 logits can flip on a backend's last digits, and the text barely
//! move.

use bunsen::{
    kits::speech::{
        testing::wer::{
            normalize_transcript,
            word_error_rate,
        },
        voxtral::VoxtralFrontEndMeta,
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;

fn wer(
    hypothesis: &str,
    reference: &str,
) -> f64 {
    word_error_rate(
        &normalize_transcript(hypothesis),
        &normalize_transcript(reference),
    )
}

/// A window's samples, as the fixture cut them.
fn window_samples(
    samples: &[f32],
    window: &WindowDecode,
    rate: usize,
) -> Vec<f32> {
    samples[window.start * rate..(window.end * rate).min(samples.len())].to_vec()
}

/// bunsen's front end and audio embeddings are `transformers`', on the same
/// samples: a synthetic waveform, padded as an utterance.
#[test]
#[serial_test::serial]
fn test_features_and_audio_match_transformers() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = speech_reference().front_end;
    let bundle = load_bundle(&device);
    let samples = synthetic_wave(fixture.samples);
    let padded = bundle.layout.pad(&samples, bundle.samples_per_position());
    let features = bundle.front_end.forward(&padded, &device).unwrap();
    let error = fixture
        .input_features
        .assert_matches("features", features.clone(), 1e-3, 1e-3);
    eprintln!("features: {error:?}");
    let error = fixture.audio_embeds.assert_matches(
        "audio embeds",
        bundle.model.audio_embeds(features),
        1e-2,
        2e-3,
    );
    eprintln!("audio embeds: {error:?}");
}

/// bunsen transcribes each window as `transformers` does: the same ids up to
/// any near-tie, and text within a word or two of it, and of the clip's
/// transcript.
#[test]
#[serial_test::serial]
fn test_transcribe_matches_transformers() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = speech_reference();
    let bundle = load_bundle(&device);
    let rate = bundle.front_end.sample_rate();
    let samples = speech_clip(&fixture.clip);

    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    for window in &fixture.windows {
        let transcript = bundle
            .transcribe(&window_samples(&samples, window, rate))
            .expect("transcribe");
        assert_eq!(
            &transcript.ids[..window.prompt_ids.len()],
            &window.prompt_ids[..],
            "prompt"
        );
        let prompt = window.prompt_ids.len();
        match transcript
            .ids
            .iter()
            .zip(&window.ids)
            .position(|(a, b)| a != b)
        {
            None => assert_eq!(transcript.ids.len(), window.ids.len(), "length"),
            Some(at) => {
                let margin = window.margins[at - prompt];
                eprintln!(
                    "{}-{} s: ids part at {at} (margin {margin:.4}): ours {}, theirs {}",
                    window.start, window.end, transcript.ids[at], window.ids[at]
                );
                assert!(
                    margin < NEAR_TIE,
                    "ids part at {at}, where transformers' margin is {margin}"
                );
            }
        }
        let w = wer(&transcript.text, &window.text);
        eprintln!(
            "{}-{} s: WER vs transformers {w:.3}: {}",
            window.start, window.end, transcript.text
        );
        assert!(w <= 0.05, "WER vs transformers {w}");
        ours.push(transcript.text);
        theirs.push(window.text.trim().to_string());
    }
    let truth = speech_transcript(&fixture.clip);
    let ours_wer = wer(&ours.join(" "), &truth);
    let theirs_wer = wer(&theirs.join(" "), &truth);
    eprintln!("WER vs the transcript: bunsen {ours_wer:.3}, transformers {theirs_wer:.3}");
    assert!(ours_wer <= theirs_wer + 0.02, "{ours_wer} vs {theirs_wer}");
}
