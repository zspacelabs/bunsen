//! Transcriptions of `jfk_moon.mp3`, in the two 30 s windows
//! `testdata/jfk_moon.json` decodes.
//!
//! Token ids are compared exactly and their differences printed, but the gates
//! are on text: a greedy argmax over 151936 logits can flip on a backend's
//! last digits and the text barely move.

use bunsen::{
    kits::speech::{
        qwen3_asr::Qwen3AsrFrontEndMeta,
        testing::wer::{
            normalize_transcript,
            word_error_rate,
        },
    },
    support::testing::{
        DeviceMemoryGuard,
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

/// Each row decodes what `transformers` decodes.
mod transcribes_like_transformers {
    use super::*;

    /// bunsen on Qwen's base weights decodes what `transformers` decodes.
    #[test]
    #[serial_test::serial]
    fn qwen3_asr_0_6b() {
        let (reference, truth) =
            transcribe_windows("qwen/qwen3-asr-0.6b", &speech_reference().base);
        // Measured on wgpu: 0.0 (ids identical) and 0.0085.
        assert!(reference <= 0.0, "WER vs transformers {reference}");
        assert!(truth <= 0.02, "WER vs the transcript {truth}");
    }

    /// bunsen's Phonon-1 decodes what `transformers` decodes on the same dense
    /// weights. Its transcript is further from the truth than the base model's:
    /// it repeats a sentence, as `transformers` does on these weights.
    #[test]
    #[serial_test::serial]
    fn phonon_1() {
        let (reference, truth) = transcribe_windows("fermion/phonon-1", &speech_reference().audio6);
        // Measured on wgpu: 0.0 (ids identical) and 0.1026.
        assert!(reference <= 0.0, "WER vs transformers {reference}");
        assert!(truth <= 0.12, "WER vs the transcript {truth}");
    }

    /// bunsen's Phonon-1-Micro decodes what `transformers` decodes.
    #[test]
    #[serial_test::serial]
    fn phonon_1_micro() {
        let (reference, truth) =
            transcribe_windows("fermion/phonon-1-micro", &speech_reference().micro);
        // Measured on wgpu: 0.0 (ids identical) and 0.1111.
        assert!(reference <= 0.0, "WER vs transformers {reference}");
        assert!(truth <= 0.13, "WER vs the transcript {truth}");
    }

    /// bunsen's Phonon-1-Big decodes what `transformers` decodes.
    #[test]
    #[serial_test::serial]
    fn phonon_1_big() {
        let (reference, truth) =
            transcribe_windows("fermion/phonon-1-big", &speech_reference().parity);
        // Measured on wgpu: 0.0 (ids identical) and 0.0513.
        assert!(reference <= 0.0, "WER vs transformers {reference}");
        assert!(truth <= 0.07, "WER vs the transcript {truth}");
    }
}

mod stream_driver_transcribes_like_transformers {
    use super::*;

    /// The stream driver, fed the clip in 0.1 s pieces under the offline
    /// policy, decodes the fixture's two windows as the bundle does: its
    /// windows are the same 30 s, and its features differ from the one-shot
    /// extractor's only at the windows' edges.
    #[test]
    #[serial_test::serial]
    fn qwen3_asr_0_6b() {
        use bunsen::kits::speech::{
            asr_driver::{
                EmissionPolicy,
                LevelPolicy,
                StreamClock,
                TranscriptEvent,
            },
            qwen3_asr::{
                Qwen3AsrEngine,
                Qwen3AsrStreamDriver,
            },
        };

        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let bundle = load_bundle("qwen/qwen3-asr-0.6b", &device);
        let rate = bundle.front_end.sample_rate();
        // Without the quiet cut, a full window is decoded whole: the fixture's
        // windows.
        let engine = Qwen3AsrEngine::new(bundle).with_quiet_cut(false);
        let driver = Qwen3AsrStreamDriver::new(engine, EmissionPolicy::offline()).unwrap();
        let mut stream = driver
            .new_context(StreamClock::uniform(rate), LevelPolicy::PerWindow)
            .unwrap();

        let samples = speech_clip(CLIP);
        let mut events = Vec::new();
        for piece in samples.chunks(rate / 10) {
            events.extend(stream.write_read(piece).unwrap());
        }
        events.extend(stream.end_read().unwrap());

        let fixture = &speech_reference().base;
        let mut texts = Vec::new();
        for event in &events {
            let TranscriptEvent::Committed(segment) = event else {
                panic!("the offline policy only commits: {event:?}");
            };
            eprintln!(
                "{:.2}-{:.2}s: {:?}",
                segment.start, segment.end, segment.text
            );
            texts.push(segment.text.clone().expect("the engine has a vocabulary"));
        }
        assert_eq!(texts.len(), fixture.len(), "one segment per window");
        assert_eq!(stream.language(), Some("English"));

        let text = texts.join(" ");
        let against_reference = wer(&text, &reference_text(fixture));
        let against_truth = wer(&text, &speech_transcript(CLIP));
        eprintln!("driver: {text}");
        eprintln!("WER vs transformers {against_reference:.4}, vs transcript {against_truth:.4}");
        // Measured on wgpu: 0.0 and 0.0085, the bundle's own.
        assert!(
            against_reference <= 0.0,
            "WER vs transformers {against_reference}"
        );
        assert!(
            against_truth <= 0.02,
            "WER vs the transcript {against_truth}"
        );
    }
}

/// The clip through bunsen's Qwen3-ASR stream driver, fed in 0.1 s pieces
/// under the offline policy: each full 30 s window is cut at its quietest
/// stretch in its last third, decoded up to there, and the stream moves on
/// from the cut.
///
/// # Returns
/// The committed texts, joined.
fn stream_quiet_cut(name: &str) -> String {
    use bunsen::kits::speech::{
        asr_driver::{
            EmissionPolicy,
            LevelPolicy,
            StreamClock,
            TranscriptEvent,
        },
        qwen3_asr::{
            Qwen3AsrEngine,
            Qwen3AsrStreamDriver,
        },
    };

    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let bundle = load_bundle(name, &device);
    let rate = bundle.front_end.sample_rate();
    let engine = Qwen3AsrEngine::new(bundle);
    let driver = Qwen3AsrStreamDriver::new(engine, EmissionPolicy::offline()).unwrap();
    let mut stream = driver
        .new_context(StreamClock::uniform(rate), LevelPolicy::PerWindow)
        .unwrap();
    let samples = speech_clip(CLIP);
    let mut events = Vec::new();
    for piece in samples.chunks(rate / 10) {
        events.extend(stream.write_read(piece).unwrap());
    }
    events.extend(stream.end_read().unwrap());

    let mut texts = Vec::new();
    for event in &events {
        let TranscriptEvent::Committed(segment) = event else {
            panic!("the offline policy only commits: {event:?}");
        };
        eprintln!(
            "{name} {:.2}-{:.2}s: {:?}",
            segment.start, segment.end, segment.text
        );
        texts.push(segment.text.clone().expect("the engine has a vocabulary"));
    }
    texts.join(" ")
}

/// Cut quiet, the windows neither lose nor repeat words at the cuts: the
/// clip says "we choose to go to the moon" three times (Phonon-1 says it a
/// fourth time decoding the first 30 s whole), and the transcript is at
/// least as close to the truth as the fixed windows'.
///
/// Measured on wgpu: WER 0.0 (base) and 0.0171 (Phonon-1), the phrase
/// three times each.
fn check_cuts_full_windows_quiet(name: &str) {
    {
        let text = stream_quiet_cut(name);
        let against_truth = wer(&text, &speech_transcript(CLIP));
        let phrase = normalize_transcript(&text)
            .join(" ")
            .matches("we choose to go to the moon")
            .count();
        eprintln!("{name}: {text}");
        eprintln!("{name}: WER vs transcript {against_truth:.4}, the phrase {phrase} times");
        assert_eq!(phrase, 3, "{name}: {text}");
        assert!(
            against_truth <= 0.03,
            "{name}: WER vs the transcript {against_truth}"
        );
    }
}

mod stream_driver_cuts_full_windows_quiet {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn qwen3_asr_0_6b() {
        check_cuts_full_windows_quiet("qwen/qwen3-asr-0.6b");
    }

    #[test]
    #[serial_test::serial]
    fn phonon_1() {
        check_cuts_full_windows_quiet("fermion/phonon-1");
    }
}
