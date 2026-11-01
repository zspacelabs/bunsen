//! Speaker-attributed transcription of `bunsen-test-audio`'s AMI excerpt:
//! each speech recognizer's stream driver with `Nemotron-3-Diarization`
//! attached, scored by cpWER against the excerpt's transcript by speaker.
//!
//! The excerpt is a far-field meeting, with overlap; the bounds are measured,
//! and hold the pipeline where it is, not where a meeting-trained recognizer
//! would be. The plain WER over the concatenated text is printed beside each
//! cpWER, so a regression in attribution shows apart from one in recognition.

use std::sync::Arc;

use bunsen::{
    kits::speech::{
        asr_driver::{
            AsrEngine,
            AsrStreamDriver,
            EmissionPolicy,
            LevelPolicy,
            MeanActivity,
            StreamClock,
            TranscriptEvent,
        },
        parakeet::{
            ParakeetEngine,
            ParakeetStreamDriver,
            pretrained::default_parakeet_factory,
        },
        qwen3_asr::{
            Qwen3AsrEngine,
            Qwen3AsrStreamDriver,
            pretrained::default_qwen3_asr_factory,
        },
        sortformer::SortformerStreamingConfig,
        testing::wer::{
            cp_word_error_rate,
            text_error_rate,
        },
        whisper::{
            driver::WhisperStreamDriverConfig,
            pretrained::default_whisper_factory,
        },
    },
    support::{
        audio::load_audio_mono_sr,
        testing::{
            DeviceMemoryGuard,
            performance_device,
        },
    },
};

use super::*;

/// One utterance of the excerpt's transcript.
#[derive(Debug, Deserialize)]
struct Utterance {
    speaker: String,
    start_time: f64,
    words: String,
}

/// The transcript by speaker, in time order.
fn reference() -> Vec<(String, String)> {
    let ami = bunsen_test_audio::AMI_EN2002D_2132;
    let text = std::fs::read_to_string(ami.transcript_path()).expect("read the transcript");
    let mut utterances: Vec<Utterance> = serde_json::from_str(&text).expect("parse the transcript");
    utterances.sort_by(|a, b| a.start_time.total_cmp(&b.start_time));
    utterances
        .into_iter()
        .map(|u| (u.speaker, u.words))
        .collect()
}

/// The clip through `driver` in one-second pieces: each committed segment's
/// speaker and text, in order.
fn transcribe<E: AsrEngine>(
    driver: &AsrStreamDriver<E>,
    level: LevelPolicy,
    samples: &[f32],
) -> Vec<(String, String)> {
    let mut ctx = driver
        .new_context(StreamClock::uniform(16_000), level)
        .unwrap();
    let mut events = Vec::new();
    for piece in samples.chunks(16_000) {
        events.extend(ctx.write_read(piece).unwrap());
    }
    events.extend(ctx.end_read().unwrap());
    events
        .into_iter()
        .filter_map(|e| match e {
            TranscriptEvent::Committed(s) => Some(s),
            TranscriptEvent::Draft(_) => None,
        })
        .filter_map(|s| {
            let text = s.text?;
            let speaker = s
                .speaker
                .map_or("nobody".to_string(), |l| format!("speaker_{}", l.index));
            Some((speaker, text))
        })
        .collect()
}

/// cpWER, with the plain WER beside it, printed and checked against `bound`.
fn score(
    name: &str,
    hypothesis: &[(String, String)],
    bound: f64,
) {
    let truth = reference();
    let cp = cp_word_error_rate(&truth, hypothesis);
    let join = |turns: &[(String, String)]| {
        turns
            .iter()
            .map(|(_, t)| t.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let plain = text_error_rate(&join(hypothesis), &join(&truth));
    let speakers: std::collections::BTreeSet<&str> =
        hypothesis.iter().map(|(s, _)| s.as_str()).collect();
    eprintln!("{name}: cpWER {cp:.4}, WER {plain:.4}, speakers {speakers:?}");
    assert!(cp <= bound, "{name}: cpWER {cp} over {bound}");
}

fn diarizer(device: &Device) -> Arc<dyn bunsen::kits::speech::diarization::SpeakerActivityModel> {
    Arc::new(
        load_bundle(device)
            .diarizer(SortformerStreamingConfig::nemotron_low_latency())
            .unwrap(),
    )
}

fn samples() -> Vec<f32> {
    load_audio_mono_sr(bunsen_test_audio::AMI_EN2002D_2132.audio_path(), 16_000)
        .expect("read the AMI clip")
}

/// Each engine with Nemotron's speakers, on the AMI clip.
mod sortformer_speakers {
    use super::*;

    /// Parakeet (Phonon-2): every word attributed.
    #[test]
    #[serial_test::serial]
    fn parakeet() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let bundle = default_parakeet_factory()
            .unwrap()
            .load_bundle("fermion/phonon-2", &weights_cache(), &device)
            .unwrap();
        let driver =
            ParakeetStreamDriver::new(ParakeetEngine::new(bundle), EmissionPolicy::offline())
                .unwrap()
                .with_speakers(diarizer(&device), Arc::new(MeanActivity::default()))
                .unwrap();
        score(
            "parakeet",
            &transcribe(&driver, LevelPolicy::PerWindow, &samples()),
            PARAKEET_BOUND,
        );
    }

    /// Qwen3-ASR (Phonon-1): windows cut at turns.
    #[test]
    #[serial_test::serial]
    fn qwen3_asr() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let bundle = default_qwen3_asr_factory()
            .unwrap()
            .load_bundle("fermion/phonon-1", &weights_cache(), &device)
            .unwrap();
        let driver =
            Qwen3AsrStreamDriver::new(Qwen3AsrEngine::new(bundle), EmissionPolicy::offline())
                .unwrap()
                .with_speakers(diarizer(&device), Arc::new(MeanActivity::default()))
                .unwrap();
        score(
            "qwen3-asr",
            &transcribe(&driver, LevelPolicy::PerWindow, &samples()),
            QWEN3_ASR_BOUND,
        );
    }

    /// Whisper (`base`, English): windows cut at turns.
    #[test]
    #[serial_test::serial]
    fn whisper() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let bundle = default_whisper_factory()
            .unwrap()
            .load_bundle("openai/base", &weights_cache(), &device)
            .unwrap();
        let driver = WhisperStreamDriverConfig::new()
            .with_language(Some("en".to_string()))
            .init_from_bundle(bundle, &device)
            .unwrap()
            .with_speakers(diarizer(&device), Arc::new(MeanActivity::default()))
            .unwrap();
        score(
            "whisper",
            &transcribe(&driver, LevelPolicy::Running, &samples()),
            WHISPER_BOUND,
        );
    }
}

/// Parakeet's cpWER bound. Measured on wgpu: 0.277 (plain WER 0.155).
const PARAKEET_BOUND: f64 = 0.32;

/// Qwen3-ASR's cpWER bound. Measured on wgpu: 0.270 (plain WER 0.223).
const QWEN3_ASR_BOUND: f64 = 0.32;

/// Whisper's cpWER bound. Measured on wgpu: 0.365 (plain WER 0.365).
const WHISPER_BOUND: f64 = 0.42;
