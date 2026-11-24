//! bunsen's driver on the checkpoint: takes of a stock voice, end to end.
//!
//! A sampled take matches the release's in distribution, never code for
//! code (the draws are not torch's), so this holds takes to what any take
//! must be: it ends on the end of audio, within a plausible length, its
//! samples bounded (`speaks`, which reports the decode speed); and what it
//! says is understood: Whisper (`openai/base`) transcribes takes of the
//! Harvard sentences within a word error rate (`round_trip`).

use std::time::Instant;

use bunsen::{
    data::formats::LoadPolicy,
    kits::{
        speech::{
            asr::{
                driver::{
                    LevelPolicy,
                    StreamClock,
                    TranscriptEvent,
                },
                whisper::{
                    driver::WhisperStreamDriverConfig,
                    pretrained::default_whisper_factory,
                },
            },
            support::testing::tts_synthesizer_round_trip,
            tts::{
                driver::{
                    TtsChunkPolicy,
                    TtsEvent,
                    TtsRequest,
                    TtsSpeech,
                    TtsStop,
                    TtsSynthesizer,
                    TtsVoiceChoice,
                },
                zonos3::driver::{
                    Zonos3Driver,
                    Zonos3Engine,
                    zonos3_driver,
                },
            },
        },
        support::tokens::BunsenTokenizer,
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;

/// The stock voice the take speaks in.
pub const VOICE: &str = "Savannah";

/// The take's text, normalized.
pub const TEXT: &str = "Hello, this is a short test of the speech synthesizer.";

/// The round trip's texts: Harvard sentences (IEEE 1969, list 1), no digits,
/// so normalization leaves them as they are.
pub const HARVARD: [&str; 5] = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
];

/// The pooled word error rate the round trip holds to. Measured on wgpu
/// f32: 0.000, every sentence heard word for word.
pub const ROUND_TRIP_WER: f64 = 0.10;

/// The most a streamed sample may stray from the synthesis'.
pub const STREAM_ABS: f32 = 1e-4;

/// How long a stream took: to its first audio, and in all, in seconds.
#[derive(Clone, Copy, Debug)]
pub struct StreamTiming {
    /// From the request to its first audio.
    pub first_audio: f64,

    /// From the request to its last event.
    pub total: f64,
}

/// `request` through `driver`'s stream (the default chunk policy), gathered
/// and timed.
fn timed_stream(
    driver: &dyn TtsSynthesizer,
    request: &TtsRequest,
) -> (TtsSpeech, StreamTiming) {
    let started = std::time::Instant::now();
    let mut stream = driver.stream(request, TtsChunkPolicy::new()).unwrap();
    let mut speech = TtsSpeech {
        samples: Vec::new(),
        sample_rate: stream.sample_rate(),
        takes: Vec::new(),
        notes: Vec::new(),
    };
    let mut first_audio = None;
    while let Some(event) = stream.next_event().unwrap() {
        match event {
            TtsEvent::Audio(samples) => {
                first_audio.get_or_insert_with(|| started.elapsed().as_secs_f64());
                speech.samples.extend(samples);
            }
            TtsEvent::TakeEnd { info, .. } => speech.takes.push(info),
            TtsEvent::Notes(notes) => speech.notes.extend(notes),
            TtsEvent::TakeStart { .. } => {}
        }
    }
    let total = started.elapsed().as_secs_f64();
    (
        speech,
        StreamTiming {
            first_audio: first_audio.unwrap_or(total),
            total,
        },
    )
}

/// ZONOS3-Mini's driver on `device`, the release's settings and
/// normalization.
fn driver(device: &Device) -> Zonos3Driver<BunsenTokenizer> {
    let construct = load_construct(device);
    let config = construct.engine_config.clone();
    driver_with(construct, config)
}

/// ZONOS3-Mini's driver over `construct` under `config`, the release's
/// normalization.
pub(crate) fn driver_with(
    construct: Arc<bunsen::kits::speech::tts::zonos3::Zonos3Construct>,
    config: bunsen::kits::speech::tts::zonos3::driver::Zonos3EngineConfig,
) -> Zonos3Driver<BunsenTokenizer> {
    let tokenizer = BunsenTokenizer::from_file(&construct.tokenizer).unwrap();
    zonos3_driver(Zonos3Engine::new(construct, tokenizer, config).unwrap()).unwrap()
}

/// Whisper (`openai/base`, English, f32) on `device`, as a round trip's
/// transcriber: 16 kHz mono samples to their committed text.
pub(crate) fn english_whisper(
    device: &Device
) -> impl FnMut(&[f32]) -> bunsen::errors::BunsenResult<String> {
    let whisper = default_whisper_factory()
        .unwrap()
        .load_construct(
            "openai/base",
            &weights_cache(),
            device,
            LoadPolicy::min_precision(burn::tensor::FloatDType::F32),
        )
        .unwrap();
    let asr = WhisperStreamDriverConfig::new()
        .with_language(Some("en".to_string()))
        .init_from_construct(whisper, device)
        .unwrap();
    move |samples| {
        let mut ctx = asr.new_context(StreamClock::uniform(16_000), LevelPolicy::Running)?;
        let mut events = ctx.write_read(samples)?;
        events.extend(ctx.end_read()?);
        Ok(events
            .into_iter()
            .filter_map(|e| match e {
                TranscriptEvent::Committed(s) => s.text,
                TranscriptEvent::Draft(_) => None,
            })
            .collect::<Vec<_>>()
            .join(" "))
    }
}

/// Each case of `report`: its length, its word error rate, the text and
/// what was heard.
pub(crate) fn print_round_trip(report: &bunsen::kits::speech::support::testing::RoundTripReport) {
    for case in &report.cases {
        eprintln!(
            "{:.2} s, WER {:.3}: {:?} heard as {:?}",
            case.seconds,
            case.word_error_rate(),
            case.text,
            case.transcript
        );
    }
}

/// `text` in [`VOICE`]'s voice, seeded.
fn request(
    text: &str,
    seed: u64,
) -> TtsRequest {
    TtsRequest::new(text)
        .with_voice(TtsVoiceChoice::Stock(VOICE.to_string()))
        .with_seed(seed)
}

mod zonos3_mini {
    use super::*;

    /// A take of [`TEXT`] in [`VOICE`]'s voice ends on the end of audio,
    /// between one and ten seconds long, every sample in `(-1, 1)`.
    #[test]
    #[serial_test::serial]
    fn speaks() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let driver = driver(&device);

        let started = Instant::now();
        let speech = driver.synthesize(&request(TEXT, 1)).unwrap();
        let elapsed = started.elapsed().as_secs_f64();

        let seconds = speech.seconds();
        let [take] = speech.takes.as_slice() else {
            panic!("one take: {:?}", speech.takes);
        };
        eprintln!(
            "{VOICE}: {} rows, {} frames, {seconds:.2} s of audio in {elapsed:.2} s \
             ({:.1} rows/s), stop {:?}",
            take.steps,
            take.frames,
            take.steps as f64 / elapsed,
            take.stop,
        );
        assert_eq!(take.stop, TtsStop::End);
        assert!((1.0..10.0).contains(&seconds), "{seconds} s");
        assert!(speech.samples.iter().all(|x| x.abs() < 1.0));
    }

    /// Takes of [`HARVARD`] in [`VOICE`]'s voice, transcribed by Whisper
    /// (`openai/base`, English), are within [`ROUND_TRIP_WER`], pooled.
    /// A sampled take as a stream is the synthesis: the same take, row for
    /// row, and its samples within [`STREAM_ABS`], its final frames decoded
    /// 1, 2, 4, 8, then 15 at a time. Its first audio's latency and its
    /// real-time factor are printed, on its first run (kernel shapes the
    /// process has not run yet are compiled and tuned) and repeated.
    #[test]
    #[serial_test::serial]
    fn stream() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let driver = driver(&device);
        let request = request(TEXT, 1);
        let (_, first) = timed_stream(&driver, &request);
        let started = std::time::Instant::now();
        let whole = driver.synthesize(&request).unwrap();
        let synthesized = started.elapsed().as_secs_f64();
        let (streamed, repeated) = timed_stream(&driver, &request);
        eprintln!(
            "whole: {synthesized:.2} s for {:.2} s of audio (real-time factor {:.2})",
            whole.seconds(),
            synthesized / whole.seconds()
        );
        for (when, timing) in [("first", first), ("repeated", repeated)] {
            eprintln!(
                "{when}: first audio at {:.2} s, {:.2} s for {:.2} s of audio (real-time factor \
                 {:.2})",
                timing.first_audio,
                timing.total,
                whole.seconds(),
                timing.total / whole.seconds()
            );
        }
        assert_eq!(streamed.takes, whole.takes);
        assert_eq!(streamed.samples.len(), whole.samples.len());
        let gap = streamed
            .samples
            .iter()
            .zip(&whole.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{:?}, {:.2} s, off by at most {gap:.2e}",
            whole.takes,
            whole.seconds()
        );
        assert!(gap <= STREAM_ABS, "{gap}");
    }

    #[test]
    #[serial_test::serial]
    fn round_trip() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let tts = driver(&device);
        let report = tts_synthesizer_round_trip(
            &HARVARD,
            &tts as &dyn TtsSynthesizer,
            &request("", 1),
            16_000,
            english_whisper(&device),
        )
        .unwrap();
        print_round_trip(&report);
        let wer = report.word_error_rate();
        eprintln!("pooled WER {wer:.4}, worst {:.3}", report.worst());
        assert!(
            wer <= ROUND_TRIP_WER,
            "pooled WER {wer} over {ROUND_TRIP_WER}"
        );
    }
}
