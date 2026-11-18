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
            testing::tts_asr_round_trip,
            tts::zonos3::driver::{
                Zonos3Driver,
                Zonos3DriverConfig,
                Zonos3Request,
                Zonos3Stop,
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

/// ZONOS3-Mini's driver on `device`, the release's settings.
fn driver(device: &Device) -> Zonos3Driver<BunsenTokenizer> {
    let construct = load_construct(device);
    let tokenizer = BunsenTokenizer::from_file(&construct.tokenizer).unwrap();
    let config = Zonos3DriverConfig::from_generation(&construct.generation);
    Zonos3Driver::new(construct, tokenizer, config).unwrap()
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
        let request = Zonos3Request::new(TEXT, driver.voice_reference(VOICE).unwrap()).with_seed(1);

        let started = Instant::now();
        let codes = driver.generate_codes(&request).unwrap();
        let generating = started.elapsed();
        let started = Instant::now();
        let samples = driver.render(&codes).unwrap();
        let rendering = started.elapsed();

        let seconds = samples.len() as f64 / f64::from(driver.sample_rate());
        eprintln!(
            "{VOICE}: {} rows in {:.2} s ({:.1} rows/s), {} frames, {seconds:.2} s of audio \
             rendered in {:.2} s, stop {:?}",
            codes.rows,
            generating.as_secs_f64(),
            codes.rows as f64 / generating.as_secs_f64(),
            codes.frames.len(),
            rendering.as_secs_f64(),
            codes.stop,
        );
        assert_eq!(codes.stop, Zonos3Stop::EndOfAudio);
        assert!((1.0..10.0).contains(&seconds), "{seconds} s");
        assert!(samples.iter().all(|x| x.abs() < 1.0));
    }

    /// Takes of [`HARVARD`] in [`VOICE`]'s voice, transcribed by Whisper
    /// (`openai/base`, English), are within [`ROUND_TRIP_WER`], pooled.
    #[test]
    #[serial_test::serial]
    fn round_trip() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let tts = driver(&device);
        let speaker = tts.voice_reference(VOICE).unwrap();
        let whisper = default_whisper_factory()
            .unwrap()
            .load_construct(
                "openai/base",
                &weights_cache(),
                &device,
                LoadPolicy::at_least(burn::tensor::FloatDType::F32),
            )
            .unwrap();
        let asr = WhisperStreamDriverConfig::new()
            .with_language(Some("en".to_string()))
            .init_from_construct(whisper, &device)
            .unwrap();

        let mut seed = 0;
        let report = tts_asr_round_trip(
            &HARVARD,
            |text| {
                seed += 1;
                let request = Zonos3Request::new(text, speaker.clone()).with_seed(seed);
                let speech = tts.synthesize(&request)?;
                Ok((speech.samples, speech.sample_rate as usize))
            },
            16_000,
            |samples| {
                let mut ctx =
                    asr.new_context(StreamClock::uniform(16_000), LevelPolicy::Running)?;
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
            },
        )
        .unwrap();
        for case in &report.cases {
            eprintln!(
                "{:.2} s, WER {:.3}: {:?} heard as {:?}",
                case.seconds,
                case.word_error_rate(),
                case.text,
                case.transcript
            );
        }
        let wer = report.word_error_rate();
        eprintln!("pooled WER {wer:.4}, worst {:.3}", report.worst());
        assert!(
            wer <= ROUND_TRIP_WER,
            "pooled WER {wer} over {ROUND_TRIP_WER}"
        );
    }
}
