//! The TTS driver over Qwen3-TTS-12Hz-0.6B-CustomVoice, judged by speech
//! recognition: its speech, transcribed by Whisper, says the text.

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
                    TtsRequest,
                    TtsStop,
                    TtsSynthesizer,
                    TtsVoiceChoice,
                },
                qwen3_tts::driver::{
                    Qwen3TtsDriver,
                    Qwen3TtsEngine,
                    Qwen3TtsEngineConfig,
                    qwen3_tts_driver,
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

/// The voice the takes are in.
pub const VOICE: &str = "ryan";

/// Its language.
pub const LANGUAGE: &str = "english";

/// Harvard sentences (IEEE 1969), the round trip's texts.
pub const HARVARD: [&str; 5] = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
];

/// The pooled word error rate the round trip holds to. Measured on wgpu
/// f32: 0.049, one take of five opening with a laugh and mishearing a word.
pub const ROUND_TRIP_WER: f64 = 0.10;

/// The driver on `device`, at the release's settings.
fn driver(device: &Device) -> Qwen3TtsDriver<BunsenTokenizer> {
    let construct = load_construct(device);
    let tokenizer = construct.tokenizer.load().expect("the release's tokenizer");
    let engine =
        Qwen3TtsEngine::new(construct, tokenizer, Qwen3TtsEngineConfig::new()).expect("the engine");
    qwen3_tts_driver(engine).expect("the driver")
}

/// `text` in [`VOICE`]'s voice and [`LANGUAGE`], seeded.
fn request(
    text: &str,
    seed: u64,
) -> TtsRequest {
    TtsRequest::new(text)
        .with_voice(TtsVoiceChoice::Stock(VOICE.to_string()))
        .with_language(LANGUAGE)
        .with_seed(seed)
}

mod qwen3_tts_12hz_0_6b_customvoice {
    use super::*;

    /// A sampled take of a sentence ends on EOS, between one and ten seconds
    /// long, every sample in `[-1, 1]`.
    #[test]
    #[serial_test::serial]
    fn speaks() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let driver = driver(&device);
        let speech = driver.synthesize(&request(HARVARD[0], 1)).unwrap();
        let [take] = speech.takes.as_slice() else {
            panic!("one take: {:?}", speech.takes);
        };
        eprintln!("{take:?}, {:.2} s", speech.seconds());
        assert_eq!(take.stop, TtsStop::End);
        assert!((1.0..10.0).contains(&speech.seconds()));
        assert!(speech.samples.iter().all(|x| x.abs() <= 1.0));
    }

    /// Takes of [`HARVARD`], transcribed by Whisper (`openai/base`,
    /// English), are within [`ROUND_TRIP_WER`], pooled.
    #[test]
    #[serial_test::serial]
    fn round_trip() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let tts = driver(&device);
        let whisper = default_whisper_factory()
            .unwrap()
            .load_construct(
                "openai/base",
                &weights_cache(),
                &device,
                LoadPolicy::min_precision(burn::tensor::FloatDType::F32),
            )
            .unwrap();
        let asr = WhisperStreamDriverConfig::new()
            .with_language(Some("en".to_string()))
            .init_from_construct(whisper, &device)
            .unwrap();
        let report = tts_synthesizer_round_trip(
            &HARVARD,
            &tts as &dyn TtsSynthesizer,
            &request("", 1),
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
