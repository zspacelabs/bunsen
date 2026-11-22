//! A voice cloned from a clip, end to end: a reference of a speaker the release
//! does not ship (JFK's 1962 Rice University address, `bunsen-test-audio`),
//! built into a voice file, spoken in.
//!
//! A cloned take has no reference to equal, so this holds it to what a
//! clone must do: what it says is understood (Whisper's word error rate over
//! the Harvard sentences, beside a stock voice's in the same embedding mode),
//! and it sounds like the clip rather than like a stock voice (the speaker
//! encoder's own cosines). The voice file reads back as it was written.

use bunsen::{
    kits::speech::{
        support::testing::tts_synthesizer_round_trip,
        tts::{
            driver::{
                TTS_VOICE_REFERENCE_VERSION,
                TtsClipRecord,
                TtsReferenceClip,
                TtsRequest,
                TtsSynthesizer,
                TtsVoiceChoice,
                TtsVoicePrint,
                TtsVoiceReference,
                TtsVoiceReferenceRecord,
            },
            zonos3::driver::Zonos3SpeakerMode,
        },
    },
    support::{
        audio::load_audio,
        testing::{
            DeviceMemoryGuard,
            cosine_similarity,
            performance_device,
        },
    },
};

use super::*;
use crate::driver::{
    HARVARD,
    driver_with,
    english_whisper,
    print_round_trip,
};

// Measured on wgpu f32 (2026-11-21), seeds 1 and 2: the clone's pooled WER
// 0.073 (worst 0.125; Savannah's, in embedding mode, 0.049); the clone's
// speech at cosine 0.741 to the clone and 0.027 to Savannah, Savannah's at
// 0.839 to Savannah and -0.077 to the clone. The clone speaks the sentences
// slower (4.0-7.8 s against 2.6-3.3 s), as the clip's speaker does.

/// The pooled word error rate the cloned voice holds to over [`HARVARD`].
pub const CLONE_WER: f64 = 0.15;

/// The least cosine between the encoder's embedding of the clone's speech
/// and the clone's own.
pub const CLONE_LIKENESS: f64 = 0.5;

/// The stock voice the clone is compared with, in embedding mode.
pub const STOCK: &str = "Savannah";

/// `clip` as a one-clip reference, its digest a placeholder (no reference
/// directory is written).
fn reference(clip: &std::path::Path) -> TtsVoiceReference {
    let audio = load_audio(clip).expect("the clip");
    let record = TtsClipRecord {
        file: clip.file_name().unwrap().to_string_lossy().into_owned(),
        sha256: String::new(),
        sample_rate: audio.sample_rate,
        channels: audio.channels.len(),
        samples: audio.channels[0].len(),
        edits: Vec::new(),
    };
    TtsVoiceReference {
        record: TtsVoiceReferenceRecord {
            version: TTS_VOICE_REFERENCE_VERSION,
            name: Some("JFK".into()),
            transcript: None,
            language: Some("en".into()),
            gender: None,
            style: None,
            created_unix: None,
            clips: vec![record.clone()],
        },
        clips: vec![TtsReferenceClip {
            record,
            channels: audio.channels,
        }],
        digest: "jfk_moon".into(),
    }
}

mod zonos3_mini {
    use super::*;

    /// The clip built into a voice; the voice's file round-trips; takes of
    /// [`HARVARD`] in it are understood within [`CLONE_WER`] (beside
    /// [`STOCK`]'s, in embedding mode); and the encoder hears the clone's
    /// speech nearer the clip's voice than [`STOCK`]'s, and [`STOCK`]'s
    /// speech nearer its own.
    #[test]
    #[serial_test::serial]
    fn clone_round_trip() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let construct = load_construct(&device);
        let config = construct
            .engine_config
            .clone()
            .with_speaker_mode(Zonos3SpeakerMode::Embedding);
        let tts = driver_with(construct.clone(), config);

        let reference = reference(&bunsen_test_audio::JFK_MOON.path());
        let voice = tts.build_voice(&reference, None).expect("the clone");
        assert_eq!(voice.identity.mode, "embedding");
        let bytes = voice.to_bytes().unwrap();
        assert_eq!(TtsVoicePrint::from_bytes(&bytes).unwrap(), voice);
        let clone = voice.f32_vector("embedding", 192).unwrap().to_vec();
        let stock = construct.voices[STOCK].embedding.clone();
        eprintln!(
            "the clone's embedding: cosine {:.4} to {STOCK}'s",
            cosine_similarity(&clone, &stock)
        );

        let mut transcribe = english_whisper(&device);
        let encoder = &construct.speaker.as_ref().unwrap().encoder;
        let rate = construct.sample_rate as usize;
        let mut heard = |name: &str, choice: TtsVoiceChoice| {
            let report = tts_synthesizer_round_trip(
                &HARVARD,
                &tts as &dyn TtsSynthesizer,
                &TtsRequest::new("").with_voice(choice.clone()).with_seed(1),
                16_000,
                &mut transcribe,
            )
            .unwrap();
            print_round_trip(&report);
            let speech = tts
                .synthesize(
                    &TtsRequest::new(HARVARD.join(" "))
                        .with_voice(choice)
                        .with_seed(2),
                )
                .unwrap();
            let embedding: Vec<f32> = encoder
                .embed(&speech.samples, rate)
                .unwrap()
                .into_data()
                .try_into_vec::<f32>()
                .unwrap();
            let (to_clone, to_stock) = (
                cosine_similarity(&embedding, &clone),
                cosine_similarity(&embedding, &stock),
            );
            let wer = report.word_error_rate();
            eprintln!(
                "{name}: pooled WER {wer:.4}, worst {:.3}; its speech's embedding: cosine \
                 {to_clone:.4} to the clone, {to_stock:.4} to {STOCK}",
                report.worst()
            );
            (wer, to_clone, to_stock)
        };
        let (clone_wer, clone_to_clone, clone_to_stock) =
            heard("the clone", TtsVoiceChoice::Print(voice));
        let (_, stock_to_clone, stock_to_stock) = heard(STOCK, TtsVoiceChoice::Stock(STOCK.into()));

        assert!(clone_wer <= CLONE_WER, "the clone's WER {clone_wer}");
        assert!(
            clone_to_clone >= CLONE_LIKENESS,
            "the clone's speech at {clone_to_clone} to the clone"
        );
        assert!(
            clone_to_clone > clone_to_stock,
            "the clone's speech sounds like {STOCK}: {clone_to_clone} vs {clone_to_stock}"
        );
        assert!(
            stock_to_stock > stock_to_clone,
            "{STOCK}'s speech sounds like the clone: {stock_to_stock} vs {stock_to_clone}"
        );
    }
}
