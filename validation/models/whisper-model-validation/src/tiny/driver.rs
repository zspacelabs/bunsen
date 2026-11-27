//! The Whisper stream driver on `openai/tiny`, against the one-shot decode.

use bunsen::{
    burner::module::InputDTypes,
    kits::speech::asr::{
        driver::{
            LevelPolicy,
            StreamClock,
        },
        whisper::{
            GreedyDecodeConfig,
            Whisper,
            WhisperMeta,
            driver::{
                WhisperEngineConfig,
                WhisperTask,
            },
            pretrained::default_whisper_factory,
        },
    },
    support::testing::{
        DeviceMemoryGuard,
        performance_device,
    },
};

use super::*;
use crate::base::audio::{
    FIXTURES,
    SAMPLE_RATE,
    clip_mels,
    report_id_diff,
    samples,
    vocab,
};

/// `openai/tiny`, through the factory and the weights cache, in f32.
fn tiny_model(device: &Device) -> Whisper {
    let construct = default_whisper_factory()
        .expect("the whisper factory")
        .load_construct(
            "openai/tiny",
            &crate::weights_cache(),
            device,
            bunsen::data::formats::LoadPolicy::min_precision(burn::tensor::FloatDType::F32),
        )
        .expect("load openai/tiny");
    assert_eq!(
        construct.model.n_mels(),
        N_MELS,
        "tiny shares base's front end"
    );
    assert_eq!(construct.model.vocab_size(), N_VOCAB, "and its vocabulary");
    assert_eq!(
        construct.model.preferred_dtype(),
        burn::tensor::FloatDType::F32,
        "min_precision(F32) loads tiny in f32"
    );
    construct.model.clone()
}

/// Offline, with no timestamps and no previous text, the driver decodes the
/// clip's 30 s windows as the model decodes them one at a time: the same
/// ids, window for window, through the stream's own features and windowing.
#[test]
#[serial_test::serial]
fn stream_matches_one_shot() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let table = vocab();
    let model = tiny_model(&device);
    let prompt = table
        .policy
        .sot_sequence(Some("en"), Some(WhisperTask::Transcribe), false)
        .expect("a multilingual layout");
    let config = GreedyDecodeConfig::new(prompt, table.policy.ids().eot);
    let driver = WhisperEngineConfig::new()
        .with_language(Some("en".to_string()))
        .with_condition_on_previous_text(false)
        .try_init(
            std::sync::Arc::new(
                bunsen::kits::speech::asr::whisper::driver::WhisperConstruct::new(
                    model.clone(),
                    table.policy.clone(),
                ),
            ),
            &device,
        )
        .and_then(|engine| {
            bunsen::kits::speech::asr::driver::AsrStreamDriver::new(
                engine,
                bunsen::kits::speech::asr::driver::EmissionPolicy::offline(),
            )
        })
        .expect("a multilingual layout with a language");

    for fixture in FIXTURES {
        let one_shot = model.decode_chunked(clip_mels(fixture.name, &device), &config);
        let mut stream = driver
            .new_context(StreamClock::uniform(SAMPLE_RATE), LevelPolicy::Running)
            .expect("a stream at the model's rate");
        let mut emissions = stream
            .write_read(&samples(fixture.name))
            .expect("the push decodes");
        emissions.extend(stream.end_read().expect("the flush decodes"));
        let streamed: Vec<Vec<i64>> = emissions
            .iter()
            .map(|e| e.segment().tokens.clone())
            .collect();
        report_id_diff("one-shot", fixture.name, &streamed, &one_shot);
        assert_eq!(streamed, one_shot, "{}: the driver's windows", fixture.name);
    }
}
