//! The Whisper stream driver on `openai/tiny`, against the one-shot decode.

use bunsen::{
    burner::module::DTypeMapper,
    kits::speech::{
        asr_driver::{
            LevelPolicy,
            StreamClock,
        },
        whisper::{
            GreedyDecodeConfig,
            Whisper,
            WhisperMeta,
            driver::{
                WhisperStreamDriverConfig,
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
use burn::module::Module;

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
    let bundle = default_whisper_factory()
        .expect("the whisper factory")
        .load_bundle("openai/tiny", &crate::weights_cache(), device)
        .expect("load openai/tiny");
    assert_eq!(
        bundle.model.n_mels(),
        N_MELS,
        "tiny shares base's front end"
    );
    assert_eq!(bundle.model.vocab_size(), N_VOCAB, "and its vocabulary");
    bundle
        .model
        .clone()
        .map(&mut DTypeMapper::new(burn::tensor::DType::F32))
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
    let driver = WhisperStreamDriverConfig::new()
        .with_language(Some("en".to_string()))
        .with_condition_on_previous_text(false)
        .init_with_layout(model.clone(), table.policy.clone(), &device)
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
