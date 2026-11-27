//! The engine on the toy fixtures: the language model's half
//! ([`generate_codes`](Zonos3Engine::generate_codes)) on the toy LM, the
//! codec's half ([`render`](Zonos3Engine::render)) on the toy codec, whose
//! codebooks differ from the LM's; and the TTS driver over it.

use alloc::{
    collections::BTreeMap,
    sync::Arc,
};
use std::path::PathBuf;

use super::*;
use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenErrorKind,
        BunsenResult,
    },
    kits::{
        speech::tts::{
            driver::{
                TTS_VOICE_REFERENCE_VERSION,
                TtsChunkPolicy,
                TtsClipRecord,
                TtsCloneMode,
                TtsDriver,
                TtsDriverConfig,
                TtsEngine,
                TtsReferenceClip,
                TtsRequest,
                TtsSamplingOverrides,
                TtsSpeech,
                TtsSynthesizer,
                TtsVoiceChoice,
                TtsVoiceReference,
                TtsVoiceReferenceRecord,
            },
            zonos3::{
                Zonos3Construct,
                Zonos3SpeakerPart,
                blocks::{
                    Zonos3Codec,
                    Zonos3Lm,
                    Zonos3SpeakerEncoder,
                },
                interop::zonos::{
                    Zonos3CodecMirror,
                    Zonos3GenerationMirror,
                    Zonos3LmMirror,
                    Zonos3SpeakerEncoderMirror,
                    load_zonos3_codec_tensors,
                    load_zonos3_lm_tensors,
                    load_zonos3_speaker_tensors,
                },
                prompt::{
                    Zonos3PromptConfig,
                    Zonos3Speaker,
                },
                voices::Zonos3Voice,
            },
        },
        support::tokens::{
            Detokenizer,
            Tokenizer,
        },
    },
    support::testing::{
        ToyFixture,
        expect_variant,
        performance_device,
        tensor_to_vec_as,
        testdata,
    },
};

/// Each byte `4 + byte % 16`, within the toy's 20 text ids.
#[derive(Debug)]
struct Bytes;

impl Detokenizer for Bytes {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        Ok(ids.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
    }
}

impl Tokenizer for Bytes {
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        Ok(text.bytes().map(|b| 4 + i64::from(b) % 16).collect())
    }

    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        self.encode(text)
    }

    fn token_id(
        &self,
        _content: &str,
    ) -> Option<i64> {
        None
    }

    fn contains(
        &self,
        id: i64,
    ) -> bool {
        (0..20).contains(&id)
    }

    fn len(&self) -> usize {
        20
    }
}

fn generation() -> Zonos3GenerationMirror {
    Zonos3GenerationMirror {
        decode_context_frames: 30,
        headroom_db: 3.0,
        languages: BTreeMap::new(),
        markup_tags: Vec::new(),
        min_p: 0.0,
        output_limit_dbfs: -1.0,
        overlap_frames: 4,
        repetition_codebooks: 6,
        repetition_penalty: 2.4,
        repetition_window: 96,
        silence_frame: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        silence_prefill_frames: 3,
        silence_prefill_max_frames: 30,
        temperature: 0.75,
        top_p: 0.69,
        topk: 80,
    }
}

/// The toy LM and codec as a construct, with one stock voice, `Toy`, and
/// the release's languages `generation` gives.
fn construct_with(generation: &Zonos3GenerationMirror) -> Zonos3Construct {
    let device = performance_device();
    let dir = testdata!("zonos3");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("toy.config.json")).expect("read the toy config"),
    )
    .expect("parse the toy config");

    let lm_json: Zonos3LmMirror =
        serde_json::from_value(json["lm"].clone()).expect("the toy's lm section");
    let lm_config = lm_json.lm_config().expect("the toy's lm config");
    let mut lm: Zonos3Lm = lm_config.init(&device);
    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    load_zonos3_lm_tensors(&mut lm, &lm_config, fixture.take_weights(), None)
        .expect("load the toy lm");

    let codec_json: Zonos3CodecMirror =
        serde_json::from_value(json["codec"].clone()).expect("the toy's codec section");
    let codec_config = codec_json.codec_config().expect("the toy's codec config");
    let mut codec: Zonos3Codec = codec_config.init(&device);
    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    load_zonos3_codec_tensors(&mut codec, &codec_config, fixture.take_weights(), None)
        .expect("load the toy codec");

    let prompt = Zonos3PromptConfig::new(
        &lm_config,
        lm_json.text_vocab as u32,
        generation.silence_frame.clone(),
        900,
    );
    let voices = BTreeMap::from([(
        "Toy".to_string(),
        Zonos3Voice {
            codes: (0..4).map(|f| vec![f as u32 % 16; 10]).collect(),
            embedding: vec![0.1; 8],
            info: None,
        },
    )]);
    Zonos3Construct {
        lm,
        lm_config,
        codec,
        codec_config,
        prompt,
        engine_config: generation.engine_config(),
        language_tags: generation.language_tags().expect("the toy's languages"),
        sample_rate: 44_100,
        voices,
        conditions: vec![crate::kits::speech::tts::zonos3::prompt::Zonos3Condition {
            name: "rate".into(),
            edges: vec![1.0, 2.0],
        }],
        tokenizer: PathBuf::new(),
        speaker: None,
    }
}

/// The toy construct, under the toy release's [`generation`].
fn construct() -> Zonos3Construct {
    construct_with(&generation())
}

/// The engine, its default voice the toy's.
fn engine(config: Zonos3EngineConfig) -> Zonos3Engine<Bytes> {
    Zonos3Engine::new(
        Arc::new(construct()),
        Bytes,
        config.with_default_voice("Toy".to_string()),
    )
    .expect("the toy engine")
}

#[test]
fn test_generate_codes() {
    let engine = engine(generation().engine_config().with_max_new_frames(Some(12)));
    let request = Zonos3Request::new("hello", engine.voice_reference("Toy").unwrap()).with_seed(7);
    let codes = engine.generate_codes(&request).unwrap();
    assert!(codes.rows <= 12);
    assert!(codes.rows == 12 || codes.stop == Zonos3Stop::EndOfAudio);
    assert_eq!(codes.start, 0, "the prefill of 3 is under the overlap of 4");
    assert!(
        codes
            .frames
            .iter()
            .all(|f| f.len() == 10 && f.iter().all(|&c| c < 16)),
        "codes alone, before the end of audio: {:?}",
        codes.frames
    );
    assert_eq!(engine.generate_codes(&request).unwrap(), codes, "seeded");
    let shorter = engine
        .generate_codes(&request.clone().with_max_frames(5))
        .unwrap();
    assert!(shorter.rows <= 5, "the request's bound under the config's");
}

#[test]
fn test_generate_codes_cold() {
    // No silence prefill and no speaker: the prompt ends in a text row.
    let engine = engine(
        Zonos3EngineConfig::new()
            .with_silence_prefill_frames(0)
            .with_max_new_frames(Some(14)),
    );
    let codes = engine
        .generate_codes(
            &Zonos3Request::new("hi", Zonos3Speaker::Embedding(vec![0.5; 8])).with_seed(3),
        )
        .unwrap();
    assert!(codes.rows >= 1 && codes.rows <= 14);
    assert!(codes.frames.iter().all(|f| f.iter().all(|&c| c < 16)));
}

#[test]
fn test_refusals() {
    let new = |config: Zonos3EngineConfig| Zonos3Engine::new(Arc::new(construct()), Bytes, config);
    let error = new(Zonos3EngineConfig::new()
        .with_default_voice("Toy".to_string())
        .with_silence_prefill_frames(31))
    .unwrap_err();
    assert_eq!(error.kind(), BunsenErrorKind::Illegal);
    let error = new(Zonos3EngineConfig::new()).unwrap_err();
    assert_eq!(
        error.kind(),
        BunsenErrorKind::Lookup,
        "no stock voice Savannah"
    );

    let engine = engine(Zonos3EngineConfig::new());
    assert_eq!(
        engine.voice_reference("Nobody").unwrap_err().kind(),
        BunsenErrorKind::Lookup
    );
    // The toy's context is 256 rows.
    let long = "x".repeat(300);
    let error = engine
        .generate_codes(&Zonos3Request::new(long, Zonos3Speaker::None))
        .unwrap_err();
    assert_eq!(error.kind(), BunsenErrorKind::Illegal);
}

/// A stock voice speaks in the config's speaker mode: its reference codes by
/// default, its embedding in embedding mode; `voice_reference` is the codes
/// whatever the mode.
#[test]
fn test_speaker_mode() {
    let reference = engine(Zonos3EngineConfig::new());
    let toy = reference.stock_voice("Toy").unwrap().clone();
    assert_eq!(
        reference.resolve_voice(&TtsVoiceChoice::Default).unwrap(),
        Zonos3Speaker::Codes(toy.codes.clone())
    );
    let embedding =
        engine(Zonos3EngineConfig::new().with_speaker_mode(Zonos3SpeakerMode::Embedding));
    assert_eq!(
        embedding
            .resolve_voice(&TtsVoiceChoice::Stock("Toy".into()))
            .unwrap(),
        Zonos3Speaker::Embedding(toy.embedding)
    );
    assert_eq!(
        embedding.voice_reference("Toy").unwrap(),
        Zonos3Speaker::Codes(toy.codes)
    );
    let codes = embedding
        .generate_codes(
            &Zonos3Request::new("hi", embedding.stock_speaker("Toy").unwrap()).with_seed(5),
        )
        .unwrap();
    assert!(codes.rows >= 1);
}

/// A condition's value is bucketed by its edges, in table order; one not
/// given is the unknown bucket; one the release does not have is refused.
#[test]
fn test_conditions() {
    let engine = engine(Zonos3EngineConfig::new());
    assert_eq!(engine.condition_buckets, [None]);
    for (value, bucket) in [(0.5, 0), (1.0, 1), (1.5, 1), (2.5, 2)] {
        let engine = self::engine(
            Zonos3EngineConfig::new().with_conditions(BTreeMap::from([("rate".into(), value)])),
        );
        assert_eq!(engine.condition_buckets, [Some(bucket)], "{value}");
    }
    let error = Zonos3Engine::new(
        Arc::new(construct()),
        Bytes,
        Zonos3EngineConfig::new()
            .with_default_voice("Toy".to_string())
            .with_conditions(BTreeMap::from([("pitch".into(), 1.0)])),
    )
    .unwrap_err();
    assert_eq!(error.kind(), BunsenErrorKind::Lookup);
}

#[test]
fn test_render() {
    let engine = engine(Zonos3EngineConfig::new());
    let codes = |frames: Vec<Vec<u32>>| Zonos3Codes {
        frames,
        start: 0,
        stop: Zonos3Stop::EndOfAudio,
        rows: 0,
    };
    // The toy codec reads 3 codebooks of 16 codes; its hop is 9 samples.
    let samples = engine
        .render(&codes((0..5).map(|f| vec![f, 15 - f, 7]).collect()))
        .unwrap();
    assert_eq!(samples.len(), 5 * 9);
    assert!(samples.iter().all(|x| x.abs() < 1.0));

    assert!(engine.render(&codes(Vec::new())).unwrap().is_empty());
    for bad in [vec![vec![0, 1]], vec![vec![0, 1, 16]]] {
        assert_eq!(
            engine.render(&codes(bad)).unwrap_err().kind(),
            BunsenErrorKind::Illegal
        );
    }
}

/// Through the TTS driver: the default voice is the toy's; the take's frames
/// are what it renders; `max_seconds` bounds its rows at the codec's frame
/// rate (44.1 kHz over a hop of 9, 4900 a second); a sampling override
/// reaches the sampler (greedy is seed-free); a designed voice is refused.
#[test]
fn test_tts_driver() {
    let engine = engine(Zonos3EngineConfig::new());
    let capabilities = engine.capabilities().clone();
    assert_eq!(capabilities.default_voice.as_deref(), Some("Toy"));
    assert!(capabilities.stock_voices && !capabilities.instruction);
    assert_eq!(TtsEngine::voices(&engine).len(), 1);
    let driver = TtsDriver::new(engine, TtsDriverConfig::new()).unwrap();

    let request = TtsRequest::new("hello")
        .with_seed(7)
        .with_max_seconds(12.0 / 4900.0);
    let speech = driver.synthesize(&request).unwrap();
    expect_variant!(let [take] = speech.takes.as_slice());
    assert!(take.steps <= 12, "{take:?}");
    assert_eq!(speech.samples.len(), take.frames * 9);
    assert_eq!(speech.sample_rate, 44_100);
    assert_eq!(driver.synthesize(&request).unwrap(), speech, "seeded");

    let greedy = request.clone().with_sampling(TtsSamplingOverrides {
        temperature: Some(0.0),
        ..TtsSamplingOverrides::default()
    });
    let a = driver.synthesize(&greedy).unwrap();
    let b = driver.synthesize(&greedy.clone().with_seed(8)).unwrap();
    assert_eq!(a, b, "greedy takes ignore the seed");

    let error = driver
        .synthesize(&TtsRequest::new("hi").with_voice(TtsVoiceChoice::Described("calm".into())))
        .unwrap_err();
    assert_eq!(error.kind(), BunsenErrorKind::Unsupported);
}

/// The stream is the synthesis: the same seeded takes, row for row, and
/// their samples, a frame final `n - 1` rows behind the model; over two
/// takes and their gap, sampled and greedy, the final frames decoded by the
/// default policy and a frame at a time.
#[test]
fn test_stream_is_the_synthesis() {
    let driver = TtsDriver::new(
        engine(Zonos3EngineConfig::new()),
        TtsDriverConfig::new().with_max_take_chars(8),
    )
    .unwrap();
    let request = TtsRequest::new("hello. there.")
        .with_seed(7)
        .with_max_seconds(30.0 / 4900.0);
    let greedy = request.clone().with_sampling(TtsSamplingOverrides {
        temperature: Some(0.0),
        ..TtsSamplingOverrides::default()
    });
    for request in [request, greedy] {
        let whole = driver.synthesize(&request).unwrap();
        assert_eq!(whole.takes.len(), 2);
        for chunks in [TtsChunkPolicy::new(), TtsChunkPolicy::new().with_max(1)] {
            let mut stream = driver.stream(&request, chunks).unwrap();
            let streamed = TtsSpeech::from_stream(&mut stream).unwrap();
            assert_eq!(streamed.takes, whole.takes, "{chunks:?}");
            assert_eq!(streamed.samples.len(), whole.samples.len());
            for (a, b) in streamed.samples.iter().zip(&whole.samples) {
                assert!((a - b).abs() <= 1e-4, "{chunks:?}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn test_language_tags() {
    let mut generation = generation();
    generation.languages = BTreeMap::from([("en_us".to_string(), "en".to_string())]);
    let construct = construct_with(&generation);
    assert_eq!(construct.language_tag("en_us").unwrap().to_string(), "en");
    assert_eq!(
        construct.language_tag("fr-FR").unwrap().to_string(),
        "fr-FR"
    );
    let error = construct.language_tag("klingonese").unwrap_err();
    assert_eq!(error.kind(), BunsenErrorKind::Illegal);
}

/// The release's normalization refuses a language it has no grammar for
/// (and, without `zyphra-textnorm`, refuses to be built).
#[test]
fn test_release_normalization_checks_the_language() {
    let error = zonos3_driver(engine(Zonos3EngineConfig::new()))
        .and_then(|driver| driver.synthesize(&TtsRequest::new("hi").with_language("tlh")))
        .unwrap_err();
    assert_eq!(error.kind(), BunsenErrorKind::Unsupported);
}

#[cfg(feature = "zyphra-textnorm")]
#[test]
fn test_release_normalization() {
    let driver =
        zonos3_driver(engine(Zonos3EngineConfig::new())).expect("the toy driver, normalized");
    let stage = &driver.stages()[0];
    assert_eq!(stage.name(), "text: collapse-line-breaks > zyphra");
    assert_eq!(driver.engine().language().to_string(), "en-US");
    let (request, notes) = stage
        .apply(
            TtsRequest::new("It costs\n$5."),
            Some(driver.engine().language()),
        )
        .unwrap();
    assert_eq!(request.text, "It costs five dollars.");
    assert!(notes.is_empty());
}

/// The toy speaker encoder (`testdata/zonos3/speaker_toy.*`), as a
/// construct's part of the build `toy@sha256:00`.
fn toy_speaker() -> Zonos3SpeakerPart {
    let device = performance_device();
    let dir = testdata!("zonos3");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("speaker_toy.config.json")).expect("read the config"),
    )
    .expect("parse the config");
    let mirror: Zonos3SpeakerEncoderMirror =
        serde_json::from_value(json["speaker_encoder"].clone()).expect("the speaker section");
    let config = mirror.speaker_config().expect("the toy's config");
    let mut fixture = ToyFixture::load(dir.join("speaker_toy.safetensors"), &device);
    let mut encoder: Zonos3SpeakerEncoder = config.init(&device);
    load_zonos3_speaker_tensors(&mut encoder, &config, fixture.take_weights(), None)
        .expect("load the toy encoder");
    Zonos3SpeakerPart {
        encoder,
        config,
        build: "toy@sha256:00".into(),
    }
}

/// A reference of one stereo clip: 0.5 s of two tones at 22.05 kHz.
fn stereo_reference() -> TtsVoiceReference {
    let rate = 22_050;
    let n = rate / 2;
    let tone = |hz: f32| -> Vec<f32> {
        (0..n)
            .map(|i| 0.3 * (core::f32::consts::TAU * hz * i as f32 / rate as f32).sin())
            .collect()
    };
    let record = TtsClipRecord {
        file: "a.wav".into(),
        sha256: "00".into(),
        sample_rate: rate as u32,
        channels: 2,
        samples: n,
        edits: Vec::new(),
    };
    TtsVoiceReference {
        record: TtsVoiceReferenceRecord {
            version: TTS_VOICE_REFERENCE_VERSION,
            name: None,
            transcript: None,
            language: None,
            gender: None,
            style: None,
            created_unix: None,
            clips: vec![record.clone()],
        },
        clips: vec![TtsReferenceClip {
            record,
            channels: vec![tone(220.0), tone(330.0)],
        }],
        digest: "feed".into(),
    }
}

/// With a speaker encoder, the engine clones in `embedding` mode: a reference
/// to a voice of its build, whose embedding is the encoder's of the clip
/// mixed down; the voice resolves to that embedding. A voice of another
/// build or mode, or with no embedding, is refused.
#[test]
fn test_build_voice() {
    let mut construct = construct();
    construct.speaker = Some(toy_speaker());
    let engine = Zonos3Engine::new(
        Arc::new(construct),
        Bytes,
        Zonos3EngineConfig::new().with_default_voice("Toy".to_string()),
    )
    .unwrap();
    assert_eq!(
        engine.capabilities().clone_modes,
        vec![TtsCloneMode::clip("embedding")]
    );
    let reference = stereo_reference();
    let voice = engine.build_voice(&reference, None).unwrap();
    assert_eq!(voice.identity.family, "zonos3");
    assert_eq!(voice.identity.encoder, "toy@sha256:00");
    assert_eq!(voice.identity.mode, "embedding");
    assert_eq!(voice.identity.reference, "feed");
    let embedding = voice.f32_vector("embedding", 5).unwrap().to_vec();

    let part = engine.speaker_part().unwrap();
    let mono = reference.clips[0].mono();
    let expected = tensor_to_vec_as::<f32, _>(&part.encoder.embed(&mono, 22_050).unwrap());
    assert_eq!(
        embedding, expected,
        "the clip mixed down, then the encoder's"
    );
    assert_eq!(
        engine
            .resolve_voice(&TtsVoiceChoice::Print(voice.clone()))
            .unwrap(),
        Zonos3Speaker::Embedding(embedding)
    );

    let mut other = voice.clone();
    other.identity.encoder = "other@sha256:11".into();
    let mut mode = voice.clone();
    mode.identity.mode = "reference".into();
    for bad in [other, mode] {
        assert_eq!(
            engine
                .resolve_voice(&TtsVoiceChoice::Print(bad))
                .unwrap_err()
                .kind(),
            BunsenErrorKind::Illegal
        );
    }
    let mut empty = voice;
    empty.tensors.clear();
    assert_eq!(
        engine
            .resolve_voice(&TtsVoiceChoice::Print(empty))
            .unwrap_err()
            .kind(),
        BunsenErrorKind::InvalidResource
    );
    assert_eq!(
        engine
            .build_voice(&reference, Some("icl"))
            .unwrap_err()
            .kind(),
        BunsenErrorKind::Lookup
    );
}

/// Without a speaker encoder, the engine clones none and speaks in no
/// voice file.
#[test]
fn test_build_voice_unsupported() {
    let engine = engine(Zonos3EngineConfig::new());
    assert!(engine.capabilities().clone_modes.is_empty());
    assert_eq!(
        engine
            .build_voice(&stereo_reference(), None)
            .unwrap_err()
            .kind(),
        BunsenErrorKind::Unsupported
    );
    let driver = TtsDriver::new(engine, TtsDriverConfig::new()).unwrap();
    assert_eq!(
        TtsSynthesizer::build_voice(&driver, &stereo_reference(), None)
            .unwrap_err()
            .kind(),
        BunsenErrorKind::Unsupported
    );
}
