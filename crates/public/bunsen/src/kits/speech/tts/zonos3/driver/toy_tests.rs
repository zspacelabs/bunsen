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
                TtsDriver,
                TtsDriverConfig,
                TtsEngine,
                TtsRequest,
                TtsSamplingOverrides,
                TtsVoiceChoice,
            },
            zonos3::{
                Zonos3Construct,
                codec::Zonos3Codec,
                interop::zonos::{
                    Zonos3CodecMirror,
                    Zonos3GenerationMirror,
                    Zonos3LmMirror,
                    load_zonos3_codec_tensors,
                    load_zonos3_lm_tensors,
                },
                lm::Zonos3Lm,
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
        performance_device,
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
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/zonos3");
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
    let [take] = speech.takes.as_slice() else {
        panic!("one take: {:?}", speech.takes);
    };
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
