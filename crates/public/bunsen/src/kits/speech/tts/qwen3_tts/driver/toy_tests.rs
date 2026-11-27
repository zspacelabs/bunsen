//! The engine on the toy model and codec, through the TTS driver.
//!
//! The toy's text table is 48 rows, so its text is read by [`Words`]: the
//! chat specials and roles as one id each (as the release's tokenizer reads
//! them), every other character as its own id below the specials.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::Arc,
};

use super::*;
use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenErrorKind,
        BunsenResult,
        testing::ErrorMatcher,
    },
    kits::{
        speech::tts::{
            driver::{
                TtsChunkPolicy,
                TtsDriver,
                TtsDriverConfig,
                TtsEngine,
                TtsRequest,
                TtsSamplingOverrides,
                TtsSpeech,
                TtsTakeRequest,
                TtsVoiceChoice,
            },
            qwen3_tts::{
                Qwen3TtsConstruct,
                Qwen3TtsModelKind,
                Qwen3TtsTokenizerFiles,
                blocks::{
                    Qwen3TtsCodecDecoder,
                    Qwen3TtsDecodeConfig,
                    Qwen3TtsModel,
                },
                driver::Qwen3TtsTakeConfig,
                interop::qwen_tts::*,
                prompt::Qwen3TtsTextPlacement,
                sampler::Qwen3TtsSamplerConfig,
            },
        },
        support::{
            interop::transformers::TransformersConfigJson,
            text::BunsenLanguageTag,
            tokens::{
                Detokenizer,
                Tokenizer,
            },
        },
    },
    support::testing::{
        ToyFixture,
        expect_variant,
        performance_device,
        testdata,
    },
};

/// The toy's text ids: the chat specials and roles one id each, the
/// generator's (`tools/gen_qwen3_tts_toy_fixtures.py`), any other character
/// `4 + (c mod 30)`.
#[derive(Debug)]
struct Words;

const WORDS: [(&str, i64); 5] = [
    ("<|im_start|>", 40),
    ("<|im_end|>", 41),
    ("assistant", 42),
    ("user", 44),
    ("\n", 43),
];

impl Detokenizer for Words {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        Ok(ids.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
    }
}

impl Tokenizer for Words {
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        let mut ids = Vec::new();
        let mut rest = text;
        while let Some(c) = rest.chars().next() {
            match WORDS.iter().find(|(word, _)| rest.starts_with(word)) {
                Some((word, id)) => {
                    ids.push(*id);
                    rest = &rest[word.len()..];
                }
                None => {
                    ids.push(4 + i64::from(u32::from(c) % 30));
                    rest = &rest[c.len_utf8()..];
                }
            }
        }
        Ok(ids)
    }

    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>> {
        self.encode(text)
    }

    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64> {
        WORDS.iter().find(|(w, _)| *w == content).map(|(_, id)| *id)
    }

    fn contains(
        &self,
        id: i64,
    ) -> bool {
        (0..48).contains(&id)
    }

    fn len(&self) -> usize {
        48
    }
}

/// The toy as a construct: its model and codec, its ids, greedy takes of at
/// most 5 frames, and a rate of 300 Hz (24 samples a frame, 12.5 frames a
/// second, as the release's).
fn construct(
    kind: Qwen3TtsModelKind,
    instruction: bool,
) -> Arc<Qwen3TtsConstruct> {
    let device = performance_device();
    let dir = testdata!("qwen3_tts");
    let text = std::fs::read_to_string(dir.join("toy.config.json")).expect("read");
    let model_json = TransformersConfigJson::parse(&text)
        .and_then(|c| c.section("model"))
        .expect("the toy's model config");
    let model_config = model_config(&model_json).expect("the toy");
    let value: serde_json::Value = serde_json::from_str(&text).expect("parse");
    let ids: Qwen3TtsPromptMirror =
        serde_json::from_value(value["model"].clone()).expect("the toy's ids");
    let codec_json: Qwen3TtsCodecDecoderMirror =
        serde_json::from_value(value["codec"]["decoder_config"].clone()).expect("the codec");
    let codec_config =
        Qwen3TtsCodecDecoderMirror::codec_decoder_config(&codec_json).expect("the toy codec");

    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    let weights = fixture.take_weights();
    let mut model: Qwen3TtsModel = model_config.init(&device);
    load_model_tensors(&mut model, weights.clone(), None).expect("the toy model");
    let mut codec: Qwen3TtsCodecDecoder = codec_config.init(&device);
    load_codec_decoder_tensors(&mut codec, &codec_config, weights, None).expect("the toy codec");

    let take = Qwen3TtsTakeConfig::new()
        .with_talker(
            Qwen3TtsSamplerConfig::new()
                .with_sample(false)
                .with_repetition_penalty(1.05),
        )
        .with_code_predictor(Qwen3TtsSamplerConfig::new().with_sample(false))
        .with_max_frames(5);
    let tag = |t: &str| BunsenLanguageTag::parse(t).unwrap();
    Arc::new(Qwen3TtsConstruct {
        model,
        model_config,
        codec,
        codec_config,
        decode: Qwen3TtsDecodeConfig::new(),
        prompt: Qwen3TtsPromptMirror::prompt_config(&ids).expect("the toy's prompt ids"),
        take,
        sample_rate: 300,
        tokenizer: Qwen3TtsTokenizerFiles {
            vocab: PathBuf::new(),
            merges: PathBuf::new(),
            tokenizer_config: PathBuf::new(),
        },
        kind,
        instruction,
        language_tags: BTreeMap::from([
            ("chinese".to_string(), tag("zh")),
            ("english".to_string(), tag("en")),
            ("sichuan_dialect".to_string(), tag("zh")),
        ]),
    })
}

fn engine(config: Qwen3TtsEngineConfig) -> Qwen3TtsEngine<Words> {
    Qwen3TtsEngine::new(
        construct(Qwen3TtsModelKind::CustomVoice, false),
        Words,
        config,
    )
    .expect("the toy engine")
}

fn take<'a>(
    text: &'a str,
    voice: &'a Qwen3TtsVoice,
    language: Option<&'a str>,
    instruction: Option<&'a str>,
    sampling: &'a TtsSamplingOverrides,
) -> TtsTakeRequest<'a, Qwen3TtsVoice> {
    TtsTakeRequest {
        text,
        voice,
        language,
        instruction,
        seed: None,
        max_seconds: None,
        sampling,
    }
}

/// The capabilities: the tagged languages, the stock speakers (`eric`'s
/// default language its dialect), no designed voices on a `CustomVoice`
/// model, the construct's instruction.
#[test]
fn test_capabilities() {
    let engine = engine(Qwen3TtsEngineConfig::new());
    let capabilities = engine.capabilities();
    let keys: Vec<&str> = capabilities
        .languages
        .iter()
        .map(|l| l.key.as_str())
        .collect();
    assert_eq!(keys, ["chinese", "english", "sichuan_dialect"]);
    assert!(capabilities.stock_voices && !capabilities.described_voices);
    assert!(!capabilities.instruction);
    let voices = TtsEngine::voices(&engine);
    assert_eq!(voices.len(), 2);
    assert_eq!(voices[0].name, "eric");
    assert_eq!(
        voices[0].default_language.as_deref(),
        Some("sichuan_dialect")
    );
    assert_eq!(voices[1].default_language, None);
}

/// The prompt request: the text wrapped as the assistant's turn (the role's
/// ids, the text's, the close's), the instruction as the user's, the
/// speaker's row, the language's row (`auto`: the speaker's dialect, or
/// none), the config's placement.
#[test]
fn test_prompt_request() {
    let engine = engine(Qwen3TtsEngineConfig::new());
    let ryan = engine
        .resolve_voice(&TtsVoiceChoice::Stock("Ryan".into()))
        .unwrap();
    assert_eq!(
        ryan,
        Qwen3TtsVoice::Speaker {
            name: "ryan".into(),
            row: 1045
        },
        "matched without case"
    );
    let none = TtsSamplingOverrides::default();
    let request = engine
        .prompt_request(&take("ab", &ryan, Some("english"), Some("x"), &none))
        .unwrap();
    assert_eq!(
        request.input_ids,
        [40, 42, 43, 4 + 97 % 30, 4 + 98 % 30, 41, 43, 40, 42, 43]
    );
    assert_eq!(
        request.instruct_ids,
        Some(vec![40, 44, 43, 4 + 120 % 30, 41, 43])
    );
    assert_eq!(request.speaker, Some(1045));
    assert_eq!(request.language, Some(34));
    assert_eq!(request.placement, Qwen3TtsTextPlacement::Upfront);

    let auto = engine
        .prompt_request(&take("ab", &ryan, None, None, &none))
        .unwrap();
    assert_eq!(auto.language, None, "ryan speaks no dialect");
    assert_eq!(auto.instruct_ids, None);
    let eric = engine
        .resolve_voice(&TtsVoiceChoice::Stock("eric".into()))
        .unwrap();
    assert_eq!(
        engine
            .prompt_request(&take("ab", &eric, Some("auto"), None, &none))
            .unwrap()
            .language,
        Some(46),
        "eric's dialect"
    );
    assert_eq!(
        engine
            .language_tag(None, &eric)
            .unwrap()
            .unwrap()
            .to_string(),
        "zh"
    );
    assert_eq!(engine.language_tag(None, &ryan).unwrap(), None);
    assert_eq!(
        engine
            .language_tag(Some("English"), &ryan)
            .unwrap()
            .unwrap()
            .to_string(),
        "en"
    );
}

/// A take through the driver: the toy's greedy take of at most 5 frames,
/// rendered at 24 samples a frame; seeded, it repeats; `max_seconds` bounds
/// its frames at 12.5 a second.
#[test]
fn test_synthesize() {
    let driver = qwen3_tts_driver(engine(Qwen3TtsEngineConfig::new())).unwrap();
    let request = TtsRequest::new("hello")
        .with_voice(TtsVoiceChoice::Stock("ryan".into()))
        .with_language("english")
        .with_seed(3);
    let speech = driver.synthesize(&request).unwrap();
    expect_variant!(let [take] = speech.takes.as_slice());
    assert!((1..=5).contains(&take.frames), "{take:?}");
    assert_eq!(speech.samples.len(), take.frames * 24);
    assert_eq!(speech.sample_rate, 300);
    assert!(
        speech
            .samples
            .iter()
            .all(|x| x.is_finite() && x.abs() <= 1.0)
    );
    assert_eq!(driver.synthesize(&request).unwrap(), speech, "repeats");

    let short = driver
        .synthesize(&request.clone().with_max_seconds(0.2))
        .unwrap();
    assert!(short.takes[0].frames <= 3, "{:?}", short.takes);
}

/// The stream is the synthesis: the same seeded take, frame for frame,
/// and its samples, under the release's decode and one that cuts the take
/// into chunks of 2 frames with 1 of context, the final frames decoded 1,
/// 2, … or 1 at a time.
#[test]
fn test_stream_is_the_synthesis() {
    let request = TtsRequest::new("hello")
        .with_voice(TtsVoiceChoice::Stock("ryan".into()))
        .with_language("english")
        .with_seed(3);
    for decode in [
        None,
        Some(Qwen3TtsDecodeConfig::new().with_chunk(2).with_context(1)),
    ] {
        let driver =
            qwen3_tts_driver(engine(Qwen3TtsEngineConfig::new().with_decode(decode))).unwrap();
        let whole = driver.synthesize(&request).unwrap();
        for chunks in [TtsChunkPolicy::new(), TtsChunkPolicy::new().with_max(1)] {
            let mut stream = driver.stream(&request, chunks).unwrap();
            let streamed = TtsSpeech::from_stream(&mut stream).unwrap();
            assert_eq!(streamed.takes, whole.takes, "{decode:?} {chunks:?}");
            assert_eq!(streamed.samples.len(), whole.samples.len());
            for (a, b) in streamed.samples.iter().zip(&whole.samples) {
                assert!((a - b).abs() <= 1e-4, "{decode:?} {chunks:?}: {a} vs {b}");
            }
        }
    }
}

/// Refused: an instruction to a model that takes none (unless stripped), a
/// described voice on a `CustomVoice` model, no voice without a default, a
/// language the model does not have, a min-p.
#[test]
fn test_refusals() {
    let driver = qwen3_tts_driver(engine(Qwen3TtsEngineConfig::new())).unwrap();
    let ryan = || TtsRequest::new("hi").with_voice(TtsVoiceChoice::Stock("ryan".into()));
    let kind = |request: TtsRequest| driver.synthesize(&request).unwrap_err().kind();
    assert_eq!(
        kind(ryan().with_instruction("slowly")),
        BunsenErrorKind::Illegal
    );
    assert_eq!(
        kind(TtsRequest::new("hi").with_voice(TtsVoiceChoice::Described("calm".into()))),
        BunsenErrorKind::Unsupported
    );
    assert_eq!(kind(TtsRequest::new("hi")), BunsenErrorKind::Lookup);
    assert_eq!(
        kind(ryan().with_language("klingon")),
        BunsenErrorKind::Lookup
    );
    assert_eq!(
        kind(ryan().with_sampling(TtsSamplingOverrides {
            min_p: Some(0.1),
            ..TtsSamplingOverrides::default()
        })),
        BunsenErrorKind::Unsupported
    );
    let stripping = TtsDriver::new(
        engine(Qwen3TtsEngineConfig::new()),
        TtsDriverConfig::new().with_strip_instruction(true),
    )
    .unwrap();
    let speech = stripping
        .synthesize(&ryan().with_instruction("slowly"))
        .unwrap();
    assert_eq!(speech.notes.len(), 1);

    let error = Qwen3TtsEngine::new(
        construct(Qwen3TtsModelKind::CustomVoice, false),
        Words,
        Qwen3TtsEngineConfig::new().with_default_voice(Some("nobody".into())),
    )
    .unwrap_err();
    ErrorMatcher::lookup().assert(&error);

    let error = Qwen3TtsEngine::new(
        construct(Qwen3TtsModelKind::CustomVoice, false),
        Words,
        Qwen3TtsEngineConfig::new().with_decode(Some(Qwen3TtsDecodeConfig::new().with_chunk(0))),
    )
    .unwrap_err();
    // a decode chunk of no frames
    ErrorMatcher::illegal().assert(&error);
}

/// A model that takes an instruction gets it; a `VoiceDesign` model speaks in
/// a described voice, its description the user's turn and no speaker row.
#[test]
fn test_instruction_and_design() {
    let instructed = qwen3_tts_driver(
        Qwen3TtsEngine::new(
            construct(Qwen3TtsModelKind::CustomVoice, true),
            Words,
            Qwen3TtsEngineConfig::new().with_default_voice(Some("ryan".into())),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(instructed.engine().capabilities().instruction);
    instructed
        .synthesize(&TtsRequest::new("hi").with_instruction("slowly"))
        .unwrap();

    let design = Qwen3TtsEngine::new(
        construct(Qwen3TtsModelKind::VoiceDesign, false),
        Words,
        Qwen3TtsEngineConfig::new(),
    )
    .unwrap();
    assert!(design.capabilities().described_voices);
    let voice = design
        .resolve_voice(&TtsVoiceChoice::Described("ab".into()))
        .unwrap();
    let none = TtsSamplingOverrides::default();
    let request = design
        .prompt_request(&take("hi", &voice, Some("english"), None, &none))
        .unwrap();
    assert_eq!(request.speaker, None);
    assert_eq!(
        request.instruct_ids,
        Some(vec![40, 44, 43, 4 + 97 % 30, 4 + 98 % 30, 41, 43])
    );
    let driver = qwen3_tts_driver(design).unwrap();
    driver
        .synthesize(&TtsRequest::new("hi").with_voice(TtsVoiceChoice::Described("calm".into())))
        .unwrap();
}
