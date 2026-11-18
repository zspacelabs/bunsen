//! The driver on the toy fixtures: the language model's half
//! ([`generate_codes`](Zonos3Driver::generate_codes)) on the toy LM, the
//! codec's half ([`render`](Zonos3Driver::render)) on the toy codec, whose
//! codebooks differ from the LM's.

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
        speech::tts::zonos3::{
            Zonos3Construct,
            Zonos3GenerationJson,
            codec::{
                Zonos3Codec,
                Zonos3CodecConfig,
                Zonos3CodecJson,
                weights::load_zonos3_codec_tensors,
            },
            lm::{
                Zonos3Lm,
                Zonos3LmConfig,
                Zonos3LmJson,
                weights::load_zonos3_lm_tensors,
            },
            prompt::{
                Zonos3PromptConfig,
                Zonos3Speaker,
            },
            voices::Zonos3Voice,
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

fn generation() -> Zonos3GenerationJson {
    Zonos3GenerationJson {
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

/// The toy LM and codec as a construct, with one stock voice, `Toy`.
fn construct() -> Zonos3Construct {
    let device = performance_device();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/zonos3");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("toy.config.json")).expect("read the toy config"),
    )
    .expect("parse the toy config");

    let lm_json: Zonos3LmJson =
        serde_json::from_value(json["lm"].clone()).expect("the toy's lm section");
    let lm_config = Zonos3LmConfig::from_json(&lm_json).expect("the toy's lm config");
    let mut lm: Zonos3Lm = lm_config.init(&device);
    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    load_zonos3_lm_tensors(&mut lm, &lm_config, fixture.take_weights(), None)
        .expect("load the toy lm");

    let codec_json: Zonos3CodecJson =
        serde_json::from_value(json["codec"].clone()).expect("the toy's codec section");
    let codec_config = Zonos3CodecConfig::from_json(&codec_json).expect("the toy's codec config");
    let mut codec: Zonos3Codec = codec_config.init(&device);
    let mut fixture = ToyFixture::load(dir.join("toy.safetensors"), &device);
    load_zonos3_codec_tensors(&mut codec, &codec_config, fixture.take_weights(), None)
        .expect("load the toy codec");

    let generation = generation();
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
        generation,
        sample_rate: 44_100,
        voices,
        tokenizer: PathBuf::new(),
    }
}

/// The driver, its text as typed (the toy's tokenizer is no release's).
fn driver(config: Zonos3DriverConfig) -> Zonos3Driver<Bytes> {
    Zonos3Driver::new(
        Arc::new(construct()),
        Bytes,
        config.with_normalization(Zonos3TextNormalization::None),
    )
    .expect("the toy driver")
}

#[test]
fn test_generate_codes() {
    let driver =
        driver(Zonos3DriverConfig::from_generation(&generation()).with_max_new_frames(Some(12)));
    let request = Zonos3Request::new("hello", driver.voice_reference("Toy").unwrap()).with_seed(7);
    let codes = driver.generate_codes(&request).unwrap();
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
    assert_eq!(driver.generate_codes(&request).unwrap(), codes, "seeded");
}

#[test]
fn test_generate_codes_cold() {
    // No silence prefill and no speaker: the prompt ends in a text row.
    let driver = driver(
        Zonos3DriverConfig::new()
            .with_silence_prefill_frames(0)
            .with_max_new_frames(Some(14)),
    );
    let codes = driver
        .generate_codes(
            &Zonos3Request::new("hi", Zonos3Speaker::Embedding(vec![0.5; 8])).with_seed(3),
        )
        .unwrap();
    assert!(codes.rows >= 1 && codes.rows <= 14);
    assert!(codes.frames.iter().all(|f| f.iter().all(|&c| c < 16)));
}

#[test]
fn test_refusals() {
    let error = Zonos3Driver::new(
        Arc::new(construct()),
        Bytes,
        Zonos3DriverConfig::new()
            .with_normalization(Zonos3TextNormalization::None)
            .with_silence_prefill_frames(31),
    )
    .unwrap_err();
    assert_eq!(error.kind(), BunsenErrorKind::Illegal);

    let driver = driver(Zonos3DriverConfig::new());
    assert_eq!(
        driver.voice_reference("Nobody").unwrap_err().kind(),
        BunsenErrorKind::Lookup
    );
    // The toy's context is 256 rows.
    let long = "x".repeat(300);
    let error = driver
        .generate_codes(&Zonos3Request::new(long, Zonos3Speaker::None))
        .unwrap_err();
    assert_eq!(error.kind(), BunsenErrorKind::Illegal);
}

#[test]
fn test_render() {
    let driver = driver(Zonos3DriverConfig::new());
    let codes = |frames: Vec<Vec<u32>>| Zonos3Codes {
        frames,
        start: 0,
        stop: Zonos3Stop::EndOfAudio,
        rows: 0,
    };
    // The toy codec reads 3 codebooks of 16 codes; its hop is 9 samples.
    let samples = driver
        .render(&codes((0..5).map(|f| vec![f, 15 - f, 7]).collect()))
        .unwrap();
    assert_eq!(samples.len(), 5 * 9);
    assert!(samples.iter().all(|x| x.abs() < 1.0));

    assert!(driver.render(&codes(Vec::new())).unwrap().is_empty());
    for bad in [vec![vec![0, 1]], vec![vec![0, 1, 16]]] {
        assert_eq!(
            driver.render(&codes(bad)).unwrap_err().kind(),
            BunsenErrorKind::Illegal
        );
    }
}
