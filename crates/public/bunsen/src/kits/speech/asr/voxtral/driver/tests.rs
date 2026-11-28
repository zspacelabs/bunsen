use alloc::sync::Arc;

use burn::tensor::Device;

use super::*;
use crate::{
    burner::module::ModuleInit,
    errors::testing::ErrorMatcher,
    kits::{
        speech::{
            asr::{
                driver::{
                    AsrEngine,
                    EmissionPolicy,
                    LevelPolicy,
                    StreamClock,
                    TranscriptEvent,
                },
                voxtral::{
                    VoxtralConstruct,
                    VoxtralRealtime,
                    VoxtralTokenLayoutConfig,
                    blocks::{
                        VoxtralAudioEncoderConfig,
                        VoxtralRealtimeConfig,
                        VoxtralTextConfig,
                    },
                },
            },
            support::testing::ToneConfig,
        },
        support::interop::mistral_common::TekkenMirror,
    },
    support::testing::{
        expect_variant,
        performance_device,
    },
};

/// An untrained toy construct: 128 mels, a 40-token vocabulary whose ids
/// past the 8 specials are single letters.
fn toy_construct(device: &Device) -> VoxtralConstruct {
    let model: VoxtralRealtime = VoxtralRealtimeConfig::new(
        VoxtralAudioEncoderConfig::new(128, 16, 2, 8, 32, 1).with_sliding_window(Some(20)),
        VoxtralTextConfig::new(40, 16, 1, 2, 1, 8, 32).with_sliding_window(Some(30)),
    )
    .init(device);
    let specials = [
        "<unk>",
        "<s>",
        "</s>",
        "[STREAMING_PAD]",
        "[STREAMING_WORD]",
    ];
    let special_tokens: Vec<String> = (0..8)
        .map(|r| {
            format!(
                r#"{{"rank": {r}, "token_str": "{}"}}"#,
                specials.get(r).copied().unwrap_or("<x>")
            )
        })
        .collect();
    // " a" .. " z", then "a" ..: base64 of each.
    let vocab: Vec<String> = (0..32u8)
        .map(|r| {
            let bytes = if r < 26 {
                vec![b' ', b'a' + r]
            } else {
                vec![b'a' + r - 26]
            };
            let b64 = base64(&bytes);
            format!(r#"{{"rank": {r}, "token_bytes": "{b64}"}}"#)
        })
        .collect();
    let json = format!(
        r#"{{"config": {{"default_vocab_size": 40, "default_num_special_tokens": 8}},
             "vocab": [{}], "special_tokens": [{}]}}"#,
        vocab.join(","),
        special_tokens.join(",")
    );
    let layout = VoxtralTokenLayoutConfig::new()
        .with_bos(1)
        .with_eos(2)
        .with_streaming_pad(3)
        .with_streaming_word(4)
        .with_left_pad_positions(4)
        .with_delay_ms(160)
        .with_right_buffer_positions(2)
        .init();
    VoxtralConstruct::new(
        model,
        TekkenMirror::from_json(&json)
            .and_then(|m| m.tekken_vocabulary())
            .unwrap(),
        layout,
    )
    .unwrap()
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().fold(0u32, |n, &b| n << 8 | u32::from(b)) << (8 * (3 - chunk.len()));
        for k in 0..=chunk.len() {
            out.push(ALPHABET[(n >> (18 - 6 * k) & 63) as usize] as char);
        }
        out.extend(core::iter::repeat_n('=', 3 - chunk.len()));
    }
    out
}

/// Streamed in pieces, a clip longer than a window commits segments in
/// order, over the clip, each spelling its tokens.
#[test]
fn test_streams_through_the_core() {
    let device = performance_device();
    let engine = VoxtralEngine::new(Arc::new(toy_construct(&device)))
        .and_then(|e| e.with_window_seconds(1.0))
        .unwrap();
    assert_eq!(engine.window_frames(), 96);
    assert_eq!(engine.grid_frames(), 8);
    let driver = VoxtralStreamDriver::new(engine, EmissionPolicy::offline()).unwrap();
    let mut stream = driver
        .new_context(StreamClock::uniform(16_000), LevelPolicy::PerWindow)
        .unwrap();
    let mut events = Vec::new();
    for piece in ToneConfig::Samples(40_000).build().chunks(1_600) {
        events.extend(stream.write_read(piece).unwrap());
    }
    events.extend(stream.end_read().unwrap());
    assert!(stream.is_finished());

    let mut end = 0.0;
    for event in &events {
        expect_variant!(let TranscriptEvent::Committed(segment) = event);
        assert!(segment.start >= end && segment.end >= segment.start);
        assert!(segment.end <= 2.5 + 1e-9);
        assert_eq!(segment.text, driver.engine().text(&segment.tokens).unwrap());
        end = segment.end;
    }
}

/// The control tokens render as nothing; an id outside the vocabulary is
/// an error.
#[test]
fn test_text() {
    let engine = VoxtralEngine::new(Arc::new(toy_construct(&performance_device()))).unwrap();
    assert_eq!(engine.text(&[4, 8, 3, 9]).unwrap().as_deref(), Some(" a b"));
    ErrorMatcher::invalid_resource().assert_err(&engine.text(&[-1]));
    ErrorMatcher::invalid_resource().assert_err(&engine.text(&[45]));
}
