//! The Voxtral Realtime stream driver: audio pushed in chunks, timed
//! transcript out.
//!
//! The driver and its stream context are the
//! [`asr_driver`](crate::kits::speech::asr_driver) core's
//! [`AsrStreamDriver`](crate::kits::speech::asr_driver::AsrStreamDriver) and
//! [`AsrStreamContext`](crate::kits::speech::asr_driver::AsrStreamContext)
//! over a [`VoxtralEngine`]: what does not depend on the model (scheduling,
//! the emission policy, the clock, the voice-activity gate, speakers) lives
//! there, as it does for every speech-to-text kit, and `bunsen-tool` drives
//! them all alike.
//!
//! # Lifecycle
//!
//! 1. A [`VoxtralBundle`](super::VoxtralBundle): from a name, through
//!    [`default_voxtral_factory`](super::pretrained::default_voxtral_factory)
//!    and a [`PretrainedCache`](crate::data::pretrained::PretrainedCache)
//!    (features `store_safetensors` and `cache`), or over a model in hand;
//!    [`with_delay_ms`](super::VoxtralBundle::with_delay_ms) sets how far the
//!    text runs behind the audio.
//! 2. A [`VoxtralEngine`] over it, its window set by
//!    [`with_window_seconds`](VoxtralEngine::with_window_seconds).
//! 3. A [`VoxtralStreamDriver`] over the engine and an
//!    [`EmissionPolicy`](crate::kits::speech::asr_driver::EmissionPolicy).
//! 4. One [`VoxtralStreamContext`] per stream, samples in through `write_read`,
//!    [`TranscriptEvent`](crate::kits::speech::asr_driver::TranscriptEvent)s
//!    out.
//!
//! # Windows
//!
//! The engine decodes a window whole, as `transformers` decodes an
//! utterance: padded with silence, then one decoder step per 80 ms of it.
//! Voxtral times every word itself (its `[STREAMING_WORD]` marker is written
//! where the word starts, a fixed delay after the audio), so a window is laid
//! out a segment per word, and a long stream is cut between words. The
//! model's natively incremental decode, one step per 80 ms written, without
//! windows, is a context of its own: it carries the convolution stem's left
//! frames and both stacks' caches across writes.

mod engine;

pub use engine::*;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use burn::tensor::Device;

    use super::*;
    use crate::{
        burner::module::ModuleInit,
        kits::{
            speech::{
                asr_driver::{
                    AsrEngine,
                    EmissionPolicy,
                    LevelPolicy,
                    StreamClock,
                    TranscriptEvent,
                },
                testing::ToneConfig,
                voxtral::{
                    VoxtralBundle,
                    VoxtralRealtime,
                    VoxtralTokenLayoutConfig,
                    blocks::{
                        VoxtralAudioEncoderConfig,
                        VoxtralRealtimeConfig,
                        VoxtralTextConfig,
                    },
                },
            },
            tokens::TekkenVocabulary,
        },
        support::testing::performance_device,
    };

    /// An untrained toy bundle: 128 mels, a 40-token vocabulary whose ids
    /// past the 8 specials are single letters.
    fn toy_bundle(device: &Device) -> VoxtralBundle {
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
        VoxtralBundle::new(
            model,
            TekkenVocabulary::from_tekken_json(&json).unwrap(),
            layout,
        )
        .unwrap()
    }

    fn base64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n =
                chunk.iter().fold(0u32, |n, &b| n << 8 | u32::from(b)) << (8 * (3 - chunk.len()));
            for k in 0..=chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * k) & 63) as usize] as char);
            }
            out.extend(std::iter::repeat_n('=', 3 - chunk.len()));
        }
        out
    }

    /// Streamed in pieces, a clip longer than a window commits segments in
    /// order, over the clip, each spelling its tokens.
    #[test]
    fn test_streams_through_the_core() {
        let device = performance_device();
        let engine = VoxtralEngine::new(Arc::new(toy_bundle(&device))).with_window_seconds(1.0);
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
            let TranscriptEvent::Committed(segment) = event else {
                panic!("the offline policy only commits: {event:?}");
            };
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
        let engine = VoxtralEngine::new(Arc::new(toy_bundle(&performance_device())));
        assert_eq!(engine.text(&[4, 8, 3, 9]).unwrap().as_deref(), Some(" a b"));
        assert!(engine.text(&[-1]).is_err());
        assert!(engine.text(&[45]).is_err());
    }
}
