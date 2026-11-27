//! # Voxtral Realtime's stream layout, as `mistral_common`'s `tekken.json` gives it
//!
//! [`voxtral_token_layout_config`]: the special ids by their spellings, and
//! the `audio` section's frame rate, delay and left padding, from the shared
//! [`TekkenMirror`].

use alloc::format;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::asr::voxtral::VoxtralTokenLayoutConfig,
        support::interop::mistral_common::TekkenMirror,
    },
};

/// The [`VoxtralTokenLayoutConfig`] a `tekken.json` describes: the special ids
/// by their spellings, and the `audio` section's frame rate, delay and left
/// padding.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
/// when the vocabulary does not read (as
/// [`TekkenMirror::tekken_vocabulary`]), lacks a special token or the
/// `audio` section, or its frame rate is not a whole number of ms per
/// position.
pub fn voxtral_token_layout_config(
    tekken: &TekkenMirror
) -> BunsenResult<VoxtralTokenLayoutConfig> {
    let vocab = tekken.tekken_vocabulary()?;
    let special = |spelling: &str| {
        vocab.special_id(spelling).ok_or_else(|| {
            BunsenError::invalid_resource(format!("tekken.json has no {spelling} token"))
        })
    };
    let audio = tekken
        .audio
        .as_ref()
        .ok_or_else(|| BunsenError::invalid_resource("tekken.json has no `audio` section"))?;
    let position_ms = 1000.0 / audio.frame_rate;
    if position_ms.fract() != 0.0 {
        return Err(BunsenError::invalid_resource(format!(
            "tekken.json: a frame rate of {} Hz is not whole ms per position",
            audio.frame_rate
        )));
    }
    Ok(VoxtralTokenLayoutConfig::new()
        .with_bos(special("<s>")?)
        .with_eos(special("</s>")?)
        .with_streaming_pad(special("[STREAMING_PAD]")?)
        .with_streaming_word(special("[STREAMING_WORD]")?)
        .with_position_ms(position_ms as usize)
        .with_delay_ms(audio.transcription_delay_ms)
        .with_left_pad_positions(audio.streaming_n_left_pad_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::testing::ErrorMatcher;

    #[test]
    fn test_voxtral_token_layout_config() {
        let json = r#"{
            "config": {"default_vocab_size": 40, "default_num_special_tokens": 34},
            "vocab": [],
            "special_tokens": [
                {"rank": 1, "token_str": "<s>"}, {"rank": 2, "token_str": "</s>"},
                {"rank": 32, "token_str": "[STREAMING_PAD]"},
                {"rank": 33, "token_str": "[STREAMING_WORD]"}
            ],
            "audio": {"sampling_rate": 16000, "frame_rate": 12.5,
                      "transcription_delay_ms": 960, "streaming_n_left_pad_tokens": 16}
        }"#;
        let config = voxtral_token_layout_config(&TekkenMirror::from_json(json).unwrap()).unwrap();
        assert_eq!(
            config,
            VoxtralTokenLayoutConfig::new()
                .with_delay_ms(960)
                .with_left_pad_positions(16)
        );
        ErrorMatcher::invalid_resource()
            .message_contains("[STREAMING_WORD]")
            .assert_err(&voxtral_token_layout_config(
                &TekkenMirror::from_json(&json.replace("[STREAMING_WORD]", "[OTHER]")).unwrap(),
            ));
    }
}
