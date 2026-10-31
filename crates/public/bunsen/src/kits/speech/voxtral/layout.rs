//! The Voxtral Realtime token layout.
use burn::config::Config;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        ConstraintError,
    },
    kits::tokens::TekkenVocabulary,
};

/// The longest delay a Voxtral Realtime model takes but 2400 ms, in ms.
pub const VOXTRAL_MAX_DELAY_MS: usize = 1200;

/// The delay Voxtral Realtime also takes past [`VOXTRAL_MAX_DELAY_MS`], in
/// ms.
pub const VOXTRAL_LONG_DELAY_MS: usize = 2400;

/// Config for [`VoxtralTokenLayout`]: what `tekken.json` and its `audio`
/// section say about the stream, the published model's by default.
#[derive(Config, Debug, PartialEq)]
pub struct VoxtralTokenLayoutConfig {
    /// `<s>`, the prompt's first token.
    #[config(default = "1")]
    pub bos: usize,

    /// `</s>`, which ends a decode.
    #[config(default = "2")]
    pub eos: usize,

    /// `[STREAMING_PAD]`: a position with nothing new to write.
    #[config(default = "32")]
    pub streaming_pad: usize,

    /// `[STREAMING_WORD]`: a word starts here; its text follows.
    #[config(default = "33")]
    pub streaming_word: usize,

    /// The audio one decoder position covers, in ms (`1000 / frame_rate`).
    #[config(default = "80")]
    pub position_ms: usize,

    /// How far the text runs behind the audio, in ms
    /// (`transcription_delay_ms`): a multiple of `position_ms` up to
    /// [`VOXTRAL_MAX_DELAY_MS`], or [`VOXTRAL_LONG_DELAY_MS`].
    #[config(default = "480")]
    pub delay_ms: usize,

    /// The positions of silence before the audio
    /// (`streaming_n_left_pad_tokens`).
    #[config(default = "32")]
    pub left_pad_positions: usize,

    /// The positions of silence after the audio beyond the delay and the
    /// first position: room for the last word to finish
    /// (`mistral_common`'s `OFFLINE_STREAMING_BUFFER_TOKENS`).
    #[config(default = "10")]
    pub right_buffer_positions: usize,
}

impl VoxtralTokenLayoutConfig {
    /// The layout `tekken.json` describes: the special ids by their
    /// spellings, and the `audio` section's frame rate, delay and left
    /// padding.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when the file does not parse (as
    /// [`TekkenVocabulary::from_tekken_json`]), lacks a special token or the
    /// `audio` section, or its frame rate is not a whole number of ms per
    /// position.
    pub fn from_tekken_json(json: &str) -> BunsenResult<Self> {
        let vocab = TekkenVocabulary::from_tekken_json(json)?;
        let special = |spelling: &str| {
            vocab.special_id(spelling).ok_or_else(|| {
                BunsenError::invalid_resource(format!("tekken.json has no {spelling} token"))
            })
        };
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| BunsenError::invalid_resource("a tekken.json").with_cause(e))?;
        let audio = value
            .get("audio")
            .ok_or_else(|| BunsenError::invalid_resource("tekken.json has no `audio` section"))?;
        let number = |name: &str| {
            audio.get(name).and_then(|v| v.as_f64()).ok_or_else(|| {
                BunsenError::invalid_resource(format!(
                    "tekken.json: `audio.{name}` is missing or not a number"
                ))
            })
        };
        let frame_rate = number("frame_rate")?;
        let position_ms = 1000.0 / frame_rate;
        if position_ms.fract() != 0.0 {
            return Err(BunsenError::invalid_resource(format!(
                "tekken.json: a frame rate of {frame_rate} Hz is not whole ms per position"
            )));
        }
        Ok(Self::new()
            .with_bos(special("<s>")?)
            .with_eos(special("</s>")?)
            .with_streaming_pad(special("[STREAMING_PAD]")?)
            .with_streaming_word(special("[STREAMING_WORD]")?)
            .with_position_ms(position_ms as usize)
            .with_delay_ms(number("transcription_delay_ms")? as usize)
            .with_left_pad_positions(number("streaming_n_left_pad_tokens")? as usize))
    }

    /// Checks the delay is one the model takes.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause naming `delay_ms`, otherwise.
    pub fn validate(&self) -> BunsenResult<()> {
        let ok = self.position_ms > 0
            && self.delay_ms > 0
            && self.delay_ms.is_multiple_of(self.position_ms)
            && (self.delay_ms <= VOXTRAL_MAX_DELAY_MS || self.delay_ms == VOXTRAL_LONG_DELAY_MS);
        if ok {
            Ok(())
        } else {
            Err(ConstraintError::out_of_range(
                "VoxtralTokenLayoutConfig",
                "delay_ms",
                self.delay_ms.to_string(),
                format!(
                    "a multiple of {} ms up to {VOXTRAL_MAX_DELAY_MS}, or {VOXTRAL_LONG_DELAY_MS}",
                    self.position_ms
                ),
            )
            .into())
        }
    }

    /// Builds the layout.
    ///
    /// # Errors
    /// As [`validate`](Self::validate).
    pub fn try_init(&self) -> BunsenResult<VoxtralTokenLayout> {
        self.validate()?;
        Ok(VoxtralTokenLayout {
            config: self.clone(),
        })
    }

    /// Builds the layout; [`try_init`](Self::try_init) is its fallible half.
    ///
    /// # Panics
    /// With `try_init`'s error.
    pub fn init(&self) -> VoxtralTokenLayout {
        use crate::errors::WithOkOrPanic;
        self.try_init().ok_or_panic()
    }
}

/// What a Voxtral Realtime stream is laid out as: one decoder position per
/// [`position_ms`](VoxtralTokenLayoutConfig::position_ms) of audio, the text
/// [`delay_positions`](Self::delay_positions) behind it.
///
/// An utterance is decoded as `mistral_common` lays it out offline:
/// [`left_pad_positions`](VoxtralTokenLayoutConfig::left_pad_positions) of
/// silence before the audio, which is rounded up to whole positions, and
/// [`right_pad_positions`](Self::right_pad_positions) after it, so the last
/// words are written. The prompt is `<s>` and a `[STREAMING_PAD]` for every
/// left-pad and delay position; after it, the model writes one token per
/// position: `[STREAMING_PAD]` when there is nothing new, `[STREAMING_WORD]`
/// where a word starts, then the word's text tokens.
///
/// Built by [`VoxtralTokenLayoutConfig`].
#[derive(Clone, Debug, PartialEq)]
pub struct VoxtralTokenLayout {
    config: VoxtralTokenLayoutConfig,
}

impl VoxtralTokenLayout {
    /// The config this was built from.
    pub fn config(&self) -> &VoxtralTokenLayoutConfig {
        &self.config
    }

    /// The delay, in positions: the model's conditioning, `6` at 480 ms.
    pub fn delay_positions(&self) -> usize {
        self.config.delay_ms / self.config.position_ms
    }

    /// The positions of silence after the audio: the delay, one more, and the
    /// buffer.
    pub fn right_pad_positions(&self) -> usize {
        self.delay_positions() + 1 + self.config.right_buffer_positions
    }

    /// The prompt: `<s>`, then a `[STREAMING_PAD]` for every left-pad and
    /// delay position.
    pub fn prompt(&self) -> Vec<i64> {
        let pads = self.config.left_pad_positions + self.delay_positions();
        std::iter::once(self.config.bos as i64)
            .chain(std::iter::repeat_n(self.config.streaming_pad as i64, pads))
            .collect()
    }

    /// `samples` laid out for an offline decode: the left padding, the
    /// audio, silence to the next whole position, and the right padding.
    ///
    /// # Arguments
    /// * `samples_per_position` - the audio one position covers, in samples:
    ///   the front end's hop times the model's mel frames per position.
    pub fn pad(
        &self,
        samples: &[f32],
        samples_per_position: usize,
    ) -> Vec<f32> {
        let left = self.config.left_pad_positions * samples_per_position;
        let right = samples.len().next_multiple_of(samples_per_position) - samples.len()
            + self.right_pad_positions() * samples_per_position;
        let mut out = vec![0.0; left];
        out.extend_from_slice(samples);
        out.resize(out.len() + right, 0.0);
        out
    }

    /// Whether `id` is a stream control token (`[STREAMING_PAD]`,
    /// `[STREAMING_WORD]`, `<s>`, `</s>`) rather than text.
    pub fn is_control(
        &self,
        id: i64,
    ) -> bool {
        let c = &self.config;
        [c.streaming_pad, c.streaming_word, c.bos, c.eos].contains(&(id as usize))
    }

    /// The audio time a word whose `[STREAMING_WORD]` is written at decoder
    /// position `position` starts at, in seconds: the position's audio less
    /// the left padding and the delay, not before 0.
    pub fn word_start_seconds(
        &self,
        position: usize,
    ) -> f64 {
        let lead = self.config.left_pad_positions + self.delay_positions();
        (position.saturating_sub(lead) * self.config.position_ms) as f64 / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::ErrorMatcher,
    };

    #[test]
    fn test_defaults_are_the_published_models() {
        let layout = VoxtralTokenLayoutConfig::new().init();
        assert_eq!(layout.delay_positions(), 6);
        assert_eq!(layout.right_pad_positions(), 17);
        let prompt = layout.prompt();
        assert_eq!(prompt.len(), 39);
        assert_eq!(prompt[0], 1);
        assert!(prompt[1..].iter().all(|&t| t == 32));
        assert!(layout.is_control(33) && !layout.is_control(1656));
        // 38 positions of left padding and delay lead the audio.
        assert_eq!(layout.word_start_seconds(39), 0.08);
        assert_eq!(layout.word_start_seconds(3), 0.0);
    }

    #[test]
    fn test_pad() {
        let layout = VoxtralTokenLayoutConfig::new()
            .with_left_pad_positions(2)
            .with_right_buffer_positions(0)
            .with_delay_ms(80)
            .init();
        let out = layout.pad(&[1.0; 5], 4);
        // 8 of left padding, 5 samples, 3 to a whole position, then (1 + 1)
        // positions.
        assert_eq!(out.len(), 8 + 5 + 3 + 8);
        assert_eq!(&out[8..13], &[1.0; 5]);
        assert!(out[..8].iter().chain(&out[13..]).all(|&s| s == 0.0));
    }

    #[test]
    fn test_delays() {
        for ms in [80, 480, 1200, 2400] {
            assert!(
                VoxtralTokenLayoutConfig::new()
                    .with_delay_ms(ms)
                    .validate()
                    .is_ok()
            );
        }
        for ms in [0, 100, 1280, 2000] {
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .message_contains("delay_ms")
                .assert_err(&VoxtralTokenLayoutConfig::new().with_delay_ms(ms).try_init());
        }
    }

    #[test]
    fn test_from_tekken_json() {
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
        let config = VoxtralTokenLayoutConfig::from_tekken_json(json).unwrap();
        assert_eq!(
            config,
            VoxtralTokenLayoutConfig::new()
                .with_delay_ms(960)
                .with_left_pad_positions(16)
        );
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource)
            .message_contains("[STREAMING_WORD]")
            .assert_err(&VoxtralTokenLayoutConfig::from_tekken_json(
                &json.replace("[STREAMING_WORD]", "[OTHER]"),
            ));
    }
}
