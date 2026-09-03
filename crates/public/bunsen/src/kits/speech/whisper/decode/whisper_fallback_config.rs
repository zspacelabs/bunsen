//! # Fallback: the temperature ladder and its thresholds.
//!
//! Upstream's `transcribe()` decodes each window at temperature zero and,
//! when the result looks bad &mdash; too repetitive by its gzip compression
//! ratio, or too improbable by its average log probability &mdash; decodes
//! it again at the next temperature of a ladder, sampling instead of
//! searching, until one passes or the ladder ends. Silence is the
//! exception: a window whose `<|nospeech|>` probability is high *and* whose
//! log probability is low is accepted as it is, and the seek loop then
//! skips it. A decode that needed a temperature above 0.5 also resets the
//! prompt carry, so a failure does not feed the next window.
//!
//! The ladder is [`decode_with_fallback`], a pure orchestration over a
//! decode closure, so the policy is testable without a model. bunsen's
//! default ladder is temperature zero alone: a stream driver re-decoding a
//! window several times is a latency choice its deployment should make,
//! not a default; [`WhisperFallbackConfig::upstream`] is the full ladder.

use std::io::Write;

use burn::config::Config;
use flate2::{
    Compression,
    write::ZlibEncoder,
};

use crate::kits::speech::whisper::decode::{
    DecodeConfig,
    DecodedTokens,
};

/// The ladder and the thresholds that climb it.
#[derive(Config, Debug, PartialEq)]
pub struct WhisperFallbackConfig {
    /// The temperatures tried in order; the first is the search proper,
    /// the rest are sampling. Never empty.
    #[config(default = "vec![0.0]")]
    pub temperatures: Vec<f64>,

    /// A decode whose text compresses better than this is a repetition
    /// loop, and fails.
    #[config(default = "Some(2.4)")]
    pub compression_ratio_threshold: Option<f64>,

    /// A decode whose average log probability is below this fails.
    #[config(default = "Some(-1.0)")]
    pub logprob_threshold: Option<f64>,

    /// A window whose `<|nospeech|>` probability is above this, when the
    /// decode also failed the log probability threshold, is silence: not a
    /// failure, and skipped.
    #[config(default = "Some(0.6)")]
    pub no_speech_threshold: Option<f64>,

    /// Sample trajectories per audio above temperature zero; `None` is one.
    #[config(default = "None")]
    pub best_of: Option<usize>,
}

impl WhisperFallbackConfig {
    /// Upstream's defaults: the ladder `0, 0.2, ..., 1.0`.
    pub fn upstream() -> Self {
        Self::new().with_temperatures(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    }

    /// Whether a decode fails its thresholds and the next temperature
    /// should be tried.
    ///
    /// # Arguments
    /// * `avg_logprob` - the decode's average log probability.
    /// * `compression_ratio` - of its text, when text was available.
    /// * `no_speech_prob` - when probed.
    pub fn needs_fallback(
        &self,
        avg_logprob: f64,
        compression_ratio: Option<f64>,
        no_speech_prob: Option<f64>,
    ) -> bool {
        let mut needs = false;
        if let (Some(threshold), Some(ratio)) =
            (self.compression_ratio_threshold, compression_ratio)
            && ratio > threshold
        {
            needs = true; // too repetitive
        }
        if let Some(threshold) = self.logprob_threshold
            && avg_logprob < threshold
        {
            needs = true; // too improbable
        }
        if let (Some(no_speech_threshold), Some(no_speech), Some(logprob_threshold)) = (
            self.no_speech_threshold,
            no_speech_prob,
            self.logprob_threshold,
        ) && no_speech > no_speech_threshold
            && avg_logprob < logprob_threshold
        {
            needs = false; // silence
        }
        needs
    }

    /// Whether the seek loop skips a window as silence: its no-speech
    /// probability is over the threshold, unless its log probability is
    /// good enough to keep it anyway.
    pub fn should_skip(
        &self,
        no_speech_prob: Option<f64>,
        avg_logprob: f64,
    ) -> bool {
        let (Some(threshold), Some(no_speech)) = (self.no_speech_threshold, no_speech_prob) else {
            return false;
        };
        let mut skip = no_speech > threshold;
        if let Some(logprob_threshold) = self.logprob_threshold
            && avg_logprob > logprob_threshold
        {
            skip = false;
        }
        skip
    }

    /// Whether a decode at `temperature` resets the prompt carry.
    pub fn resets_prompt(temperature: f64) -> bool {
        temperature > 0.5
    }

    /// The config for one rung: above zero the beam is off and `best_of`
    /// is on, as upstream drops `beam_size` and `patience` for sampling.
    pub fn rung(
        &self,
        base: &DecodeConfig,
        temperature: f64,
    ) -> DecodeConfig {
        let mut config = base.clone().with_temperature(temperature);
        if temperature > 0.0 {
            config.beam_size = 1;
            config.patience = None;
            config.best_of = self.best_of;
        } else {
            config.best_of = None;
        }
        config
    }
}

/// Upstream's `compression_ratio`: the text's UTF-8 length over its zlib
/// compressed length. Empty text is zero.
pub fn compression_ratio(text: &str) -> f64 {
    let bytes = text.as_bytes();
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(bytes).expect("writing to a vector");
    let compressed = encoder.finish().expect("finishing a vector");
    bytes.len() as f64 / compressed.len() as f64
}

/// Decodes one window up the ladder until a rung passes.
///
/// # Arguments
/// * `fallback` - the ladder and thresholds.
/// * `base` - the decode config at temperature zero; each rung derives from it.
/// * `first` - the first rung's result, when the caller already has it (a
///   batched decode at temperature zero, say); `None` decodes it.
/// * `decode` - runs one rung.
/// * `text_of` - the text of a decode's ids, for the compression ratio; `None`
///   when no detokenizer is at hand, which disables that threshold.
///
/// # Returns
/// The last decode run: the first that passed, or the top rung's.
pub fn decode_with_fallback(
    fallback: &WhisperFallbackConfig,
    base: &DecodeConfig,
    first: Option<DecodedTokens>,
    mut decode: impl FnMut(&DecodeConfig) -> DecodedTokens,
    text_of: impl Fn(&[i64]) -> Option<String>,
) -> DecodedTokens {
    assert!(
        !fallback.temperatures.is_empty(),
        "the ladder has at least one rung"
    );
    let mut first = first;
    let mut result = None;

    for &temperature in &fallback.temperatures {
        let decoded = match first.take() {
            Some(given) => given,
            None => decode(&fallback.rung(base, temperature)),
        };
        let ratio = text_of(&decoded.tokens).map(|t| compression_ratio(&t));
        let needs = fallback.needs_fallback(
            decoded.avg_logprob(),
            ratio,
            decoded.no_speech_prob.map(f64::from),
        );
        result = Some(decoded);
        if !needs {
            break;
        }
    }

    result.expect("at least one rung ran")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(
        n: usize,
        sum_logprob: f32,
        no_speech: Option<f32>,
        temperature: f64,
    ) -> DecodedTokens {
        DecodedTokens {
            tokens: (0..n as i64).collect(),
            sum_logprob,
            no_speech_prob: no_speech,
            temperature,
        }
    }

    #[test]
    fn test_defaults_and_upstream() {
        let ours = WhisperFallbackConfig::new();
        assert_eq!(ours.temperatures, vec![0.0]);
        assert_eq!(ours.compression_ratio_threshold, Some(2.4));
        assert_eq!(ours.logprob_threshold, Some(-1.0));
        assert_eq!(ours.no_speech_threshold, Some(0.6));
        assert_eq!(ours.best_of, None);
        assert_eq!(
            WhisperFallbackConfig::upstream().temperatures,
            vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        );
    }

    /// Every clause of the failure test, by hand.
    #[test]
    fn test_needs_fallback() {
        let f = WhisperFallbackConfig::new();
        assert!(
            !f.needs_fallback(-0.5, Some(1.2), Some(0.1)),
            "a good decode"
        );
        assert!(
            f.needs_fallback(-0.5, Some(3.0), Some(0.1)),
            "too repetitive"
        );
        assert!(
            f.needs_fallback(-1.5, Some(1.2), Some(0.1)),
            "too improbable"
        );
        assert!(
            f.needs_fallback(-1.5, None, None),
            "no text and no probe still fail on logprob"
        );
        assert!(
            !f.needs_fallback(-1.5, Some(1.2), Some(0.9)),
            "silence is not a failure"
        );
        assert!(
            f.needs_fallback(-0.5, Some(3.0), Some(0.9)),
            "silence excuses logprob, not repetition"
        );

        let off = WhisperFallbackConfig::new()
            .with_compression_ratio_threshold(None)
            .with_logprob_threshold(None);
        assert!(
            !off.needs_fallback(-9.0, Some(9.0), None),
            "no thresholds, no failure"
        );
    }

    #[test]
    fn test_should_skip() {
        let f = WhisperFallbackConfig::new();
        assert!(f.should_skip(Some(0.9), -1.5));
        assert!(
            !f.should_skip(Some(0.9), -0.5),
            "good logprob keeps a window"
        );
        assert!(!f.should_skip(Some(0.1), -1.5));
        assert!(!f.should_skip(None, -1.5), "unprobed, never skipped");
        let no_logprob = WhisperFallbackConfig::new().with_logprob_threshold(None);
        assert!(no_logprob.should_skip(Some(0.9), -0.5));
        assert!(WhisperFallbackConfig::resets_prompt(0.6));
        assert!(!WhisperFallbackConfig::resets_prompt(0.5));
    }

    /// Against Python's `zlib.compress`: 38 bytes of `la la ...` to 13, a
    /// sentence to more bytes than it had, nothing to nothing.
    #[test]
    fn test_compression_ratio() {
        let looping = "la la la la la la la la la la la la la";
        let ratio = compression_ratio(looping);
        assert!((ratio - 38.0 / 13.0).abs() < 0.3, "{ratio}");
        assert!(ratio > 2.4, "a loop fails the threshold");
        let sentence = compression_ratio(" We choose to go to the moon.");
        assert!(sentence < 1.0, "{sentence}");
        assert_eq!(compression_ratio(""), 0.0);
    }

    /// A rung above zero samples: beam and patience off, best_of on.
    #[test]
    fn test_rung() {
        let base = DecodeConfig::new(vec![1], 0)
            .with_beam_size(5)
            .with_patience(Some(2.0));
        let f = WhisperFallbackConfig::new().with_best_of(Some(3));
        let zero = f.rung(&base, 0.0);
        assert_eq!(
            (
                zero.beam_size,
                zero.patience,
                zero.best_of,
                zero.temperature
            ),
            (5, Some(2.0), None, 0.0)
        );
        let warm = f.rung(&base, 0.4);
        assert_eq!(
            (
                warm.beam_size,
                warm.patience,
                warm.best_of,
                warm.temperature
            ),
            (1, None, Some(3), 0.4)
        );
    }

    /// The ladder: the first rung that passes wins; a given first result
    /// is used as the first rung; when every rung fails the top one is
    /// returned.
    #[test]
    fn test_ladder() {
        let f = WhisperFallbackConfig::upstream();
        let base = DecodeConfig::new(vec![1], 0);

        // Fails at 0 and 0.2 on logprob, passes at 0.4.
        let mut ran = Vec::new();
        let out = decode_with_fallback(
            &f,
            &base,
            None,
            |c| {
                ran.push(c.temperature);
                let sum = if c.temperature < 0.4 { -20.0 } else { -1.0 };
                decoded(4, sum, Some(0.1), c.temperature)
            },
            |_| None,
        );
        assert_eq!(ran, vec![0.0, 0.2, 0.4]);
        assert_eq!(out.temperature, 0.4);

        // A given first result stands in for the first rung.
        let mut ran = Vec::new();
        let out = decode_with_fallback(
            &f,
            &base,
            Some(decoded(4, -20.0, Some(0.1), 0.0)),
            |c| {
                ran.push(c.temperature);
                decoded(4, -1.0, Some(0.1), c.temperature)
            },
            |_| None,
        );
        assert_eq!(ran, vec![0.2]);
        assert_eq!(out.temperature, 0.2);

        // Every rung fails: the top one comes back.
        let out = decode_with_fallback(
            &f,
            &base,
            None,
            |c| decoded(4, -20.0, Some(0.1), c.temperature),
            |_| None,
        );
        assert_eq!(out.temperature, 1.0);

        // Repetition, seen through the text, fails the first rung.
        let mut ran = Vec::new();
        let out = decode_with_fallback(
            &f,
            &base,
            None,
            |c| {
                ran.push(c.temperature);
                decoded(4, -0.1, Some(0.1), c.temperature)
            },
            |ids| {
                Some(if ids.len() == 4 {
                    "la ".repeat(20)
                } else {
                    "fine".into()
                })
            },
        );
        assert_eq!(out.temperature, 1.0, "the closure loops at every rung");
        assert_eq!(ran.len(), 6);

        // Silence at temperature zero is accepted, not climbed.
        let mut ran = Vec::new();
        let _ = decode_with_fallback(
            &f,
            &base,
            None,
            |c| {
                ran.push(c.temperature);
                decoded(0, -3.0, Some(0.95), c.temperature)
            },
            |_| None,
        );
        assert_eq!(ran, vec![0.0]);
    }
}
