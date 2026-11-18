//! Speech synthesis judged by speech recognition: each text spoken, the
//! speech transcribed, the transcript scored against the text
//! ([`tts_asr_round_trip`]).
//!
//! A sampled take can match no reference sample for sample, so this judges
//! what a take is for: whether what it says can be understood. Any
//! synthesizer and any recognizer: each is a closure, and the speech is
//! resampled from the synthesizer's rate to the recognizer's.

use alloc::{
    string::String,
    vec::Vec,
};

use super::wer::{
    normalize_transcript,
    word_errors,
};
use crate::{
    errors::BunsenResult,
    support::audio::resample,
};

/// One text's round trip.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundTripCase {
    /// The text spoken.
    pub text: String,

    /// What the recognizer heard.
    pub transcript: String,

    /// The speech's length, in seconds.
    pub seconds: f64,

    /// The word errors ([`word_errors`]).
    pub errors: usize,

    /// The text's words ([`normalize_transcript`]).
    pub words: usize,
}

impl RoundTripCase {
    /// The word error rate: errors over the text's words.
    pub fn word_error_rate(&self) -> f64 {
        if self.words == 0 {
            return if self.errors == 0 { 0.0 } else { 1.0 };
        }
        self.errors as f64 / self.words as f64
    }
}

/// A round trip's cases, and their pooled word error rate.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundTripReport {
    /// Each text's, in order.
    pub cases: Vec<RoundTripCase>,
}

impl RoundTripReport {
    /// Every case's errors over every case's words.
    pub fn word_error_rate(&self) -> f64 {
        let errors: usize = self.cases.iter().map(|c| c.errors).sum();
        let words: usize = self.cases.iter().map(|c| c.words).sum();
        if words == 0 {
            return if errors == 0 { 0.0 } else { 1.0 };
        }
        errors as f64 / words as f64
    }

    /// The worst case's word error rate; zero without cases.
    pub fn worst(&self) -> f64 {
        self.cases
            .iter()
            .map(RoundTripCase::word_error_rate)
            .fold(0.0, f64::max)
    }
}

/// Speaks each of `texts`, transcribes the speech, and scores each
/// transcript against its text, words normalized as
/// [`normalize_transcript`] does.
///
/// # Arguments
/// - `texts`: what to speak.
/// - `synthesize`: a text to its speech, mono samples and their rate in Hz.
/// - `asr_rate`: the rate `transcribe` reads, in Hz; the speech is resampled to
///   it ([`resample`]).
/// - `transcribe`: mono samples at `asr_rate` to their transcript.
///
/// # Errors
/// The first of `synthesize`'s or `transcribe`'s.
pub fn tts_asr_round_trip<S, T>(
    texts: &[&str],
    mut synthesize: S,
    asr_rate: usize,
    mut transcribe: T,
) -> BunsenResult<RoundTripReport>
where
    S: FnMut(&str) -> BunsenResult<(Vec<f32>, usize)>,
    T: FnMut(&[f32]) -> BunsenResult<String>,
{
    let mut cases = Vec::with_capacity(texts.len());
    for &text in texts {
        let (speech, rate) = synthesize(text)?;
        let heard = resample(&speech, rate, asr_rate);
        let transcript = transcribe(&heard)?;
        let reference = normalize_transcript(text);
        cases.push(RoundTripCase {
            text: text.into(),
            errors: word_errors(&normalize_transcript(&transcript), &reference),
            words: reference.len(),
            seconds: speech.len() as f64 / rate as f64,
            transcript,
        });
    }
    Ok(RoundTripReport { cases })
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use core::cell::RefCell;

    use super::*;
    use crate::errors::BunsenError;

    /// A synthesizer that "speaks" each text as a second of silence at 8
    /// kHz, and a recognizer that hears 16 kHz and echoes the text, two
    /// words wrong in the third: the round trip carries the rates and the
    /// scores.
    #[test]
    fn test_round_trip() {
        let texts = ["Hello, world.", "one two three", "a b c d"];
        let spoken = RefCell::new(Vec::new());
        let report = tts_asr_round_trip(
            &texts,
            |text| {
                spoken.borrow_mut().push(text.to_string());
                Ok((vec![0.0; 8000], 8000))
            },
            16000,
            |samples| {
                assert_eq!(samples.len(), 16000, "resampled to the recognizer's rate");
                let text = spoken.borrow().last().unwrap().clone();
                Ok(if text.starts_with("a b") {
                    "a c".to_string()
                } else {
                    text.to_lowercase()
                })
            },
        )
        .unwrap();
        assert_eq!(report.cases.len(), 3);
        assert_eq!(report.cases[0].transcript, "hello, world.");
        assert_eq!(report.cases[0].word_error_rate(), 0.0);
        assert_eq!(report.cases[2].errors, 2);
        assert_eq!(report.cases[2].seconds, 1.0);
        assert_eq!(report.worst(), 0.5);
        // 2 errors over 2 + 3 + 4 words.
        assert_eq!(report.word_error_rate(), 2.0 / 9.0);
    }

    #[test]
    fn test_round_trip_errors() {
        let error = tts_asr_round_trip(
            &["x"],
            |_| Err(BunsenError::illegal("no speech")),
            16000,
            |_| Ok(String::new()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("no speech"));
    }
}
