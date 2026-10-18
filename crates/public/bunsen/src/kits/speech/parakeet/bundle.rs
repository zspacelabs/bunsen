//! A loaded Parakeet model, with what it needs to transcribe.
use alloc::{
    string::String,
    vec,
    vec::Vec,
};

use burn::{
    Tensor,
    tensor::Device,
};

use super::{
    ParakeetFrontEnd,
    ParakeetFrontEndConfig,
    ParakeetTdt,
    ParakeetVocabulary,
    TdtGreedyDecoderConfig,
    TdtHypothesis,
};
use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// A Parakeet TDT model, its front end, and its vocabulary: what a
/// pretrained loads to, and everything a transcription needs.
///
/// [`default_parakeet_factory`](super::pretrained::default_parakeet_factory)
/// builds one from a name.
#[derive(Debug)]
pub struct ParakeetBundle {
    /// The model.
    pub model: ParakeetTdt,

    /// The feature extractor the model was trained behind.
    pub front_end: ParakeetFrontEnd,

    /// The pieces the model's token ids spell.
    pub vocab: ParakeetVocabulary,
}

/// One transcribed item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParakeetTranscript {
    /// The decode: tokens, and the encoder frames they were emitted at.
    pub hypothesis: TdtHypothesis,

    /// The text the tokens spell.
    pub text: String,
}

/// A word of a transcript, and when it was spoken.
#[derive(Clone, Debug, PartialEq)]
pub struct TimedWord {
    /// The word, with any punctuation its pieces carry.
    pub text: String,

    /// When its first token was emitted, in seconds.
    pub start: f64,

    /// When its last token's duration ends, in seconds.
    pub end: f64,
}

impl ParakeetTranscript {
    /// The transcript's words, timed by the encoder frames their tokens were
    /// emitted at.
    ///
    /// A word begins at a piece that begins with the word mark (`▁`), and
    /// takes every following piece that does not; special pieces are
    /// skipped. It starts at its first token's frame and ends where its last
    /// token's duration (at least one frame) ends; `frame_seconds` is
    /// [`ParakeetBundle::frame_seconds`].
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if a token is outside `vocab`.
    pub fn words(
        &self,
        vocab: &ParakeetVocabulary,
        frame_seconds: f64,
    ) -> BunsenResult<Vec<TimedWord>> {
        let h = &self.hypothesis;
        let mut words: Vec<TimedWord> = Vec::new();
        for (k, &token) in h.tokens.iter().enumerate() {
            if vocab.is_special(token) {
                continue;
            }
            let piece = vocab.transcript(&[token])?;
            let start = h.frames[k] as f64 * frame_seconds;
            let end = (h.frames[k] + h.durations[k].max(1)) as f64 * frame_seconds;
            let begins = vocab
                .piece(token)
                .is_some_and(|p| p.starts_with(super::WORD_MARK));
            match words.last_mut() {
                Some(word) if !begins => {
                    word.text.push_str(&piece);
                    word.end = end;
                }
                _ => words.push(TimedWord {
                    text: piece,
                    start,
                    end,
                }),
            }
        }
        Ok(words)
    }
}

impl ParakeetBundle {
    /// A bundle of `model` and `vocab`, behind the default front end.
    ///
    /// # Errors
    /// As [`ParakeetFrontEndConfig`]'s `try_init`.
    pub fn new(
        model: ParakeetTdt,
        vocab: ParakeetVocabulary,
        device: &Device,
    ) -> BunsenResult<Self> {
        Ok(Self {
            model,
            front_end: ParakeetFrontEndConfig::new().try_init(device)?,
            vocab,
        })
    }

    /// The duration of one encoder frame, in seconds: the front end's hop
    /// times the encoder's subsampling.
    pub fn frame_seconds(&self) -> f64 {
        use super::{
            ParakeetFrontEndMeta,
            ParakeetTdtMeta,
        };
        (self.front_end.hop() * self.model.subsampling_factor()) as f64
            / self.front_end.sample_rate() as f64
    }

    /// The encoder frame, in samples: the front end's hop times the encoder's
    /// subsampling.
    pub fn frame_samples(&self) -> usize {
        use super::{
            ParakeetFrontEndMeta,
            ParakeetTdtMeta,
        };
        self.front_end.hop() * self.model.subsampling_factor()
    }

    /// Transcribes one waveform of any length, in windows of at most
    /// `max_window` samples.
    ///
    /// The encoder attends over its whole input, so its cost grows with the
    /// square of the input's length; a long recording is cut into windows,
    /// each cut at the quietest encoder frame of the window's last sixth,
    /// and the windows are transcribed `batch` at a time. The hypotheses are
    /// stitched in order, their frames counted from the start of `samples`.
    ///
    /// # Errors
    /// As [`transcribe`](Self::transcribe).
    ///
    /// # Panics
    /// If `max_window` is shorter than six encoder frames, or `batch` is 0.
    pub fn transcribe_long(
        &self,
        samples: &[f32],
        max_window: usize,
        batch: usize,
        device: &Device,
    ) -> BunsenResult<ParakeetTranscript> {
        let frame = self.frame_samples();
        assert!(
            max_window >= 6 * frame,
            "max_window shorter than six frames"
        );
        assert!(batch > 0, "batch is 0");
        let windows = crate::ops::signal::quiet_cuts(samples, max_window, frame);

        let mut joined = TdtHypothesis::default();
        let mut texts = Vec::new();
        for group in windows.chunks(batch) {
            let longest = group.iter().map(|(a, b)| b - a).max().unwrap_or(0);
            let mut data = vec![0.0f32; group.len() * longest];
            for (row, &(a, b)) in group.iter().enumerate() {
                data[row * longest..row * longest + (b - a)].copy_from_slice(&samples[a..b]);
            }
            let lengths: Vec<usize> = group.iter().map(|(a, b)| b - a).collect();
            let waveforms = Tensor::<2>::from_data(
                burn::prelude::TensorData::new(data, [group.len(), longest]),
                device,
            );
            for (transcript, &(a, _)) in self
                .transcribe(waveforms, Some(&lengths))?
                .into_iter()
                .zip(group)
            {
                let offset = a / frame;
                let h = transcript.hypothesis;
                joined.tokens.extend(h.tokens);
                joined.frames.extend(h.frames.iter().map(|f| f + offset));
                joined.durations.extend(h.durations);
                if !transcript.text.is_empty() {
                    texts.push(transcript.text);
                }
            }
        }
        Ok(ParakeetTranscript {
            hypothesis: joined,
            text: texts.join(" "),
        })
    }

    /// Transcribes a batch of waveforms with the greedy decoder.
    ///
    /// # Arguments
    /// * `waveforms` - `[batch, samples]`, mono, at the front end's sample
    ///   rate.
    /// * `lengths` - each item's valid sample count; `None` means every item
    ///   fills `samples`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if a decoded token is outside the vocabulary.
    pub fn transcribe(
        &self,
        waveforms: Tensor<2>,
        lengths: Option<&[usize]>,
    ) -> BunsenResult<Vec<ParakeetTranscript>> {
        let (features, frames) = self.front_end.forward(waveforms, lengths);
        let (encoded, frames) = self.model.encode(features, Some(&frames));
        TdtGreedyDecoderConfig::new()
            .init()
            .decode(&self.model, encoded, &frames)
            .into_iter()
            .map(|hypothesis| {
                let text = self.vocab.transcript(&hypothesis.tokens)?;
                Ok(ParakeetTranscript { hypothesis, text })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_words() {
        let vocab = ParakeetVocabulary::new(
            [
                "<unk>",
                "\u{2581}the",
                "\u{2581}mo",
                "on",
                ".",
                "\u{2581}why",
            ]
            .map(String::from)
            .to_vec(),
        );
        let transcript = ParakeetTranscript {
            hypothesis: TdtHypothesis {
                tokens: vec![1, 0, 2, 3, 4, 5],
                frames: vec![3, 4, 5, 6, 6, 10],
                durations: vec![2, 1, 1, 0, 2, 4],
            },
            text: "the moon. why".into(),
        };
        let words = transcript.words(&vocab, 0.08).unwrap();
        let summary: Vec<(String, f64, f64)> = words
            .iter()
            .map(|w| {
                (
                    w.text.clone(),
                    (w.start * 100.0).round(),
                    (w.end * 100.0).round(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                ("the".to_string(), 24.0, 40.0),
                ("moon.".to_string(), 40.0, 64.0),
                ("why".to_string(), 80.0, 112.0),
            ]
        );
    }
}
