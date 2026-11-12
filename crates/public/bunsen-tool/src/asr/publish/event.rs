use bunsen::kits::speech::asr::driver::{
    SpeakerLabel,
    TranscriptSegment,
};

use super::config::EventKind;
use crate::asr::{
    Stamp,
    Transcript,
};

/// A run of committed text one speaker spoke; undiarized, a run of text.
#[derive(Clone, Debug, PartialEq)]
pub struct Statement {
    /// Its start on the stream's clock, in seconds.
    pub start: f64,

    /// Its end on the stream's clock, in seconds.
    pub end: f64,

    /// The segments' texts, trimmed and joined with spaces.
    pub text: String,

    /// Under `--diarize`, its speaker, `None` for nobody, the confidence the
    /// mean of its segments'; `None` undiarized.
    pub speaker: Option<SpeakerLabel>,
}

/// What `transcribe` and `live` push into a [`Publisher`](super::Publisher).
#[derive(Clone, Copy, Debug)]
pub enum Event<'a> {
    /// A statement, of `file` when the audio is a file's.
    Statement {
        file: Option<&'a str>,
        statement: &'a Statement,
    },

    /// The latest draft of the speech past the last commit (`live` under
    /// `--emission drafting`), from `start` to `end` on the stream's clock.
    #[cfg_attr(not(feature = "live"), allow(dead_code))]
    Draft { start: f64, end: f64, text: &'a str },

    /// A file's whole transcript, and what chose its model, after its
    /// statements.
    Transcript {
        file: &'a str,
        stamp: Option<&'a Stamp>,
        transcript: &'a Transcript,
    },
}

impl Event<'_> {
    /// The event's kind, as records name it.
    pub(super) fn kind(&self) -> EventKind {
        match self {
            Event::Statement { .. } => EventKind::Statement,
            Event::Draft { .. } => EventKind::Draft,
            Event::Transcript { .. } => EventKind::Transcript,
        }
    }
}

/// The statements of `segments`: runs of one speaker (undiarized, every
/// segment is nobody's, so one run), their texts joined; the segments with
/// no text are left out.
pub fn statements(segments: &[TranscriptSegment]) -> Vec<Statement> {
    // Each run's statement, and its segments' summed confidence and count.
    let mut runs: Vec<(Statement, f32, usize)> = Vec::new();
    for segment in segments {
        let Some(text) = segment
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        else {
            continue;
        };
        let index = segment.speaker.map(|label| label.index);
        let confidence = segment.speaker.map_or(0.0, |label| label.confidence);
        match runs.last_mut() {
            Some((run, sum, n)) if run.speaker.map(|label| label.index) == index => {
                run.end = segment.end;
                run.text.push(' ');
                run.text.push_str(text);
                *sum += confidence;
                *n += 1;
            }
            _ => runs.push((
                Statement {
                    start: segment.start,
                    end: segment.end,
                    text: text.to_string(),
                    speaker: segment.speaker,
                },
                confidence,
                1,
            )),
        }
    }
    runs.into_iter()
        .map(|(mut statement, sum, n)| {
            if let Some(label) = &mut statement.speaker {
                label.confidence = sum / n as f32;
            }
            statement
        })
        .collect()
}
