//! `--publish`, and the publisher `transcribe` and `live` push into.
mod config;
#[cfg(feature = "mqtt")]
mod mqtt;
mod record;
#[cfg(feature = "webhook")]
mod webhook;
mod writer;

use std::time::{
    Duration,
    SystemTime,
};

use bunsen::{
    errors::BunsenResult,
    kits::speech::asr_driver::{
        SpeakerLabel,
        TranscriptSegment,
    },
};
pub use config::PublishArgs;
use config::{
    EventKind,
    Opts,
};

use crate::{
    profile::Stamp,
    transcribe_cmd::Transcript,
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

/// What `transcribe` and `live` push into a [`Publisher`].
#[derive(Clone, Copy, Debug)]
pub enum Event<'a> {
    /// A statement, of `file` when the audio is a file's.
    Statement {
        file: Option<&'a str>,
        statement: &'a Statement,
    },

    /// The latest draft of the speech past the last commit (`live` under
    /// `--preset responsive`), from `start` to `end` on the stream's clock.
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
    fn kind(&self) -> EventKind {
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

/// How a stream's clock reads as times: from the start of the audio, or,
/// for the microphone, as the wall clock.
#[derive(Clone, Copy, Debug)]
pub enum Clock {
    /// Seconds from the start of the audio: a file's.
    Offset,

    /// The wall clock, the stream's 0 at `origin`: the microphone's.
    #[cfg_attr(not(feature = "live"), allow(dead_code))]
    Wall { origin: SystemTime },
}

#[cfg_attr(not(feature = "live"), allow(dead_code))]
impl Clock {
    /// The wall clock, the stream's 0 now.
    pub fn wall_now() -> Self {
        Clock::Wall {
            origin: SystemTime::now(),
        }
    }

    /// The wall clock, `seconds` of the stream's clock having passed now.
    pub fn wall_since(seconds: f64) -> Self {
        let now = SystemTime::now();
        Clock::Wall {
            origin: now
                .checked_sub(Duration::from_secs_f64(seconds.max(0.0)))
                .unwrap_or(now),
        }
    }
}

/// What every sink renders an event under: the clock, and whether the
/// statements carry speakers.
#[derive(Clone, Copy, Debug)]
struct Cx {
    clock: Clock,
    diarized: bool,
}

/// One delivery target.
trait Sink {
    /// Delivers `event`, rendered under `cx` as `opts` shows it.
    fn push(
        &mut self,
        event: &Event,
        cx: &Cx,
        opts: &Opts,
    ) -> BunsenResult<()>;

    /// Delivers what is pending, and closes.
    fn finish(self: Box<Self>) -> BunsenResult<()>;
}

/// Pushes each event to every target `--publish` configures: built by
/// [`PublishArgs::init`], fed by `transcribe` and `live`, and
/// [`finish`](Self::finish)ed once the audio ends.
pub struct Publisher {
    sinks: Vec<(Opts, Box<dyn Sink>)>,
    cx: Cx,
}

impl Publisher {
    /// Reads the stream's clock as `clock` from here on: `live` sets the
    /// wall clock's origin once the first push says when capture began.
    #[cfg_attr(not(feature = "live"), allow(dead_code))]
    pub fn set_clock(
        &mut self,
        clock: Clock,
    ) {
        self.cx.clock = clock;
    }

    /// Delivers `event` to each target that takes its kind.
    ///
    /// # Errors
    /// As a target's delivery: a local write that fails. A network target
    /// logs its failures and goes on.
    pub fn push(
        &mut self,
        event: &Event,
    ) -> BunsenResult<()> {
        let kind = event.kind();
        for (opts, sink) in &mut self.sinks {
            if opts.events.contains(&kind) {
                sink.push(event, &self.cx, opts)?;
            }
        }
        Ok(())
    }

    /// Delivers what each target has pending, and closes them all.
    ///
    /// # Errors
    /// As the first target that fails to finish; the rest are finished
    /// regardless.
    pub fn finish(self) -> BunsenResult<()> {
        let mut first = Ok(());
        for (_, sink) in self.sinks {
            let result = sink.finish();
            if first.is_ok() {
                first = result;
            }
        }
        first
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(
        text: &str,
        start: f64,
        speaker: Option<(usize, f32)>,
    ) -> TranscriptSegment {
        TranscriptSegment {
            start,
            end: start + 0.5,
            tokens: Vec::new(),
            text: Some(text.to_string()),
            speaker: speaker.map(|(index, confidence)| SpeakerLabel { index, confidence }),
        }
    }

    /// Consecutive segments of one speaker are one statement, its
    /// confidence their mean; empty ones are left out, and a change of
    /// speaker, nobody included, starts a statement.
    #[test]
    fn test_statements() {
        let st = statements(&[
            segment(" hello", 0.0, Some((0, 0.8))),
            segment("  ", 0.5, Some((1, 0.9))),
            segment("there", 1.0, Some((0, 0.6))),
            segment("hi", 2.0, Some((1, 0.9))),
            segment("um", 3.0, None),
        ]);
        let texts: Vec<&str> = st.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["hello there", "hi", "um"]);
        assert_eq!((st[0].start, st[0].end), (0.0, 1.5));
        let label = st[0].speaker.unwrap();
        assert_eq!(label.index, 0);
        assert!((label.confidence - 0.7).abs() < 1e-6);
        assert_eq!(st[2].speaker, None);

        // Undiarized, the segments are one statement.
        let st = statements(&[segment("the", 0.0, None), segment("moon.", 1.0, None)]);
        assert_eq!(st.len(), 1);
        assert_eq!(st[0].text, "the moon.");
        assert!(statements(&[]).is_empty());
    }
}
