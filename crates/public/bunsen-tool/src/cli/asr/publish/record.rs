//! An event as each format renders it: a JSON line, a text line, an RTTM line.
use std::{
    path::Path,
    time::Duration,
};

use bunsen::kits::speech::{
    asr_driver::SpeakerLabel,
    diarization::RttmSegment,
};
use serde::Serialize;
use time::OffsetDateTime;

use super::{
    Clock,
    Cx,
    Event,
    Statement,
    config::Opts,
};
use crate::cli::asr::{
    diarize::{
        recording_id,
        speaker_name,
    },
    profile::Stamp,
    transcribe_cmd::{
        Transcript,
        millis,
    },
};

/// A time as a record carries it: seconds from the start of a file's audio,
/// rounded to the millisecond; RFC 3339 in UTC, to the millisecond, for the
/// microphone.
#[derive(Serialize)]
#[serde(untagged)]
enum At {
    Offset(f64),
    Wall(String),
}

/// The wall-clock time `seconds` past `origin`.
fn wall(
    clock: &Clock,
    seconds: f64,
) -> Option<OffsetDateTime> {
    match clock {
        Clock::Offset => None,
        Clock::Wall { origin } => Some(OffsetDateTime::from(
            *origin + Duration::from_secs_f64(seconds.max(0.0)),
        )),
    }
}

impl At {
    fn new(
        clock: &Clock,
        seconds: f64,
    ) -> Self {
        match wall(clock, seconds) {
            None => At::Offset(millis(seconds)),
            Some(t) => At::Wall(format!(
                "{:04}-{:02}-{:02}T{}Z",
                t.year(),
                u8::from(t.month()),
                t.day(),
                clock_time(t.hour(), t.minute(), t.second(), t.millisecond())
            )),
        }
    }
}

/// `HH:MM:SS.mmm`.
fn clock_time(
    hour: u8,
    minute: u8,
    second: u8,
    milli: u16,
) -> String {
    format!("{hour:02}:{minute:02}:{second:02}.{milli:03}")
}

/// A time as a text line shows it: `HH:MM:SS.mmm` from the start of a
/// file's audio; the wall clock's time of day in UTC, `HH:MM:SS.mmmZ`, for
/// the microphone.
pub fn text_time(
    clock: &Clock,
    seconds: f64,
) -> String {
    match wall(clock, seconds) {
        Some(t) => format!(
            "{}Z",
            clock_time(t.hour(), t.minute(), t.second(), t.millisecond())
        ),
        None => {
            let ms = (seconds.max(0.0) * 1000.0).round() as u64;
            clock_time(
                (ms / 3_600_000).min(u8::MAX as u64) as u8,
                (ms / 60_000 % 60) as u8,
                (ms / 1000 % 60) as u8,
                (ms % 1000) as u16,
            )
        }
    }
}

/// A span's times, when the target shows them, and its text.
#[derive(Serialize)]
struct Span<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    start: Option<At>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end: Option<At>,
    text: &'a str,
}

impl<'a> Span<'a> {
    fn new(
        start: f64,
        end: f64,
        text: &'a str,
        opts: &Opts,
        cx: &Cx,
    ) -> Self {
        let at = |t| opts.times.then(|| At::new(&cx.clock, t));
        Span {
            start: at(start),
            end: at(end),
            text,
        }
    }
}

/// A statement's speaker, as a diarized record carries it.
#[derive(Serialize)]
struct Speaker {
    id: String,
    confidence: f64,
}

impl From<SpeakerLabel> for Speaker {
    fn from(label: SpeakerLabel) -> Self {
        Speaker {
            id: speaker_name(label.index),
            confidence: (label.confidence as f64 * 1000.0).round() / 1000.0,
        }
    }
}

/// A statement's body: its span alone, or, diarized, its span wrapped
/// under its speaker (`null` for nobody).
#[derive(Serialize)]
#[serde(untagged)]
enum Body<'a> {
    Plain(Span<'a>),
    Diarized {
        speaker: Option<Speaker>,
        statement: Span<'a>,
    },
}

/// One JSON line.
#[derive(Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum Record<'a> {
    Statement {
        #[serde(skip_serializing_if = "Option::is_none")]
        file: Option<&'a str>,
        #[serde(flatten)]
        body: Body<'a>,
    },
    Draft(Span<'a>),
    Transcript {
        file: &'a str,
        #[serde(flatten)]
        stamp: Option<&'a Stamp>,
        #[serde(flatten)]
        transcript: &'a Transcript,
    },
}

/// Whether `opts` shows speakers under `cx`.
fn speakers(
    opts: &Opts,
    cx: &Cx,
) -> bool {
    cx.diarized && opts.speakers
}

/// `event` as one JSON line, no newline.
pub fn json_line(
    event: &Event,
    opts: &Opts,
    cx: &Cx,
) -> String {
    let record = match *event {
        Event::Statement { file, statement } => {
            let span = Span::new(statement.start, statement.end, &statement.text, opts, cx);
            Record::Statement {
                file,
                body: if speakers(opts, cx) {
                    Body::Diarized {
                        speaker: statement.speaker.map(Speaker::from),
                        statement: span,
                    }
                } else {
                    Body::Plain(span)
                },
            }
        }
        Event::Draft { start, end, text } => Record::Draft(Span::new(start, end, text, opts, cx)),
        Event::Transcript {
            file,
            stamp,
            transcript,
        } => Record::Transcript {
            file,
            stamp,
            transcript,
        },
    };
    serde_json::to_string(&record).expect("a record serializes")
}

/// A statement as a text line: `[start --> end] ` when the target shows
/// times, `[speaker_k] ` (`[?]` for nobody) when it shows speakers, then the
/// text.
pub fn text_line(
    statement: &Statement,
    opts: &Opts,
    cx: &Cx,
) -> String {
    let mut line = String::new();
    if opts.times {
        line.push_str(&format!(
            "[{} --> {}] ",
            text_time(&cx.clock, statement.start),
            text_time(&cx.clock, statement.end)
        ));
    }
    if speakers(opts, cx) {
        let name = statement
            .speaker
            .map_or("?".to_string(), |label| speaker_name(label.index));
        line.push_str(&format!("[{name}] "));
    }
    line.push_str(&statement.text);
    line
}

/// A statement as an NIST RTTM `SPEAKER` line, the file's stem as its
/// recording id (`live` for the microphone), its times seconds from the
/// stream's start; `None` for nobody's.
pub fn rttm_line(
    file: Option<&str>,
    statement: &Statement,
) -> Option<String> {
    let label = statement.speaker?;
    Some(
        RttmSegment {
            file: file.map_or("live".to_string(), |f| recording_id(Path::new(f))),
            channel: 1,
            start: millis(statement.start),
            duration: millis(statement.end - statement.start),
            speaker: speaker_name(label.index),
        }
        .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use std::time::UNIX_EPOCH;

    use super::*;
    use crate::cli::asr::publish::config::EventKind;

    fn opts(
        times: bool,
        speakers: bool,
    ) -> Opts {
        Opts {
            times,
            speakers,
            events: EventKind::ALL.to_vec(),
        }
    }

    fn statement(speaker: Option<(usize, f32)>) -> Statement {
        Statement {
            start: 61.25,
            end: 63.5,
            text: "hello there".to_string(),
            speaker: speaker.map(|(index, confidence)| SpeakerLabel { index, confidence }),
        }
    }

    const OFFSET: Cx = Cx {
        clock: Clock::Offset,
        diarized: false,
    };

    const DIARIZED: Cx = Cx {
        clock: Clock::Offset,
        diarized: true,
    };

    #[test]
    fn test_text_time() {
        assert_eq!(text_time(&Clock::Offset, 3723.4567), "01:02:03.457");
        let clock = Clock::Wall {
            origin: UNIX_EPOCH + Duration::from_secs(1_761_228_000),
        };
        assert_eq!(text_time(&clock, 1.5), "14:00:01.500Z");
        assert!(matches!(At::new(&clock, 1.5), At::Wall(s) if s == "2025-10-23T14:00:01.500Z"));
    }

    /// Undiarized, a statement is flat; diarized, it is wrapped under its
    /// speaker, `null` for nobody; times only where the target shows them.
    #[test]
    fn test_json_line() {
        let st = statement(None);
        let event = Event::Statement {
            file: Some("a.wav"),
            statement: &st,
        };
        assert_eq!(
            json_line(&event, &opts(true, true), &OFFSET),
            r#"{"event":"statement","file":"a.wav","start":61.25,"end":63.5,"text":"hello there"}"#
        );
        assert_eq!(
            json_line(&event, &opts(true, true), &DIARIZED),
            r#"{"event":"statement","file":"a.wav","speaker":null,"statement":{"start":61.25,"end":63.5,"text":"hello there"}}"#
        );
        let st = statement(Some((1, 0.91234)));
        let event = Event::Statement {
            file: None,
            statement: &st,
        };
        assert_eq!(
            json_line(&event, &opts(false, true), &DIARIZED),
            r#"{"event":"statement","speaker":{"id":"speaker_1","confidence":0.912},"statement":{"text":"hello there"}}"#
        );
        // A target that hides speakers sees a plain statement.
        assert_eq!(
            json_line(&event, &opts(false, false), &DIARIZED),
            r#"{"event":"statement","text":"hello there"}"#
        );
        let draft = Event::Draft {
            start: 0.0,
            end: 1.0,
            text: "hel",
        };
        assert_eq!(
            json_line(&draft, &opts(false, true), &OFFSET),
            r#"{"event":"draft","text":"hel"}"#
        );
    }

    #[test]
    fn test_text_line() {
        let st = statement(Some((0, 0.9)));
        assert_eq!(
            text_line(&st, &opts(true, true), &DIARIZED),
            "[00:01:01.250 --> 00:01:03.500] [speaker_0] hello there"
        );
        assert_eq!(text_line(&st, &opts(false, true), &OFFSET), "hello there");
        assert_eq!(
            text_line(&statement(None), &opts(false, true), &DIARIZED),
            "[?] hello there"
        );
    }

    #[test]
    fn test_rttm_line() {
        let line = rttm_line(Some("dir/meeting.wav"), &statement(Some((2, 0.9)))).unwrap();
        assert!(line.starts_with("SPEAKER meeting 1 61.25"), "{line}");
        assert!(line.contains("speaker_2"), "{line}");
        assert_eq!(rttm_line(None, &statement(None)), None);
    }
}
