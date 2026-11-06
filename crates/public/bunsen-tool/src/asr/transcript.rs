//! A transcript as the outputs carry it, whichever family made it.
use std::path::Path;

use bunsen::kits::speech::asr::driver::TranscriptSegment;
use serde::Serialize;

/// A transcript, whichever family made it.
#[derive(Clone, Debug, Serialize)]
pub struct Transcript {
    /// The text.
    pub text: String,

    /// The language the model named, if it names one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Each word with its start and end in seconds, if the model times them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,

    /// Each segment with its start and end in seconds, and its token ids,
    /// if the model emits segments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,

    /// The speaker turns, under `--diarize`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turns: Option<Vec<Turn>>,
}

/// A timed word.
#[derive(Clone, Debug, Serialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    /// The token ids the model decoded it as.
    pub tokens: Vec<i64>,
    /// Its speaker, under `--diarize`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// A timed segment.
#[derive(Clone, Debug, Serialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// The token ids the model decoded it as.
    pub tokens: Vec<i64>,
    /// Its speaker, under `--diarize`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
}

/// One speaker's turn: a run of consecutive words or segments attributed to
/// them.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Turn {
    /// The speaker, `speaker_{k}`; `null` where the diarizer heard nobody.
    pub speaker: Option<String>,
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// `segment`'s speaker, as the output names it.
pub fn segment_speaker(segment: &TranscriptSegment) -> Option<String> {
    segment.speaker.map(|label| speaker_name(label.index))
}

/// The turns of `segments`: runs of one speaker, their texts joined; the
/// segments with no text are left out.
pub fn speaker_turns(segments: &[TranscriptSegment]) -> Vec<Turn> {
    let mut turns: Vec<Turn> = Vec::new();
    for segment in segments {
        let Some(text) = segment
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        else {
            continue;
        };
        let speaker = segment_speaker(segment);
        match turns.last_mut() {
            Some(turn) if turn.speaker == speaker => {
                turn.end = millis(segment.end);
                turn.text.push(' ');
                turn.text.push_str(text);
            }
            _ => turns.push(Turn {
                speaker,
                start: millis(segment.start),
                end: millis(segment.end),
                text: text.to_string(),
            }),
        }
    }
    turns
}

/// A speaker's name, as `NeMo` and the RTTM output name it.
pub fn speaker_name(index: usize) -> String {
    format!("speaker_{index}")
}

/// The recording id `--rttm` gives `path`: its stem.
pub fn recording_id(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Seconds, rounded to the millisecond.
pub fn millis(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use bunsen::kits::speech::asr::driver::SpeakerLabel;

    use super::*;

    fn segment(
        text: &str,
        start: f64,
        speaker: Option<usize>,
    ) -> TranscriptSegment {
        TranscriptSegment {
            start,
            end: start + 0.5,
            tokens: Vec::new(),
            text: Some(text.to_string()),
            speaker: speaker.map(|index| SpeakerLabel {
                index,
                confidence: 0.9,
            }),
        }
    }

    /// Consecutive segments of one speaker are one turn; empty ones are left
    /// out, and a change of speaker, nobody included, starts a turn.
    #[test]
    fn test_speaker_turns() {
        let turns = speaker_turns(&[
            segment(" hello", 0.0, Some(0)),
            segment("  ", 0.5, Some(1)),
            segment("there", 1.0, Some(0)),
            segment("hi", 2.0, Some(1)),
            segment("um", 3.0, None),
        ]);
        let lines: Vec<(Option<&str>, &str)> = turns
            .iter()
            .map(|t| (t.speaker.as_deref(), t.text.as_str()))
            .collect();
        assert_eq!(
            lines,
            [
                (Some("speaker_0"), "hello there"),
                (Some("speaker_1"), "hi"),
                (None, "um")
            ]
        );
        assert_eq!((turns[0].start, turns[0].end), (0.0, 1.5));
    }
}
