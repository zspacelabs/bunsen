//! # RTTM: speaker turns as NIST Rich Transcription Time Marked lines.

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::diarization::SpeakerTurn,
};

/// One `SPEAKER` line of an RTTM file: who spoke in which recording, from when
/// and for how long.
///
/// The line has NIST's ten fields:
/// `SPEAKER {file} {channel} {start} {duration} <NA> <NA> {speaker} <NA> <NA>`,
/// times in seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct RttmSegment {
    /// The recording's id.
    pub file: String,

    /// The channel, 1 for a mono recording.
    pub channel: u32,

    /// The start, in seconds.
    pub start: f64,

    /// The length, in seconds.
    pub duration: f64,

    /// The speaker's name.
    pub speaker: String,
}

impl RttmSegment {
    /// The end, in seconds.
    pub fn end(&self) -> f64 {
        self.start + self.duration
    }

    /// The line for `turn` in recording `file`, channel 1, its speaker named
    /// `speaker_{index}` as `NeMo` names them.
    pub fn from_turn(
        file: &str,
        turn: &SpeakerTurn,
    ) -> Self {
        Self {
            file: file.to_string(),
            channel: 1,
            start: turn.start,
            duration: turn.duration(),
            speaker: format!("speaker_{}", turn.speaker),
        }
    }

    /// Reads the `SPEAKER` lines of an RTTM file. Blank lines, `;;` comments,
    /// and lines of other types (`SPKR-INFO`, `LEXEME`, ...) are skipped.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
    /// naming the line, for a `SPEAKER` line with fewer than eight fields, or
    /// a channel, start or duration that does not parse.
    pub fn parse(text: &str) -> BunsenResult<Vec<Self>> {
        let mut segments = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(";;") {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields[0] != "SPEAKER" {
                continue;
            }
            let bad = |what: &str| {
                BunsenError::invalid_resource(format!("RTTM line {}: {what}: {line:?}", n + 1))
            };
            if fields.len() < 8 {
                return Err(bad("a SPEAKER line needs at least eight fields"));
            }
            segments.push(Self {
                file: fields[1].to_string(),
                channel: fields[2].parse().map_err(|_| bad("the channel"))?,
                start: fields[3].parse().map_err(|_| bad("the start"))?,
                duration: fields[4].parse().map_err(|_| bad("the duration"))?,
                speaker: fields[7].to_string(),
            });
        }
        Ok(segments)
    }

    /// The segments as RTTM text, one line each, times to the millisecond.
    pub fn write(segments: &[Self]) -> String {
        segments.iter().map(|s| format!("{s}\n")).collect()
    }
}

impl std::fmt::Display for RttmSegment {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "SPEAKER {} {} {:.3} {:.3} <NA> <NA> {} <NA> <NA>",
            self.file, self.channel, self.start, self.duration, self.speaker
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip() {
        let turns = [
            SpeakerTurn {
                speaker: 0,
                start: 0.299,
                end: 2.77,
            },
            SpeakerTurn {
                speaker: 1,
                start: 3.164,
                end: 5.147,
            },
        ];
        let segments: Vec<_> = turns
            .iter()
            .map(|t| RttmSegment::from_turn("an4_diarize_test", t))
            .collect();
        let text = RttmSegment::write(&segments);
        assert_eq!(
            text.lines().next().unwrap(),
            "SPEAKER an4_diarize_test 1 0.299 2.471 <NA> <NA> speaker_0 <NA> <NA>"
        );

        let back = RttmSegment::parse(&text).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].speaker, "speaker_1");
        assert!((back[1].end() - 5.147).abs() < 1e-9);
    }

    #[test]
    fn test_parse_skips_what_is_not_a_turn() {
        let text = ";; a comment\n\nSPKR-INFO f 1 <NA> <NA> <NA> unknown A <NA>\n\
                    SPEAKER f 1 1.5 0.5 <NA> <NA> A <NA> <NA>\n";
        let segments = RttmSegment::parse(text).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].speaker, "A");
        assert_eq!(segments[0].start, 1.5);
    }

    #[test]
    fn test_parse_refuses_a_bad_line() {
        assert!(RttmSegment::parse("SPEAKER f 1 x 0.5 <NA> <NA> A <NA> <NA>").is_err());
        assert!(RttmSegment::parse("SPEAKER f 1 1.0").is_err());
    }
}
