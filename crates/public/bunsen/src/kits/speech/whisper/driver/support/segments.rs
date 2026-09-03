//! # Segments: what a timestamped decode says about time.
//!
//! A decode prompted for timestamps returns text bracketed by timestamp
//! tokens. Upstream's seek loop turns that into segments and a seek
//! advance, and this is that logic as a pure function over the ids:
//!
//! - **Consecutive timestamps** (an end followed by the next start) split the
//!   sequence into segments, each running from its opening timestamp to its
//!   closing one. A transcript ending in a lone timestamp closes its last
//!   segment there, and says there is no speech after it, so the seek advances
//!   a whole window; otherwise the unfinished trailing segment is dropped and
//!   the seek advances to the last closed timestamp, to be decoded again with
//!   more audio behind it.
//! - **No consecutive timestamps** means the window is one segment: from its
//!   start to its last timestamp if it has one, else to its end; the seek
//!   advances a whole window.
//!
//! Positions are in mel frames relative to the decoded unit's start; the
//! caller puts them on its clock. Upstream clears a segment that is
//! instantaneous or has no text; that is dropped here, since a segment that
//! carries nothing has nothing to emit and nothing to carry as a prompt.

use crate::kits::speech::whisper::driver::whisper_token_layout::WhisperSpecialIds;

/// A segment of a decoded unit, in frames relative to the unit's start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TimedTokens {
    /// The first frame the segment covers.
    pub start: usize,

    /// The frame the segment ends at; may exceed the unit's frames when the
    /// model says so.
    pub end: usize,

    /// The ids, timestamp tokens included.
    pub tokens: Vec<i64>,
}

/// What one decoded unit splits into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowSplit {
    /// The closed segments, in order.
    pub segments: Vec<TimedTokens>,

    /// How many frames the seek pointer advances.
    pub advance: usize,

    /// The unfinished trailing segment, which the next decode redoes;
    /// empty when the transcript closed cleanly.
    pub tail: Vec<i64>,
}

/// Splits a timestamped decode of one unit into segments, as upstream's
/// seek loop does.
///
/// # Arguments
/// * `tokens` - the decoded ids after the prompt, stop token excluded.
/// * `ids` - the token layout.
/// * `count` - the frames of audio the unit held (a whole window, or the
///   remainder at the end of a stream).
/// * `frames_per_index` - mel frames per timestamp index: two.
pub(crate) fn split_window(
    tokens: &[i64],
    ids: &WhisperSpecialIds,
    count: usize,
    frames_per_index: usize,
) -> WindowSplit {
    let tb = ids.timestamp_begin;
    let is_ts = |t: i64| t >= tb;
    let frames = |t: i64| (t - tb) as usize * frames_per_index;
    let n = tokens.len();

    let single_timestamp_ending = n >= 2 && !is_ts(tokens[n - 2]) && is_ts(tokens[n - 1]);
    let consecutive: Vec<usize> = (0..n.saturating_sub(1))
        .filter(|&i| is_ts(tokens[i]) && is_ts(tokens[i + 1]))
        .map(|i| i + 1)
        .collect();

    let (mut segments, advance, tail) = if !consecutive.is_empty() {
        let mut slices = consecutive;
        if single_timestamp_ending {
            slices.push(n);
        }

        let mut segments = Vec::with_capacity(slices.len());
        let mut last = 0;
        for current in slices {
            let sliced = &tokens[last..current];
            segments.push(TimedTokens {
                start: frames(sliced[0]),
                end: frames(sliced[sliced.len() - 1]),
                tokens: sliced.to_vec(),
            });
            last = current;
        }

        let advance = if single_timestamp_ending {
            count
        } else {
            frames(tokens[last - 1])
        };
        (segments, advance, tokens[last..].to_vec())
    } else {
        let mut end = count;
        if let Some(&last) = tokens.iter().rev().find(|&&t| is_ts(t))
            && last != tb
        {
            end = frames(last);
        }
        (
            vec![TimedTokens {
                start: 0,
                end,
                tokens: tokens.to_vec(),
            }],
            count,
            Vec::new(),
        )
    };

    segments.retain(|s| s.start != s.end && s.tokens.iter().any(|&t| t < ids.eot));

    WindowSplit {
        segments,
        advance,
        tail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// eot 5, timestamps from 14.
    fn ids() -> WhisperSpecialIds {
        WhisperSpecialIds::new(5, 1).unwrap()
    }

    fn ts(i: i64) -> i64 {
        ids().timestamp_begin + i
    }

    /// Two closed segments and an unfinished third: the third is dropped
    /// and the seek goes to the second's close.
    #[test]
    fn test_consecutive_pairs_split_and_seek_to_last_close() {
        let ids = ids();
        let tokens = vec![ts(0), 1, 2, ts(3), ts(3), 4, ts(6), ts(7), 1, 2];
        let split = split_window(&tokens, &ids, 100, 2);
        assert_eq!(
            split.segments,
            vec![
                TimedTokens {
                    start: 0,
                    end: 6,
                    tokens: vec![ts(0), 1, 2, ts(3)],
                },
                TimedTokens {
                    start: 6,
                    end: 12,
                    tokens: vec![ts(3), 4, ts(6)],
                },
            ]
        );
        assert_eq!(split.advance, 12, "to the last closed timestamp, in frames");
        assert_eq!(split.tail, vec![ts(7), 1, 2]);
    }

    /// A lone timestamp at the end closes the last segment and says the
    /// rest of the window is silence: advance the whole unit.
    #[test]
    fn test_single_timestamp_ending_takes_the_whole_window() {
        let ids = ids();
        let tokens = vec![ts(0), 1, ts(2), ts(2), 3, ts(5)];
        let split = split_window(&tokens, &ids, 100, 2);
        assert_eq!(split.segments.len(), 2);
        assert_eq!(split.segments[1].tokens, vec![ts(2), 3, ts(5)]);
        assert_eq!((split.segments[1].start, split.segments[1].end), (4, 10));
        assert_eq!(split.advance, 100);
        assert!(split.tail.is_empty());
    }

    /// No consecutive pair: the window is one segment, to its last
    /// timestamp when it has one.
    #[test]
    fn test_no_pairs_is_one_segment() {
        let ids = ids();

        // Opened, some text, closed, and nothing after: one segment to the
        // close. Upstream's `single_timestamp_ending` with no consecutive
        // pair lands here too.
        let split = split_window(&[ts(1), 2, 3, ts(4)], &ids, 100, 2);
        assert_eq!(
            split.segments,
            vec![TimedTokens {
                start: 0,
                end: 8,
                tokens: vec![ts(1), 2, 3, ts(4)],
            }]
        );
        assert_eq!(split.advance, 100);

        // No timestamps at all: the whole unit.
        let split = split_window(&[2, 3], &ids, 40, 2);
        assert_eq!((split.segments[0].start, split.segments[0].end), (0, 40));
        assert_eq!(split.advance, 40);

        // A last timestamp of <|0.00|> does not shorten anything.
        let split = split_window(&[ts(0), 2, 3, ts(0)], &ids, 40, 2);
        assert_eq!(split.segments[0].end, 40);
    }

    /// Nothing decoded, or nothing but timestamps: no segment, the seek
    /// still moves.
    #[test]
    fn test_empty_and_textless_are_dropped() {
        let ids = ids();
        let split = split_window(&[], &ids, 40, 2);
        assert!(split.segments.is_empty());
        assert_eq!(split.advance, 40);

        let split = split_window(&[ts(0), ts(3), ts(3), 4, ts(5), ts(6)], &ids, 40, 2);
        assert_eq!(
            split.segments.len(),
            1,
            "the textless first pair is dropped"
        );
        assert_eq!(split.segments[0].tokens, vec![ts(3), 4, ts(5)]);
        assert_eq!(split.advance, 10);
        assert_eq!(split.tail, vec![ts(6)]);

        // An instantaneous segment goes too.
        let split = split_window(&[ts(2), 4, ts(2), ts(2), 4, ts(5), ts(5)], &ids, 40, 2);
        assert_eq!(split.segments.len(), 1);
        assert_eq!(split.segments[0].start, 4);
    }

    /// Upstream's own answer for the first fixed window of the reference
    /// clip decoded with timestamps: the ids from
    /// `jfk_moon.reference.json`, ending `<|18.96|><|25.36|>`, where the
    /// final lone start is the tail and the seek lands on 18.96 s: frame
    /// 1896, which is where `transcribe()` put its second window.
    #[test]
    fn test_reference_window_shape() {
        let ids = WhisperSpecialIds::from_vocab_size(51865).unwrap();
        let tb = ids.timestamp_begin;
        // Three closed segments and a reopened fourth.
        let tokens = vec![
            tb + 0,
            264,
            7135,
            13,
            tb + 388,
            tb + 388,
            1545,
            tb + 440,
            tb + 440,
            492,
            tb + 948,
            tb + 1268,
        ];
        let split = split_window(&tokens, &ids, 3000, 2);
        assert_eq!(split.segments.len(), 3);
        assert_eq!((split.segments[0].start, split.segments[0].end), (0, 776));
        assert_eq!((split.segments[1].start, split.segments[1].end), (776, 880));
        assert_eq!(
            (split.segments[2].start, split.segments[2].end),
            (880, 1896)
        );
        assert_eq!(split.advance, 1896);
        assert_eq!(split.tail, vec![tb + 1268]);
    }
}
