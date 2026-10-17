//! Timed words of a decode: where a full window is cut, and each word as a
//! segment.

use crate::kits::speech::asr_driver::TimedTokens;

/// A word of a decode: its first token, and the frames it spans in the
/// window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordSpan {
    /// The index of the word's first token in the decode's tokens.
    pub first_token: usize,

    /// The frame its first token is emitted on.
    pub start: usize,

    /// One past the frame its last token ends on; it may reach past the next
    /// word's start.
    pub end: usize,
}

/// Where a window cut by length is committed to, and the first word past
/// the cut.
///
/// The window may end inside a word, and a window opening on a word's onset
/// loses the word, so the cut goes in the widest pause between two words
/// whose later word starts in the window's last third: midway between where
/// the earlier word ends and the later word starts, on a multiple of `grid`
/// from that end. Of equally wide pauses, the latest. Without a pause there,
/// the cut is where the word before the last word ends.
///
/// # Arguments
/// * `words` - the decode's words, in order.
/// * `frames` - the window's frames.
/// * `grid` - the frames a word's start and end fall on multiples of.
///
/// # Returns
/// The cut, at most `frames`, and the index into `words` of the first word
/// past it; `None` without two words or when the cut would be at frame 0.
pub fn pause_cut(
    words: &[WordSpan],
    frames: usize,
    grid: usize,
) -> Option<(usize, usize)> {
    let late = frames * 2 / 3;
    let pause = |i: usize| {
        let end = words[i - 1].end;
        (words[i].start.saturating_sub(end), end)
    };
    // `max_by_key` keeps the last of equals: the latest widest pause.
    let (i, mid) = (1..words.len())
        .map(|i| {
            let (gap, end) = pause(i);
            (i, gap, end + gap / grid / 2 * grid)
        })
        .filter(|&(i, gap, _)| gap > 0 && words[i].start >= late)
        .max_by_key(|&(_, gap, _)| gap)
        .map(|(i, _, mid)| (i, mid))
        .or_else(|| {
            let i = words.len().checked_sub(1).filter(|&i| i > 0)?;
            Some((i, pause(i).1))
        })?;
    let cut = mid.min(frames);
    (cut > 0).then_some((cut, i))
}

/// The words as segments, each with its tokens: from its start to its end,
/// but no later than the next word's start or `limit`.
///
/// # Arguments
/// * `words` - the words, in order; word `i`'s tokens run from its first to the
///   next word's first, the last word's to the end of `tokens`.
/// * `tokens` - the tokens the words cover.
/// * `limit` - the frame no segment reaches past.
pub fn word_segments(
    words: &[WordSpan],
    tokens: &[i64],
    limit: usize,
) -> Vec<TimedTokens> {
    words
        .iter()
        .enumerate()
        .map(|(i, word)| {
            let next = words.get(i + 1);
            let start = word.start.min(limit);
            let bound = next.map_or(limit, |n| n.start.min(limit));
            TimedTokens {
                start,
                end: word.end.min(bound).max(start),
                tokens: tokens[word.first_token..next.map_or(tokens.len(), |n| n.first_token)]
                    .to_vec(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(
        first_token: usize,
        start: usize,
        end: usize,
    ) -> WordSpan {
        WordSpan {
            first_token,
            start,
            end,
        }
    }

    #[test]
    fn test_pause_cut() {
        // The widest pause whose later word starts in the last third (from
        // 66): 56 to 144, cut on the grid of 8 midway, at 96.
        let words = [
            span(0, 16, 24),
            span(1, 40, 56),
            span(3, 144, 152),
            span(4, 160, 168),
        ];
        assert_eq!(pause_cut(&words, 100 * 2, 8), Some((96, 2)));
        // On a grid of 1, exactly midway.
        assert_eq!(pause_cut(&words, 200, 1), Some((100, 2)));
        // Equal pauses: the latest.
        let even = [
            span(0, 0, 140),
            span(1, 150, 160),
            span(2, 170, 180),
            span(3, 190, 195),
        ];
        assert_eq!(pause_cut(&even, 200, 1), Some((185, 3)));
        // No pause late: where the word before the last ends.
        let packed = [span(0, 0, 50), span(1, 50, 120), span(2, 120, 190)];
        assert_eq!(pause_cut(&packed, 200, 1), Some((120, 2)));
        // One word, or a cut at 0: none.
        assert_eq!(pause_cut(&[span(0, 0, 50)], 200, 1), None);
        assert_eq!(pause_cut(&[span(0, 0, 0), span(1, 0, 4)], 200, 1), None);
    }

    #[test]
    fn test_word_segments() {
        let words = [span(0, 16, 48), span(2, 40, 56), span(3, 60, 90)];
        let tokens = [7, 8, 9, 10, 11];
        assert_eq!(
            word_segments(&words, &tokens, 80),
            [
                TimedTokens {
                    start: 16,
                    end: 40,
                    tokens: vec![7, 8],
                },
                TimedTokens {
                    start: 40,
                    end: 56,
                    tokens: vec![9],
                },
                TimedTokens {
                    start: 60,
                    end: 80,
                    tokens: vec![10, 11],
                },
            ]
        );
        assert!(word_segments(&[], &[], 10).is_empty());
    }
}
