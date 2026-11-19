//! Long text into takes.

use alloc::{
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

/// Ends a sentence when whitespace or the text's end follows it.
const SENTENCE_ENDS: [char; 4] = ['.', '!', '?', '…'];

/// Ends a sentence on its own (full-width forms, which no space follows).
const WIDE_SENTENCE_ENDS: [char; 3] = ['。', '！', '？'];

/// Ends a clause when whitespace follows it.
const CLAUSE_ENDS: [char; 4] = [',', ';', ':', '—'];

/// Ends a clause on its own.
const WIDE_CLAUSE_ENDS: [char; 3] = ['，', '；', '、'];

/// `text` as takes of at most `max_chars` characters each: trimmed and whole
/// when it fits, else split where speech pauses: sentences packed whole into a
/// take while they fit; a sentence longer than a take, at its clauses; a clause
/// longer than that, at its words; a word longer than that, cut. Line breaks
/// end a sentence. Pieces join with a space, except after a full-width end.
/// Each take is trimmed, and none is empty.
///
/// A sentence end is `.`, `!`, `?` or `…` before whitespace, or a full-width
/// `。`, `！` or `？`. An abbreviation's period ends a sentence too; that
/// changes where a take may end, never what is spoken.
///
/// # Panics
/// For a `max_chars` of 0.
pub fn tts_split_text(
    text: &str,
    max_chars: usize,
) -> Vec<String> {
    assert!(max_chars > 0, "a take holds at least one character");
    let whole = text.trim();
    if whole.is_empty() {
        return Vec::new();
    }
    if whole.chars().count() <= max_chars {
        return Vec::from([whole.to_string()]);
    }
    let mut takes = Vec::new();
    let mut take = String::new();
    for sentence in pieces(text, &SENTENCE_ENDS, &WIDE_SENTENCE_ENDS, true) {
        for part in fit(sentence, max_chars) {
            push(&mut takes, &mut take, &part, max_chars);
        }
    }
    if !take.is_empty() {
        takes.push(take);
    }
    takes
}

/// `unit` onto `current`, or onto a fresh one when it would not fit,
/// `current` then finished into `done`. Units join with a space, but after a
/// full-width end, which takes none.
fn push(
    done: &mut Vec<String>,
    current: &mut String,
    unit: &str,
    max_chars: usize,
) {
    let space = !current.is_empty()
        && !current.ends_with(|c| WIDE_SENTENCE_ENDS.contains(&c) || WIDE_CLAUSE_ENDS.contains(&c));
    let joined = current.chars().count() + usize::from(space) + unit.chars().count();
    if !current.is_empty() && joined > max_chars {
        done.push(core::mem::take(current));
    } else if space {
        current.push(' ');
    }
    current.push_str(unit);
}

/// `sentence` as parts of at most `max_chars`: whole, or at its clauses
/// packed, or at its words packed, or cut.
fn fit(
    sentence: &str,
    max_chars: usize,
) -> Vec<String> {
    if sentence.chars().count() <= max_chars {
        return Vec::from([sentence.to_string()]);
    }
    let clauses = pieces(sentence, &CLAUSE_ENDS, &WIDE_CLAUSE_ENDS, false);
    let units: Vec<String> = if clauses.len() > 1 {
        clauses
            .into_iter()
            .flat_map(|clause| fit(clause, max_chars))
            .collect()
    } else {
        sentence
            .split_whitespace()
            .flat_map(|word| cut(word, max_chars))
            .collect()
    };
    pack(units, max_chars)
}

/// `units` joined (as [`push`] joins them) into parts of at most `max_chars`,
/// in order.
fn pack(
    units: Vec<String>,
    max_chars: usize,
) -> Vec<String> {
    let mut parts = Vec::new();
    let mut part = String::new();
    for unit in units {
        push(&mut parts, &mut part, &unit, max_chars);
    }
    if !part.is_empty() {
        parts.push(part);
    }
    parts
}

/// `word` cut into pieces of at most `max_chars`.
fn cut(
    word: &str,
    max_chars: usize,
) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    chars
        .chunks(max_chars)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

/// `text` split after each end in `ends` that whitespace or the text's end
/// follows, and after each in `wide`; with `lines`, at line breaks too.
/// Each piece is trimmed, and none is empty.
fn pieces<'a>(
    text: &'a str,
    ends: &[char],
    wide: &[char],
    lines: bool,
) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let next = chars.peek().map(|&(_, n)| n);
        let after = i + c.len_utf8();
        let boundary = (lines && c == '\n')
            || wide.contains(&c)
            || (ends.contains(&c) && next.is_none_or(char::is_whitespace));
        if boundary {
            let piece = text[start..after].trim();
            if !piece.is_empty() {
                out.push(piece);
            }
            start = after;
        }
    }
    let piece = text[start..].trim();
    if !piece.is_empty() {
        out.push(piece);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(
        text: &str,
        max: usize,
    ) -> Vec<String> {
        let takes = tts_split_text(text, max);
        for take in &takes {
            assert!(take.chars().count() <= max, "{take:?} over {max}");
            assert!(!take.is_empty() && take.trim() == take, "{take:?}");
        }
        takes
    }

    /// Text that fits is one take, trimmed and otherwise as it is; none is
    /// no takes.
    #[test]
    fn test_short_text() {
        assert_eq!(split("  Hello there.  ", 300), ["Hello there."]);
        assert_eq!(split("Hello\nthere.", 300), ["Hello\nthere."]);
        assert!(split(" \n ", 300).is_empty());
    }

    /// Sentences pack whole into takes while they fit; a line break ends a
    /// sentence; a period inside a number or before no space is no end.
    #[test]
    fn test_sentences_pack() {
        assert_eq!(
            split("One two. Three four! Five six? Seven.", 20),
            ["One two. Three four!", "Five six? Seven."]
        );
        assert_eq!(split("Line one\nLine two", 9), ["Line one", "Line two"]);
        assert_eq!(split("It costs 3.50 now.", 18), ["It costs 3.50 now."]);
    }

    /// A sentence over a take splits at its clauses, then its words, then
    /// within a word.
    #[test]
    fn test_long_sentences() {
        assert_eq!(
            split("First clause, second clause; third.", 15),
            ["First clause,", "second clause;", "third."]
        );
        assert_eq!(split("aaa bbb ccc ddd", 7), ["aaa bbb", "ccc ddd"]);
        assert_eq!(split("abcdefghij", 4), ["abcd", "efgh", "ij"]);
    }

    /// Full-width ends split with no space after them; characters, not bytes,
    /// are counted.
    #[test]
    fn test_wide_text() {
        assert_eq!(split("你好。世界！", 3), ["你好。", "世界！"]);
        assert_eq!(split("你好。世界！", 6), ["你好。世界！"]);
        assert_eq!(split("一，二，三", 2), ["一，", "二，", "三"]);
    }
}
