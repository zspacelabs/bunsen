//! A reply's text, one delta per token.
use std::sync::Arc;

use crate::{
    errors::BunsenResult,
    kits::tokens::Detokenizer,
};

/// A reply's text as its tokens arrive: each [`push`](Self::push) returns
/// what the new token adds.
///
/// A character whose bytes span two tokens renders, until its last byte
/// arrives, as a replacement character (`U+FFFD`) at the end of the text;
/// that tail is held back until the next token completes it, or
/// [`finish`](Self::finish) gives up on it. The deltas joined are the
/// detokenizer's rendering of every token pushed.
#[derive(Clone, Debug)]
pub struct TextStream {
    detokenizer: Arc<dyn Detokenizer>,
    ids: Vec<i64>,
    emitted: usize,
}

impl TextStream {
    /// A stream with nothing pushed.
    pub fn new(detokenizer: Arc<dyn Detokenizer>) -> Self {
        Self {
            detokenizer,
            ids: Vec::new(),
            emitted: 0,
        }
    }

    /// The tokens pushed.
    pub fn ids(&self) -> &[i64] {
        &self.ids
    }

    /// Adds a token, and returns the text it completes.
    ///
    /// # Errors
    /// As the detokenizer.
    pub fn push(
        &mut self,
        id: i64,
    ) -> BunsenResult<String> {
        self.ids.push(id);
        let text = self.detokenizer.detokenize(&self.ids)?;
        let ready = text.trim_end_matches(char::REPLACEMENT_CHARACTER).len();
        Ok(self.take(&text, ready))
    }

    /// The text still held back, now final.
    ///
    /// # Errors
    /// As the detokenizer.
    pub fn finish(&mut self) -> BunsenResult<String> {
        let text = self.detokenizer.detokenize(&self.ids)?;
        let len = text.len();
        Ok(self.take(&text, len))
    }

    /// `text[emitted..ready]`, now emitted.
    fn take(
        &mut self,
        text: &str,
        ready: usize,
    ) -> String {
        if ready <= self.emitted {
            return String::new();
        }
        let delta = text[self.emitted..ready].to_string();
        self.emitted = ready;
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ids are bytes.
    #[derive(Debug)]
    struct Bytes;

    impl Detokenizer for Bytes {
        fn detokenize(
            &self,
            ids: &[i64],
        ) -> BunsenResult<String> {
            let bytes: Vec<u8> = ids.iter().map(|&b| b as u8).collect();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }

    #[test]
    fn test_deltas_never_split_a_character() {
        let mut stream = TextStream::new(Arc::new(Bytes));
        let mut deltas = Vec::new();
        // "a", then "é" in two bytes, then "b".
        for id in [0x61, 0xC3, 0xA9, 0x62] {
            deltas.push(stream.push(id).unwrap());
        }
        assert_eq!(deltas, ["a", "", "é", "b"]);
        assert_eq!(stream.finish().unwrap(), "");
        assert_eq!(stream.ids(), [0x61, 0xC3, 0xA9, 0x62]);
    }

    #[test]
    fn test_finish_gives_up_on_a_broken_tail() {
        let mut stream = TextStream::new(Arc::new(Bytes));
        assert_eq!(stream.push(0x61).unwrap(), "a");
        assert_eq!(stream.push(0xC3).unwrap(), "");
        assert_eq!(stream.finish().unwrap(), "\u{FFFD}");
    }
}
