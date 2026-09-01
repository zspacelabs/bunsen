//! The `wordchipper`-backed [`Detokenizer`].
//!
//! This is the only place in the crate that names `wordchipper`. It is also
//! where `i64` ids narrow to the vocabulary's token type and where `WCError`
//! becomes `BunsenError`; neither type reaches a kit's public API.
//!
//! [`WordchipperDetokenizer::from_spans`] is the decode-only path: a
//! `{ id -> bytes }` table and a concatenation, which is `TokenDictDecoder`
//! with no encoder, no merge table, no regex and no `UnifiedTokenVocab`
//! behind it. That is deliberate twice over. It is the cheap path — nothing
//! derives a BPE pair table for a decoder that never reads one. And it is
//! the only path that can hold an **empty token**, which Whisper's
//! multilingual vocabulary has at rank 50256: `SlabIndexDecoder` stores
//! `(start, end)` offsets and reads `end == start` as *absent*, which would
//! truncate every decode that crosses that id, and `UnifiedTokenVocab`
//! rejects a token that is neither a byte nor a merge. `TokenDictDecoder` is
//! a map lookup, and returns the empty span as what it is.

use std::{
    fmt::{
        self,
        Debug,
    },
    sync::Arc,
};

use wordchipper::{
    TokenDecoder,
    TokenType,
    decoders::TokenDictDecoder,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::tokens::Detokenizer,
};

/// A [`Detokenizer`] over any `wordchipper` decoder.
///
/// `T` is the vocabulary's token type — `u16` for Whisper's 51866 ids, `u32`
/// for the larger GPT vocabularies. Built once and shared through an `Arc`;
/// it is `Send + Sync`.
pub struct WordchipperDetokenizer<T: TokenType> {
    decoder: Arc<dyn TokenDecoder<T>>,
    vocab_size: usize,
}

impl<T: TokenType> WordchipperDetokenizer<T> {
    /// Wraps an existing decoder.
    ///
    /// # Arguments
    /// * `decoder` - any `wordchipper` decoder over `T`.
    /// * `vocab_size` - one past the largest id it renders. Ids at or above it
    ///   are rejected before the decoder sees them, with a message that names
    ///   the id.
    pub fn new(
        decoder: Arc<dyn TokenDecoder<T>>,
        vocab_size: usize,
    ) -> Self {
        Self {
            decoder,
            vocab_size,
        }
    }

    /// Builds the decode-only path over a `{ id -> bytes }` table.
    ///
    /// Ids need not be contiguous; an id missing from the table is outside
    /// the vocabulary. An empty span is a real token that renders as nothing,
    /// not a gap.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if an id does not fit `T`.
    pub fn from_spans(spans: impl IntoIterator<Item = (usize, Vec<u8>)>) -> BunsenResult<Self> {
        let mut table = wordchipper::hash_map_new::<T, Vec<u8>>();
        let mut vocab_size = 0;

        for (id, bytes) in spans {
            let token = T::from_usize(id).ok_or_else(|| {
                BunsenError::Invalid(format!(
                    "token id {id} does not fit {}",
                    std::any::type_name::<T>(),
                ))
            })?;
            table.insert(token, bytes);
            vocab_size = vocab_size.max(id + 1);
        }

        Ok(Self::new(
            Arc::new(TokenDictDecoder::new(table)),
            vocab_size,
        ))
    }

    /// One past the largest id this renders.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// By hand: `dyn TokenDecoder` is not `Debug`, and a 50k-entry table would
// not be a useful one anyway.
impl<T: TokenType> Debug for WordchipperDetokenizer<T> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("WordchipperDetokenizer")
            .field("token_type", &std::any::type_name::<T>())
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl<T: TokenType> Detokenizer for WordchipperDetokenizer<T> {
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String> {
        let narrowed = ids
            .iter()
            .map(|&id| {
                T::from_i64(id)
                    .filter(|t| t.to_usize().is_some_and(|t| t < self.vocab_size))
                    .ok_or_else(|| {
                        BunsenError::Invalid(format!(
                            "token id {id} is outside the {}-id vocabulary",
                            self.vocab_size,
                        ))
                    })
            })
            .collect::<BunsenResult<Vec<T>>>()?;

        // An id inside the range but missing from the table stops the decode
        // short; going through `try_result` rather than `.value` turns that
        // into an error instead of a silent truncation.
        self.decoder
            .try_decode_to_string(&narrowed)
            .map_err(BunsenError::external)?
            .try_result()
            .map_err(BunsenError::external)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Five ids, including an empty one and a gap: the party popper's four
    /// bytes split across ids 1 and 2.
    fn tiny<T: TokenType>() -> WordchipperDetokenizer<T> {
        WordchipperDetokenizer::from_spans([
            (0, b"ok".to_vec()),
            (1, b" \xf0\x9f\x8e".to_vec()),
            (2, b"\x89".to_vec()),
            (3, b" done".to_vec()),
            (4, Vec::new()),
            (6, b"<|special|>".to_vec()),
        ])
        .unwrap()
    }

    #[test]
    fn test_detokenizes() {
        let detok = tiny::<u16>();

        assert_eq!(detok.vocab_size(), 7);
        assert_eq!(detok.detokenize(&[]).unwrap(), "");
        assert_eq!(detok.detokenize(&[0, 3]).unwrap(), "ok done");
        assert_eq!(detok.detokenize(&[6]).unwrap(), "<|special|>");
    }

    /// A character split across two ids survives, because the bytes are
    /// joined before the UTF-8 decode.
    #[test]
    fn test_joins_bytes_before_utf8() {
        let detok = tiny::<u16>();

        assert_eq!(detok.detokenize(&[0, 1, 2, 3]).unwrap(), "ok 🎉 done");

        // Half a character alone is lossy, as it must be — but it is not an
        // error, and it is not dropped.
        assert_eq!(detok.detokenize(&[1]).unwrap(), " \u{fffd}");
    }

    /// The empty token renders as nothing and does not stop the decode —
    /// the property that rules out `SlabIndexDecoder`.
    #[test]
    fn test_empty_token_is_not_a_gap() {
        let detok = tiny::<u16>();

        assert_eq!(detok.detokenize(&[0, 4, 3]).unwrap(), "ok done");
        assert_eq!(detok.detokenize(&[4]).unwrap(), "");
    }

    #[test]
    fn test_rejects_ids_outside_the_vocabulary() {
        let detok = tiny::<u16>();

        for bad in [-1, 7, i64::from(u16::MAX) + 1, i64::MAX, i64::MIN] {
            let err = detok.detokenize(&[0, bad, 3]).expect_err("out of range");
            assert!(matches!(err, BunsenError::Invalid(_)), "{bad}: {err:?}");
            assert!(err.to_string().contains(&bad.to_string()), "{err}");
        }

        // Inside the range but not in the table: the decode stops, and that
        // is reported rather than swallowed.
        let err = detok.detokenize(&[0, 5, 3]).expect_err("a gap");
        assert!(matches!(err, BunsenError::External(_)), "{err:?}");
    }

    #[test]
    fn test_generic_over_token_type() {
        assert_eq!(
            tiny::<u32>().detokenize(&[0, 1, 2, 3]).unwrap(),
            "ok 🎉 done"
        );

        // An id that does not fit the token type is refused at build time.
        let err = WordchipperDetokenizer::<u16>::from_spans([(70_000, b"x".to_vec())]).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)), "{err:?}");
    }

    #[test]
    fn test_debug_and_trait_object() {
        let detok: Arc<dyn Detokenizer> = Arc::new(tiny::<u16>());

        let shown = format!("{detok:?}");
        assert!(shown.starts_with("WordchipperDetokenizer"), "{shown}");
        assert!(shown.contains("token_type: \"u16\""), "{shown}");
        assert!(shown.contains("vocab_size: 7"), "{shown}");
        assert_eq!(detok.detokenize(&[0]).unwrap(), "ok");
    }
}
