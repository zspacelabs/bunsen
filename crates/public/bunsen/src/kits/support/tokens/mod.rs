//! # Tokens
//!
//! The token side of a model kit: what is shared between kits that emit ids
//! and want text back, and kits that build prompts from text. Two seams,
//! [`Detokenizer`] (ids to text) and [`Tokenizer`] (text to ids, and back);
//! an implementation of each, behind the `tokenizer` feature:
//! `WordchipperDetokenizer`, decode-only, and `BunsenTokenizer`, both ways;
//! and the parser of the `.tiktoken` rank files a
//! byte-level vocabulary ships as, [`TiktokenRanks`].
//!
//! A model kit owns its own token *layout* — which ids are special, what they
//! mean, how a prompt is built — because that is model-specific and needs no
//! dependency; Whisper's is [`WhisperTokenLayout`]. What it does not own is
//! the tokenizer. Ids-to-text is the same operation for every byte-level
//! vocabulary, and the crate that does it well is `wordchipper`, so that
//! lives here once and a kit hands it a table: Whisper's layout spells its
//! specials after the base ranks and hands the whole `{ id -> bytes }`
//! table to `WordchipperDetokenizer::from_spans`.
//!
//! Encoding — the merge table, the pre-tokenizer, the special-token
//! splitter — is a tokenizer library's whole job, and `BunsenTokenizer`
//! leaves it to the `tokenizers` crate, `transformers`' own encoder, so it
//! encodes as `transformers` does by construction. It reads a
//! `tokenizer.json`, with the added tokens of a `tokenizer_config.json` as
//! `transformers` loads them; or a pickled `tiktoken` `Encoding` (nanochat's
//! `tokenizer.pkl`, through `serde_pickle`), as the `tokenizer.json`
//! `transformers`' tiktoken converter writes for it. A kit that builds
//! prompts takes any [`Tokenizer`].
//!
//! `testing` (feature `testing`) has toy tokenizer files, in the shapes real
//! checkpoints ship, for tests of code that reads text through one.
//!
//! [`ByteLevelVocabulary`] reads the other common spelling of a byte-level
//! vocabulary, `transformers`' `vocab.json` / `tokenizer.json` (GPT-2's, and
//! Qwen's), into the same `{ id -> bytes }` table, and
//! [`TekkenVocabulary`] reads Mistral's, `tekken.json`.
//!
//! # Whisper-specific helpers
//!
//! [`SYMBOLS`], [`MISCELLANEOUS`], [`TiktokenRanks::non_speech_tokens`] and
//! [`TiktokenRanks::blank_token`] are not shared: they compute Whisper's
//! default suppress list and blank token from its rank file, and only the
//! Whisper kit's logit filters use them. They live here beside the rank parser
//! they read.
//!
//! [`WhisperTokenLayout`]: crate::kits::speech::asr::whisper::driver::WhisperTokenLayout

use core::fmt::Debug;

use crate::errors::BunsenResult;

#[cfg(feature = "tokenizer")]
mod wordchipper;

#[cfg(feature = "tokenizer")]
pub use wordchipper::WordchipperDetokenizer;

#[cfg(feature = "tokenizer")]
mod bunsen_tokenizer;
#[cfg(feature = "tokenizer")]
mod tiktoken_pickle;

#[cfg(feature = "tokenizer")]
pub use bunsen_tokenizer::*;

#[cfg(feature = "testing")]
pub mod testing;

mod byte_level;
mod tekken;
mod tiktoken_util;
pub use byte_level::*;
pub use tekken::*;
pub use tiktoken_util::*;

/// Turns token ids into text.
///
/// The seam a kit holds as `Option<Arc<dyn Detokenizer>>`: without one, its
/// output is ids, which is still a complete result. Ids are `i64`, as the
/// model's token tensors are; the implementation narrows them.
///
/// `Debug` is a supertrait so that a module holding one can still derive
/// `Debug`; `Send + Sync` so that one can be shared across streams.
pub trait Detokenizer: Send + Sync + Debug {
    /// Renders `ids` as text, a special token as its own spelling.
    ///
    /// Bytes are joined **before** the UTF-8 decode, so a character whose
    /// bytes span two ids survives; decoding each id alone would lose both
    /// halves. Which ids to render — whether to drop the prompt, the
    /// timestamps, the stop token — is the caller's policy, applied before
    /// the call; the detokenizer renders whatever it is handed.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// if an id is outside the vocabulary. Nothing is skipped silently. The
    /// ids are a model's output, so an id the vocabulary does not cover means
    /// the model and its vocabulary disagree; a caller that built the ids
    /// itself can re-mark the error
    /// [`as_illegal`](crate::errors::ResultContext::as_illegal).
    fn detokenize(
        &self,
        ids: &[i64],
    ) -> BunsenResult<String>;
}

/// A checkpoint's tokenizer, text to ids and back: what a kit that builds
/// its own prompts, or holds a conversation, reads and writes text through.
///
/// Ids are `i64`, as the model's token tensors are. It is a [`Detokenizer`],
/// and shares its supertraits for the same reasons.
pub trait Tokenizer: Detokenizer {
    /// Encodes `text`, adding no special tokens of its own, an added token's
    /// spelling anywhere in it as that token: `transformers`'
    /// `tokenizer(text, add_special_tokens=False)`.
    ///
    /// # Errors
    /// As the encoder.
    fn encode(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>>;

    /// Encodes `text` as [`encode`](Self::encode) does, but with a special
    /// added token's spelling as text: `transformers`'
    /// `encode_special_tokens`, `tiktoken`'s `encode_ordinary`. An added token
    /// that is not special still encodes as itself.
    ///
    /// # Errors
    /// As the encoder.
    fn encode_plain(
        &self,
        text: &str,
    ) -> BunsenResult<Vec<i64>>;

    /// The id of an added token, by its content: `[MASK]`, `<end_of_turn>`.
    fn token_id(
        &self,
        content: &str,
    ) -> Option<i64>;

    /// Whether `id` is a token.
    fn contains(
        &self,
        id: i64,
    ) -> bool;

    /// The token count, added tokens included.
    fn len(&self) -> usize;

    /// Whether there are no tokens.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;

    use super::*;

    /// A detokenizer with no assets: renders each id as its number.
    #[derive(Debug)]
    struct Numbers;

    impl Detokenizer for Numbers {
        fn detokenize(
            &self,
            ids: &[i64],
        ) -> BunsenResult<String> {
            Ok(ids.iter().map(i64::to_string).collect::<Vec<_>>().join(" "))
        }
    }

    /// The shape a kit holds: shareable, dynamic, and debuggable.
    #[test]
    fn test_trait_object() {
        let detokenizer: Option<Arc<dyn Detokenizer>> = Some(Arc::new(Numbers));

        assert_eq!(format!("{detokenizer:?}"), "Some(Numbers)");
        assert_eq!(
            detokenizer
                .as_ref()
                .unwrap()
                .detokenize(&[1, 2, 3])
                .unwrap(),
            "1 2 3"
        );
    }
}
