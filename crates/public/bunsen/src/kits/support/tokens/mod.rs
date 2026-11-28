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
//! [`TekkenVocabulary`] is Mistral's, which
//! [`TekkenMirror`](super::mistral_common::interop::TekkenMirror) reads from
//! `tekken.json`.
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
pub(crate) mod tiktoken_util;
pub use byte_level::*;
pub use tekken::*;
pub use tiktoken_util::*;

mod seam;

pub use seam::*;

#[cfg(test)]
mod tests;
