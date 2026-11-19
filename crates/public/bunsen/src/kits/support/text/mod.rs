//! # Text normalization, and language tags
//!
//! Text as written, to the form a model or a comparison reads. Not speech
//! alone: a TTS driver normalizes what it speaks, and a word error rate
//! normalizes both transcripts it compares.
//!
//! - [`TextNormalizer`]: the seam, plugged in as an `Arc<dyn TextNormalizer>`;
//!   [`TextNormalizerChain`] runs several in order.
//! - The normalizers bunsen ships: [`CollapseLineBreaks`],
//!   [`TranscriptTextNormalizer`], and Zyphra's [`ZyphraTextNormalizer`]
//!   (numbers, money and dates spoken, in 43 languages; feature
//!   `zyphra-textnorm`), and a `tokenizers` normalizer run on its own
//!   (`TokenizersTextNormalizer`, feature `tokenizer`: Unicode forms,
//!   lowercasing, a model's own normalizer); [`TextNormalizationConfig`] names
//!   them as config and builds them.
//! - [`BunsenLanguageTag`]: a well-known language code (BCP 47), the language a
//!   normalizer is told a text is in.

mod basic;
mod config;
mod language;
mod normalizer;
#[cfg(feature = "tokenizer")]
mod tokenizers_normalizer;
mod zyphra;

pub use basic::*;
pub use config::*;
pub use language::*;
pub use normalizer::*;
#[cfg(feature = "tokenizer")]
pub use tokenizers_normalizer::*;
pub use zyphra::*;
