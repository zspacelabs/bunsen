//! # Decoding: from mel windows to token ids.
//!
//! One window of packaged log-mels in, the ids the model says for it out,
//! as upstream's `DecodingTask` and `transcribe()` do it, over the shared
//! search machinery in [`kits::decoding`](crate::kits::decoding). The
//! pieces:
//!
//! - [`DecodeConfig`]: the prompt, stop token and cap of a decode, and its
//!   [`SearchConfig`](crate::kits::decoding::SearchConfig).
//!   [`init_decoder`](DecodeConfig::init_decoder) picks the search,
//!   [`init_ranker`](DecodeConfig::init_ranker) the ranker.
//! - [`WhisperFallbackConfig`] and [`decode_with_fallback`]: a
//!   [`TemperatureLadder`](crate::kits::decoding::TemperatureLadder) that
//!   re-decodes a window that looks bad, judged by upstream's thresholds.
//! - The loop itself is on the model: [`decode_windows`] and its siblings on
//!   [`Whisper`](super::Whisper) return ids per window, and the `_full` forms
//!   return [`DecodedTokens`], which carry what the fallback ladder judges by.
//!   The loop owns the tensors and the text decoder's cache; the
//!   [`TokenSearch`](crate::kits::decoding::TokenSearch) owns the sequences'
//!   bookkeeping.
//!
//! The stream driver ([`driver`](super::driver)) builds the
//! [`DecodeConfig`] for every window and climbs the ladder; a caller with
//! whole windows in hand can call the model directly.
//!
//! [`decode_windows`]: super::Whisper::decode_windows

mod token_decoder;
mod whisper_fallback_config;

pub use token_decoder::*;
pub use whisper_fallback_config::*;
