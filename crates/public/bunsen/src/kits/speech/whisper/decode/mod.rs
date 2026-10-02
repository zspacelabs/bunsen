//! # Decoding: from mel windows to token ids.
//!
//! One window of packaged log-mels in, the ids the model says for it out,
//! as upstream's `DecodingTask` and `transcribe()` do it. The pieces:
//!
//! - [`DecodeConfig`]: the prompt, stop token, cap, and search width of a
//!   decode. [`init_decoder`](DecodeConfig::init_decoder) picks the search,
//!   [`init_ranker`](DecodeConfig::init_ranker) the ranker.
//! - [`TokenDecoder`]: the search seam. The decode loop owns the tensors and
//!   the text decoder's cache; the search owns the sequences' bookkeeping.
//!   [`WhisperGreedyDecoder`] is the argmax, or sampling above temperature
//!   zero; [`WhisperBeamSearchDecoder`] is upstream's beam search.
//! - [`SequenceRanker`]: which finished candidate a decode returns.
//! - [`WhisperFallbackConfig`] and [`decode_with_fallback`]: the temperature
//!   ladder that re-decodes a window that looks bad.
//! - The loop itself is on the model: [`decode_windows`] and its siblings on
//!   [`Whisper`](super::Whisper) return ids per window, and the `_full` forms
//!   return [`DecodedTokens`], which carry what the fallback ladder judges by.
//!
//! The stream driver ([`driver`](super::driver)) builds the
//! [`DecodeConfig`] for every window and climbs the ladder; a caller with
//! whole windows in hand can call the model directly.
//!
//! [`decode_windows`]: super::Whisper::decode_windows

mod beam_search_decoder;
mod greedy_decoder;
mod sequence_ranker;
mod token_decoder;
mod whisper_fallback_config;

pub use beam_search_decoder::*;
pub use greedy_decoder::*;
pub use sequence_ranker::*;
pub use token_decoder::*;
pub use whisper_fallback_config::*;
