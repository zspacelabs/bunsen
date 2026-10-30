//! # Driver support: the model-generic parts of a generation driver
//!
//! What a driver that streams a language model's output needs, whatever the
//! model: none of it knows a tokenizer, a template or a weight.
//!
//! - [`TextStream`]: a reply's text as it grows, one delta per token, never
//!   splitting a character whose bytes span two tokens; over any
//!   [`Detokenizer`](crate::kits::tokens::Detokenizer).
//! - [`StopStrings`]: stop strings matched across those deltas, holding back
//!   what might yet become one.
//! - [`CachedHistory`]: the tokens a
//!   [`KVCache`](crate::ops::transformers::attention::KVCache) holds, and the
//!   rewind to the prefix the next prompt shares with them
//!   ([`KVCache::truncate`](crate::ops::transformers::attention::KVCache::truncate)),
//!   so a multi-turn session prefills only what changed.
//!
//! The searches a driver runs are
//! [`kits::decoding`](crate::kits::decoding)'s.

mod cached_history;
mod stop_strings;
mod text_stream;

pub use cached_history::*;
pub use stop_strings::*;
pub use text_stream::*;
