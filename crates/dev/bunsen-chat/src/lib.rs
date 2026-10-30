//! # bunsen-chat
//!
//! Chat with bunsen's Qwen3 ([`bunsen::kits::gpts::qwen3`]): what turns a
//! conversation into the ids the model reads and its ids back into text, and
//! the driver that holds a conversation on the KV cache.
//!
//! - [`ChatModel`] and [`ChatSession`]: the shared, immutable half of a chat (a
//!   loaded checkpoint and its files) and one conversation (its messages and
//!   the cache that holds them). [`send`](ChatSession::send) streams a reply
//!   through [`decode_cached_observed`](bunsen::kits::decoding::decode_cached_observed),
//!   and prefills only what the cache does not already hold.
//! - [`GenerationDefaults`]: a checkpoint's stop tokens and sampling.
//! - [`Qwen3Tokenizer`]: a checkpoint's tokenizer, text to ids and back, as
//!   `transformers` encodes; a
//!   [`Detokenizer`](bunsen::kits::tokens::Detokenizer).
//! - [`ChatTemplate`] and [`ChatMessage`]: a checkpoint's chat template, a
//!   conversation rendered as `transformers` renders it.

mod generation;
mod session;
mod template;
mod tokenizer;

pub use generation::*;
pub use session::*;
pub use template::*;
pub use tokenizer::*;
