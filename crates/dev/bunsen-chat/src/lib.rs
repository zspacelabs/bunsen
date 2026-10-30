//! # bunsen-chat
//!
//! Chat with bunsen's Qwen3 ([`bunsen::kits::gpts::qwen3`]): what turns a
//! conversation into the ids the model reads and its ids back into text.
//!
//! - [`Qwen3Tokenizer`]: a checkpoint's tokenizer, text to ids and back, as
//!   `transformers` encodes; a
//!   [`Detokenizer`](bunsen::kits::tokens::Detokenizer).
//! - [`ChatTemplate`] and [`ChatMessage`]: a checkpoint's chat template, a
//!   conversation rendered as `transformers` renders it.

mod template;
mod tokenizer;

pub use template::*;
pub use tokenizer::*;
