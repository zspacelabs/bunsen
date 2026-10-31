//! # bunsen-chat
//!
//! Chat with bunsen's Qwen3 ([`bunsen::kits::gpts::qwen3`]) or nanochat
//! ([`bunsen::kits::gpts::nanochat`]): what turns a conversation into the ids
//! the model reads and its ids back into text, and the driver that holds a
//! conversation on the KV cache.
//!
//! - [`ChatModel`] and [`ChatSession`]: the shared, immutable half of a chat (a
//!   loaded checkpoint and its files) and one conversation (its messages and
//!   the cache that holds them), over a [`ChatBackend`]: the checkpoint's
//!   model, and what turns a conversation into its ids and its ids into text
//!   ([`Qwen3Chat`], Qwen3's). [`send`](ChatSession::send) streams a reply
//!   through [`decode_cached_observed`](bunsen::kits::decoding::decode_cached_observed),
//!   and prefills only what the cache does not already hold.
//! - [`GenerationDefaults`]: a checkpoint's stop tokens and sampling.
//! - [`Qwen3Tokenizer`]: a checkpoint's tokenizer, text to ids and back, as
//!   `transformers` encodes; a
//!   [`Detokenizer`](bunsen::kits::tokens::Detokenizer).
//! - [`ChatTemplate`] and [`ChatMessage`]: a checkpoint's chat template, a
//!   conversation rendered as `transformers` renders it.
//! - [`NanoChatChat`], [`NanoChatTokenizer`] and [`NanoChatLayout`]: a nanochat
//!   checkpoint's chat: its `tokenizer.pkl`, read through `serde_pickle`, and
//!   upstream's conversation layout, each as upstream encodes and renders.
//!
//! The `bunsen-chat` binary runs them. Its `list`, `fetch`, `run` and `chat`
//! take a checkpoint of any family by its name; `family qwen3` and `family
//! nanochat` have the same commands for one family, its own default by
//! default, with only the flags that apply to it:
//!
//! ```text
//! bunsen-chat list
//! bunsen-chat cache list qwen3
//! bunsen-chat run --device wgpu --no-think "What is the capital of France?"
//! bunsen-chat chat --device wgpu --model qwen3-4b-instruct-2507 --stats
//! bunsen-chat family nanochat list --urls
//! bunsen-chat family nanochat chat --device cuda --precision half
//! ```

mod backend;
mod generation;
mod nanochat_chat;
mod nanochat_layout;
mod nanochat_tokenizer;
mod session;
mod template;
mod tokenizer;

pub use backend::*;
pub use generation::*;
pub use nanochat_chat::*;
pub use nanochat_layout::*;
pub use nanochat_tokenizer::*;
pub use session::*;
pub use template::*;
pub use tokenizer::*;
