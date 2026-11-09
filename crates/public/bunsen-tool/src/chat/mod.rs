//! Chat with bunsen's Qwen3 ([`bunsen::kits::language::qwen3`]), Gemma
//! ([`bunsen::kits::language::gemma`]) or nanochat
//! ([`bunsen::kits::language::nanochat`]): what turns a conversation into the
//! ids the model reads and its ids back into text, and the driver that holds a
//! conversation on the KV cache.
//!
//! - [`ChatModel`] and [`ChatSession`]: the shared, immutable half of a chat (a
//!   loaded checkpoint and its files) and one conversation (its messages and
//!   the cache that holds them), over a [`ChatBackend`]: the checkpoint's
//!   model, and what turns a conversation into its ids and its ids into text
//!   ([`LmChat`], a stacked language model's: [`Qwen3Chat`], [`GemmaChat`],
//!   [`Qwen3_5Chat`]). [`send`](ChatSession::send) streams a reply through
//!   [`decode_cached_observed`](bunsen::kits::support::decoding::decode_cached_observed),
//!   and prefills only what the cache does not already hold.
//! - [`search_config`]: the search a checkpoint's generation policy says.
//! - Every family's tokenizer is a
//!   [`BunsenTokenizer`](bunsen::kits::support::tokens::BunsenTokenizer):
//!   Qwen3's and Gemma's from their `tokenizer.json`, nanochat's from its
//!   `tokenizer.pkl`; a chat holds it as a
//!   [`Tokenizer`](bunsen::kits::support::tokens::Tokenizer).
//! - [`ChatTemplate`] and [`ChatMessage`]: a checkpoint's chat template, a
//!   conversation rendered as `transformers` renders it.
//! - [`NanoChatChat`] and [`NanoChatLayout`]: a nanochat checkpoint's chat,
//!   through its tokenizer, and upstream's conversation layout, as upstream
//!   renders it.
//!
//! - [`choose_checkpoint`]: the checkpoint a command chooses, by its
//!   `--select`s and `--preset`s, among every family's listed checkpoints (or
//!   one family's), each a selection candidate tagged with its family
//!   ([`chat_selection`]: the defaults, and each family's `default`).
//!
//! `bunsen-tool chat` runs them. Its `list`, `fetch`, `run` and `repl` take
//! a checkpoint of any family by its name; `family qwen3`, `family gemma`
//! and `family nanochat` have the same commands for one family, its own
//! default by default, with only the flags that apply to it:
//!
//! ```text
//! bunsen-tool chat list
//! bunsen-tool cache list qwen3
//! bunsen-tool chat run --device wgpu --no-think "What is the capital of France?"
//! bunsen-tool chat repl --device wgpu --model qwen3-4b-instruct-2507 --stats
//! bunsen-tool chat family nanochat list --urls
//! bunsen-tool chat family nanochat repl --device cuda
//! ```

mod backend;
mod chat_command_group;
mod chat_model_family_flags;
mod chat_select;
mod generation;
mod nanochat_chat;
mod nanochat_layout;
mod session;
mod template;

pub use backend::*;
pub use chat_command_group::*;
pub use chat_model_family_flags::*;
pub use chat_select::*;
pub use generation::*;
pub use nanochat_chat::*;
pub use nanochat_layout::*;
pub use session::*;
pub use template::*;
