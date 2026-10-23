//! # Attention operations
//!
//! Attention math and the cache state it carries, as functions and plain
//! values. Nothing here owns trainable weights: a function either borrows
//! burn's [`MultiHeadAttention`] module and uses its `Linear` layers, or takes
//! `q`, `k` and `v` tensors that are already projected.
//!
//! ## Two attention stacks
//!
//! bunsen has two attention stacks, and this module serves both.
//!
//! - **Over burn's `MultiHeadAttention`** (the Whisper kit): [`norm_self_attn`]
//!   and [`norm_cross_attn`], their `_w_kv_cache` variants,
//!   [`attend_q_kv_mask`] and [`causal_mask`], over a [`KVCache`]. The model
//!   keeps burn's module, so its weights, loading and any cross-checks are
//!   unchanged; only the forward path differs.
//! - **bunsen's [`CausalSelfAttention`]** block (the `NanoChat` and Qwen3 GPT
//!   kits): its own bias-free projections, QK-norm, rotary embeddings and
//!   grouped-query attention, computed with [`scaled_dot_product_attention`]
//!   over a [`KVCache`].
//!
//! Use `CausalSelfAttention` for a decoder-only model whose layers you
//! define. Use the `MultiHeadAttention` functions when the model is built on
//! burn's module (a ported checkpoint in that layout, or encoder-decoder
//! cross-attention) and you want KV-cached decoding without replacing the
//! module.
//!
//! ## Relative positions
//!
//! [`rel_position_table`] and [`rel_shift`] are the two pieces of
//! Transformer-XL relative-position attention that are not weights: the
//! sinusoidal table of relative positions, and the shift that aligns
//! query-by-position scores into query-by-key scores. The Conformer's
//! [`RelPosSelfAttention`](crate::blocks::transformers::attention::rel_pos::RelPosSelfAttention)
//! block is built on them.
//!
//! [`length_mask`] is the `[batch, time]` validity mask of a padded batch;
//! [`window_block_mask`] is the `[n, n]` mask of block-diagonal (windowed)
//! self-attention.
//!
//! ## The cache
//!
//! Every decoder decodes through one [`KVCache`], and it is *injected*: the
//! caller builds one per decode and passes it in, and the model never owns
//! it, so one model can run several decodes in the same process. A model
//! hands out one sized for it ([`KVCacheModel`]).
//!
//! It holds each layer's self-attention history, stored as its config's
//! [`KVCacheStorage`] policy says, and, for an encoder-decoder, each layer's
//! cross-attention keys and values: projected once from the encoder output
//! ([`project_kv`]), set once, and shared by a group of decode rows (a beam
//! search's beams) rather than repeated per row.
//! [`insert_kv`](KVCache::insert_kv) appends one layer's keys and values and
//! returns that layer's full history; the position advances after the last
//! layer. It is a `Module` over bare tensors: state, not parameters.
//!
//! ## Why not `MhaCache`
//!
//! The `MultiHeadAttention` functions keep a genuine KV cache: keys and
//! values are kept **projected and head-split**, so a decode step attends
//! over the whole history while projecting only the new tokens, and the
//! attention itself only scores the new queries.
//!
//! [`MhaCache`](burn::nn::attention::MhaCache) caches the *projections* and
//! expects the whole sequence back on every call, so its attention still scores
//! the full prefix each step. That is a real saving for cross-attention, whose
//! keys and values never change, but it leaves self-attention quadratic. These
//! functions take only the new tokens.
//!
//! ## Matching the uncached path
//!
//! The arithmetic mirrors `MultiHeadAttention::forward` exactly — the same
//! `1/sqrt(d_k)` scaling, the same `min_float` mask fill, the same
//! `quiet_softmax` switch — so a cached decode reproduces an uncached one.
//! `norm_self_attn_w_kv_cache`'s contract test is what holds that true.
//!
//! Note this path deliberately omits the attention-score dropout that
//! `MultiHeadAttention::attn_scores` applies: caching is an inference concern,
//! and burn's `Dropout` is a no-op outside training anyway.
//!
//! [`MultiHeadAttention`]: burn::nn::attention::MultiHeadAttention
//! [`CausalSelfAttention`]: crate::blocks::transformers::attention::csa::CausalSelfAttention

mod attend;
mod causal_mask;
mod kv_cache;
mod length_mask;
mod rel_pos;
mod sdpa;

pub use attend::*;
pub use causal_mask::*;
pub use kv_cache::*;
pub use length_mask::*;
pub use rel_pos::*;
pub use sdpa::*;
