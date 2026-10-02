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
//! - **Over burn's `MultiHeadAttention`** (the Whisper kit):
//!   [`layer_norm_self_attn`] and [`layer_norm_cross_attn`], their
//!   `_w_kv_cache` variants, [`attend_q_kv_mask`] and [`causal_mask`], with a
//!   per-layer [`AttnKvPair`] cache. The model keeps burn's module, so its
//!   weights, loading and any cross-checks are unchanged; only the forward path
//!   differs.
//! - **bunsen's [`CausalSelfAttention`]** block (the `NanoChat` GPT kit): its
//!   own bias-free projections, QK-norm, rotary embeddings and grouped-query
//!   attention, computed with [`scaled_dot_product_attention`] over a
//!   preallocated multi-layer [`KVCache`].
//!
//! Use `CausalSelfAttention` for a decoder-only model whose layers you
//! define. Use the `MultiHeadAttention` functions when the model is built on
//! burn's module (a ported checkpoint in that layout, or encoder-decoder
//! cross-attention) and you want KV-cached decoding without replacing the
//! module.
//!
//! ## Two caches
//!
//! Both caches are *injected*: the caller builds one per decode and passes
//! it in, and the model never owns it, so one model can run several decodes
//! in the same process.
//!
//! [`KVCache`] is a preallocated *multi-layer* cache: a single
//! `[layers, kv, batch, heads, seq, d_k]` tensor, grown in chunks. Its
//! geometry (layer count, head count, head dimension, batch size, and an
//! initial sequence length) is declared up front in a [`KVCacheConfig`].
//! [`insert_kv`](KVCache::insert_kv) appends one layer's keys and values and
//! returns that layer's full history; the position advances after the last
//! layer. It is a `Module` over a bare tensor: state, not parameters.
//!
//! [`AttnKvPair`] is a plain per-layer value over burn's
//! [`MultiHeadAttention`] weights. It declares no geometry, and it also covers
//! a case `KVCache` does not model: cross-attention keys and values, projected
//! once from the encoder output and then reused unchanged rather than grown.
//!
//! Reach for `KVCache` when the geometry is known up front and the allocation
//! matters; reach for `AttnKvPair` when the attention is burn's own and the
//! cache is per-layer.
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
//! `layer_norm_self_attn_w_kv_cache`'s contract test is what holds that true.
//!
//! Note this path deliberately omits the attention-score dropout that
//! `MultiHeadAttention::attn_scores` applies: caching is an inference concern,
//! and burn's `Dropout` is a no-op outside training anyway.
//!
//! [`MultiHeadAttention`]: burn::nn::attention::MultiHeadAttention
//! [`CausalSelfAttention`]: crate::blocks::transformers::attention::csa::CausalSelfAttention

mod attend;
mod attn_kv_pair;
mod causal_mask;
mod kv_cache;
mod sdpa;

pub use attend::*;
pub use attn_kv_pair::*;
pub use causal_mask::*;
pub use kv_cache::*;
pub use sdpa::*;
