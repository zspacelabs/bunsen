//! # Attention Extensions
//!
//! # Key/value cached attention over `MultiHeadAttention` weights.
//!
//! A genuine KV cache: keys and values are kept **projected and head-split**,
//! so a decode step attends over the whole history while projecting only the
//! new tokens, and the attention itself only scores the new queries.
//!
//! This reuses [`MultiHeadAttention`]'s own `Linear` layers rather than
//! replacing the module, so weights, loading and any cross-checks stay exactly
//! as they are — only the forward path differs.
//!
//! ## Why not `MhaCache`
//!
//! [`MhaCache`](burn::nn::attention::MhaCache) caches the *projections* and
//! expects the whole sequence back on every call, so its attention still scores
//! the full prefix each step. That is a real saving for cross-attention, whose
//! keys and values never change, but it leaves self-attention quadratic. These
//! functions take only the new tokens.
//!
//! ## Why not `KVCache`
//!
//! [`KVCache`](super::KVCache) is a preallocated *multi-layer* cache: a single
//! `Tensor<B, 6>` over `[layers, kv, batch, heads, seq, d_k]`, grown in chunks,
//! and a `Module` in its own right. It wants the geometry — layer count, head
//! count, head dimension, batch size, and a sequence bound — declared up front,
//! and it serves bunsen's own attention stack
//! ([`CausalSelfAttention`](super::CausalSelfAttention)).
//!
//! [`AttnKvPair`] is a plain per-layer value over burn's [`MultiHeadAttention`]
//! weights. It declares no geometry, and it also covers a case `KVCache` does
//! not model: cross-attention keys and values, projected once from the encoder
//! output and then reused unchanged rather than grown.
//!
//! Reach for `KVCache` when the geometry is known up front and the allocation
//! matters; reach for these when the attention is burn's own and the cache is
//! per-layer.
//!
//! ## Matching the uncached path
//!
//! The arithmetic mirrors `MultiHeadAttention::forward` exactly — the same
//! `1/sqrt(d_k)` scaling, the same `min_float` mask fill, the same
//! `quiet_softmax` switch — so a cached decode reproduces an uncached one.
//! `layer_norm_self_attn_kv`'s contract test is what holds that true.
//!
//! Note this path deliberately omits the attention-score dropout that
//! `MultiHeadAttention::attn_scores` applies: caching is an inference concern,
//! and burn's `Dropout` is a no-op outside training anyway.

mod attend;
mod attn_kv_pair;
mod causal_mask;

#[doc(inline)]
pub use attend::*;
#[doc(inline)]
pub use attn_kv_pair::*;
#[doc(inline)]
pub use causal_mask::*;
