//! # Attention blocks
//!
//! [`CausalSelfAttention`] is bunsen's own multi-head causal self-attention
//! layer, the one the [`NanoChatGpt`] kit stacks. It is a part of a module
//! tree, built once per layer. The operations it calls and the cache state it
//! carries live in [`crate::ops::transformers::attention`].
//!
//! ## What it computes
//!
//! `CausalSelfAttention` is projections + QK-norm + `RoPE` + [`KVCache`] +
//! SDPA. For a `[B, T, D]` input:
//!
//! 1. bias-free `Linear` projections to `n_head` query heads and `n_kv_head`
//!    key/value heads, each `head_dim = n_embed / n_head` wide;
//! 2. rotary position embedding on the queries and keys, from the
//!    [`RotaryEmbedding`] passed to `forward`;
//! 3. QK-norm: per-head normalization of the queries and keys, sized
//!    `head_dim`, applied after the rotation (RMS norm by default; the config's
//!    `norm`);
//! 4. with a cache, appending this step's keys and values to the [`KVCache`]
//!    and reading back the full history;
//! 5. [`scaled_dot_product_attention`], then the output projection.
//!
//! **Grouped-query attention.** With `n_kv_head < n_head` (it must divide
//! `n_head`), the keys and values have fewer heads, and SDPA repeats each one
//! `n_head / n_kv_head` times. `n_kv_head == n_head` is plain multi-head
//! attention.
//!
//! ## KV-cache decode modes
//!
//! [`forward`](csa::CausalSelfAttention::forward) takes
//! `kv_cache: &mut Option<&mut KVCache>`. The cache is injected: the
//! caller owns it (for example from [`NanoChatGpt::new_kv_cache`]), so one
//! model can run several decodes at once.
//!
//! - `None`: no cache. The input is the whole sequence, attended causally; a
//!   decode loop recomputes every position each step.
//! - `Some(cache)`: the step's keys and values are written at this layer's
//!   `layer_index`, and attention runs over the cached history plus the step:
//!   - into an empty cache (prefill), the query and key lengths match and the
//!     attention is causal;
//!   - for one new token, the query attends to every cached position, with no
//!     mask;
//!   - for several new tokens after a cached prefix, an explicit `[T_q, T_kv]`
//!     mask lets each new token attend to the whole prefix, to itself, and to
//!     the new tokens before it, so the chunk's outputs match a pass over the
//!     whole sequence.
//!
//! The cache advances its position after the last layer, so one `KVCache`
//! whose `num_layers` is the model depth serves the whole stack.
//!
//! ## Two attention stacks
//!
//! The other stack runs attention over burn's own
//! [`MultiHeadAttention`] weights, with a per-layer cache, and serves the
//! Whisper kit. [`crate::ops::transformers::attention`] compares the two and
//! says when to use which.
//!
//! ## Lifecycle
//!
//! [`CausalSelfAttentionConfig`] builds the layer with an inherent
//! `init(layer_index, device)` rather than [`ModuleInit`], because the layer
//! needs its index in the stack. [`CausalSelfAttentionMeta`] reads the same
//! on the config and the module.
//!
//! [`CausalSelfAttention`]: csa::CausalSelfAttention
//! [`CausalSelfAttentionConfig`]: csa::CausalSelfAttentionConfig
//! [`CausalSelfAttentionMeta`]: csa::CausalSelfAttentionMeta
//! [`RotaryEmbedding`]: crate::blocks::transformers::embedding::RotaryEmbedding
//! [`KVCache`]: crate::ops::transformers::attention::KVCache
//! [`scaled_dot_product_attention`]: crate::ops::transformers::attention::scaled_dot_product_attention
//! [`MultiHeadAttention`]: burn::nn::attention::MultiHeadAttention
//! [`ModuleInit`]: crate::burner::module::ModuleInit
//! [`NanoChatGpt`]: crate::kits::gpts::nanochat::NanoChatGpt
//! [`NanoChatGpt::new_kv_cache`]: crate::kits::gpts::nanochat::NanoChatGpt::new_kv_cache

pub mod csa;
