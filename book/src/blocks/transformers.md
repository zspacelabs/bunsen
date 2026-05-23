# Transformer Blocks

`bunsen::blocks::transformers` collects the building blocks used by
transformer-family models &mdash; attention layers, their caching
machinery, and positional embeddings.

API: <https://docs.rs/bunsen/latest/bunsen/blocks/transformers/>

## Attention

The [`attention`](https://docs.rs/bunsen/latest/bunsen/blocks/transformers/attention/index.html)
submodule houses the attention layers themselves and the helpers
they're built from.

### `CausalSelfAttention`

[`CausalSelfAttention`](https://docs.rs/bunsen/latest/bunsen/blocks/transformers/attention/struct.CausalSelfAttention.html)
is multi-head causal self-attention with optional KV-grouping. The
config carries:

- `n_head` &mdash; number of query heads,
- `n_kv_head` &mdash; number of key/value heads (must divide `n_head`;
  equals `n_head` for plain MHA, less for grouped-query attention),
- `n_embed` &mdash; embedding dimension,
- a pluggable `NormalizationConfig` applied inside the block.

The module exposes a `CausalSelfAttentionMeta` trait, implemented on
both the config and the live module. Parents can read `n_head`,
`n_kv_head`, and `head_dim` of whichever form they're holding, so
larger transformers don't need to cache those numbers themselves.
This is the pattern documented in
[Building Reusable Modules](../guides/building-reusable-modules.md).

`forward` takes the input embedding plus an optional `&mut KVCache`
for autoregressive decoding. When the cache is `None` the layer runs
in training/prefill mode and recomputes K and V each call; when it's
`Some`, K and V are appended into the cache and read back across the
full sequence.

### `KVCache`

[`KVCache`](https://docs.rs/bunsen/latest/bunsen/blocks/transformers/attention/struct.KVCache.html)
is the per-layer key/value tensor cache for fast incremental
decoding. Built from a `KVCacheConfig` carrying `batch_size`,
`num_heads`, `seq_len`, `head_dim`, and `num_layers`, it provides:

- `pos()` &mdash; the current write head position,
- `prefill(...)` &mdash; bulk-load K/V from a prompt encode,
- `insert_kv(...)` &mdash; append a single decoded step's K/V,
- `reset()` &mdash; rewind to position 0 without reallocating.

`NanoChatGpt` uses one shared `KVCache` across all its layers; see
[`bunsen::kits::gpts::nanochat`](../kits/gpts.md) for the integrated
example.

### Scaled-dot-product helpers

When you need to wire attention by hand &mdash; for a custom block,
a fused-kernel experiment, or unit tests &mdash; the functional API
is available:

- [`scaled_dot_product_attention`](https://docs.rs/bunsen/latest/bunsen/blocks/transformers/attention/fn.scaled_dot_product_attention.html)
  &mdash; the full SDPA op given Q, K, V and an optional mask/bias.
- `sdpa_attn_weight` &mdash; just the softmax-of-scaled-QK^T factor.
- `sdpa_bias` &mdash; build an additive bias tensor (causal mask,
  ALiBi, etc.) of the right shape for SDPA.

## Embedding

The [`embedding`](https://docs.rs/bunsen/latest/bunsen/blocks/transformers/embedding/index.html)
submodule collects positional embeddings.

### `RotaryEmbedding`

[`RotaryEmbedding`](https://docs.rs/bunsen/latest/bunsen/blocks/transformers/embedding/struct.RotaryEmbedding.html)
is RoPE with a precomputed frequency table:

- `RotaryEmbeddingConfig::new(seq_len, head_dim)` then `.init(device)`
  allocates the table once for the maximum sequence length.
- `apply(q, k)` rotates query and key tensors.
- `clip_range(t0..t1)` returns a sliced view for serving a partial
  sequence &mdash; the natural fit for KV-cache decoding, where each
  step only needs the rotations for the new positions.
- `cast(dtype)` converts the precomputed table between float dtypes
  without recomputing the trigonometric values.

The free functions `inverse_frequency_table` and
`positional_frequency_table` are exposed for callers that want to
build their own variant of rotary embedding without going through
the packaged module.
