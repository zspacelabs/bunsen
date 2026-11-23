//! # Row reductions over a wide last axis
//!
//! A decode step reduces its logits, `[rows, V]` with a few rows and a
//! vocabulary `V` of 50k-250k, once or more per token: the argmax a greedy
//! step reads, the top `k + 1` a beam keeps, the normalizer of a log
//! probability or a nucleus. burn's whole-row reductions on that shape
//! parallelize over the rows, so a few rows leave the device nearly idle:
//! on CUDA, `argmax` over `[1, 151936]` takes ~2 ms and `topk(20)` ~10 ms,
//! against a ~16 ms step of Qwen3-0.6B.
//!
//! These reduce each row by chunks of [`ROW_CHUNK`] positions, the chunks
//! in parallel, then across the chunks: the same values (the argmax's and
//! the top-k's ties broken as burn's are, the first index first), at a
//! tenth to a fiftieth of the time ([`argmax_rows`], [`max_rows`],
//! [`logsumexp_rows`], [`softmax_normalizer_rows`]).
//!
//! The top-k ([`topk_rows`], [`topk_rows_host`]) reads to the host: the
//! chunks' maxima pick the `k` chunks the top `k` lie in, and the host
//! selects among their values. burn's top-k reduction compiles a kernel for
//! each `k` and shape, seconds each on CUDA, which a short decode pays in
//! full; these compile a few small kernels, whatever `k`.
//!
//! All of them reduce in f32, whatever the input's dtype.

mod rows;

pub use rows::*;
