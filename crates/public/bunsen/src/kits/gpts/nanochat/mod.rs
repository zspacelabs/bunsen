//! # `NanoChat`
//!
//! A port of the GPT in karpathy's
//! [nanochat](https://github.com/karpathy/nanochat) (`nanochat/gpt.py`):
//! a decoder-only transformer with rotary embeddings, grouped-query
//! attention, a squared-ReLU MLP, and softcapped logits. With it comes the
//! corpus nanochat trains on, as a shard-set table.
//!
//! The model trains from scratch (`examples/train-chat`), and loads
//! upstream's own checkpoints, `model_<step>.pt` and `meta_<step>.json` as
//! upstream's checkpoint manager writes them, through [`weights`]
//! ([`load_pt_checkpoint`](weights::load_pt_checkpoint)): the meta's geometry,
//! which of upstream's models the state dict is, and the weights through the
//! burn-store's `PyTorch` reader.
//!
//! # Lifecycle
//!
//! The model's config is
//! [Stacked](crate::burner::module::ModuleInit#stacked-config):
//!
//! 1. [`NanoChatGptContractConfig`] is the policy: the vocabulary size, the
//!    depth, the query and KV head counts, the width, the MLP expansion and
//!    activation, the norm, the logit softcap, and the sequence lengths.
//! 2. [`to_structure`](crate::burner::module::ToStructureConfig::to_structure)
//!    unrolls it into a [`NanoChatGptStructureConfig`]: the token embedding,
//!    one [`CausalDecoderBlockConfig`](crate::blocks::transformers::decoders::causal::CausalDecoderBlockConfig) per layer,
//!    the head, and the rotary-embedding table.
//! 3. `init` builds the [`NanoChatGpt`]: from the structure, or from the policy
//!    in one step through the blanket
//!    [`ModuleInit`](crate::burner::module::ModuleInit).
//! 4. [`forward`](NanoChatGpt::forward) maps `[batch, seq]` token ids to
//!    `[batch, seq, vocab_size]` logits.
//!
//! [`NanoChatGptMeta`] reads the geometry back from all three forms.
//!
//! # Upstream's two models
//!
//! The policy's defaults are upstream's model as of late 2025: an unbiased
//! head, rotary base 10000 with its tables rounded to bf16 values (as
//! upstream keeps them), norms with an explicit epsilon
//! ([`norm_eps`](NanoChatGptContractConfig::norm_eps)). Upstream's 2026 model
//! adds a set of options, each off by default and all on through
//! [`with_upstream_2026`](NanoChatGptContractConfig::with_upstream_2026): the
//! smear (each position mixes in the previous one's embedding), per-layer
//! residual and embedding lambdas, value embeddings on every other layer
//! ([`has_value_embed`]), the mid-stack backout, a query/key gain, sliding
//! windows from a pattern ([`window_pattern_windows`]), rotary base 100000,
//! and a padded vocabulary.
//!
//! The KV cache is injected state: the model never owns one.
//! The model (and its configs) implement
//! [`KVCacheModel`](crate::ops::transformers::attention::KVCacheModel):
//! [`new_kv_cache`](crate::ops::transformers::attention::KVCacheModel::new_kv_cache)
//! builds a [`KVCache`](crate::ops::transformers::attention::KVCache) sized
//! for the model (`init_seq_len` positions up front, capped at the rotary
//! table's `max_seq_len`), and the caller passes it to each `forward` of one
//! decode. Every
//! layer appends its keys and values to it, and a step's positions start
//! where the cache's position stands, so one model can run several decodes
//! at once. Training passes no cache, and runs the whole sequence in one
//! call. A smeared model's step past the first position also reads the token
//! before it, which the caller passes to
//! [`forward_after`](NanoChatGpt::forward_after): the model keeps no state of
//! its own, so a decode the caller rewinds stays correct.
//!
//! # Datasets
//!
//! [`datasets`] holds the corpus nanochat trains on as
//! [`data::shards`](crate::data::shards) tables:
//! [`NANOCHAT_SHARD_SETS`](datasets::NANOCHAT_SHARD_SETS), whose one set is
//! [`FINEWEB_EDU_100B_SHUFFLE`](datasets::FINEWEB_EDU_100B_SHUFFLE), 1823
//! parquet shards pinned to one revision of their Hugging Face repository.
//! `data::shards` fetches and verifies the shards; `examples/train-chat`
//! binds the set to its data loader.
//!
//! # Example
//!
//! ```rust
//! use bunsen::{
//!     burner::module::ModuleInit,
//!     kits::gpts::nanochat::{
//!         NanoChatGpt,
//!         NanoChatGptContractConfig,
//!     },
//!     ops::transformers::attention::KVCacheModel,
//!     support::testing::cpu_device,
//! };
//! use burn::prelude::{
//!     Int,
//!     Tensor,
//! };
//!
//! let device = cpu_device();
//!
//! let gpt: NanoChatGpt = NanoChatGptContractConfig::new()
//!     .with_vocab_size(64)
//!     .with_n_layer(2)
//!     .with_n_head(2)
//!     .with_n_kv_head(2)
//!     .with_n_embed(32)
//!     .with_init_seq_len(16)
//!     .init(&device);
//!
//! // Training: the whole sequence at once, with no cache.
//! let ids = Tensor::<2, Int>::zeros([1, 8], &device);
//! assert_eq!(gpt.forward(ids, &mut None).dims(), [1, 8, 64]);
//!
//! // Decoding: the caller owns the cache, and passes it to every step.
//! let mut cache = gpt.new_kv_cache(1);
//! let prompt = Tensor::<2, Int>::zeros([1, 4], &device);
//! let _ = gpt.forward(prompt, &mut Some(&mut cache));
//! let next = Tensor::<2, Int>::zeros([1, 1], &device);
//! assert_eq!(gpt.forward(next, &mut Some(&mut cache)).dims(), [1, 1, 64]);
//! ```

pub mod blocks;
pub mod datasets;

#[cfg(feature = "store_pytorch")]
pub mod weights;

#[cfg(all(test, feature = "store_pytorch", feature = "store_safetensors"))]
mod toy_tests;

pub use blocks::model::*;
