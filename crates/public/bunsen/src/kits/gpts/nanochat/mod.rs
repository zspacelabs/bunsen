//! # `NanoChat`
//!
//! A port of the GPT in karpathy's
//! [nanochat](https://github.com/karpathy/nanochat) (`nanochat/gpt.py`):
//! a decoder-only transformer with rotary embeddings, grouped-query
//! attention, a squared-ReLU MLP, and softcapped logits. With it comes the
//! corpus nanochat trains on, as a shard-set table.
//!
//! This is a work in progress. The model trains from scratch
//! (`examples/train-chat`); there are no pretrained weights, nothing reads
//! upstream's checkpoints, and the model departs from upstream in ways
//! listed under [known issues](#known-issues).
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
//!    one [`NanoChatGptBlockConfig`](blocks::NanoChatGptBlockConfig) per layer,
//!    the head, and the rotary-embedding table.
//! 3. `init` builds the [`NanoChatGpt`]: from the structure, or from the policy
//!    in one step through the blanket
//!    [`ModuleInit`](crate::burner::module::ModuleInit).
//! 4. [`forward`](NanoChatGpt::forward) maps `[batch, seq]` token ids to
//!    `[batch, seq, vocab_size]` logits.
//!
//! [`NanoChatGptMeta`] reads the geometry back from all three forms.
//!
//! The KV cache is injected state: the model never owns one.
//! [`new_kv_cache`](NanoChatGpt::new_kv_cache) builds a
//! [`KVCache`](crate::ops::transformers::attention::KVCache) sized for the
//! model, and the caller passes it to each `forward` of one decode. Every
//! layer appends its keys and values to it, and a step's positions start
//! where the cache's position stands, so one model can run several decodes
//! at once. Training passes no cache, and runs the whole sequence in one
//! call.
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
//! # Known issues
//!
//! - **No embedding norm.** Upstream normalizes the token embeddings before the
//!   first block, so its residual stream starts normalized. [`NanoChatGpt`]
//!   passes the raw embeddings to the first block, which normalizes only its
//!   sub-layers' inputs.
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
//!     support::testing::{
//!         CpuBackend,
//!         default_device,
//!     },
//! };
//! use burn::prelude::{
//!     Int,
//!     Tensor,
//! };
//!
//! type B = CpuBackend;
//! let device = default_device();
//!
//! let gpt: NanoChatGpt<B> = NanoChatGptContractConfig::new()
//!     .with_vocab_size(64)
//!     .with_n_layer(2)
//!     .with_n_head(2)
//!     .with_n_kv_head(2)
//!     .with_n_embed(32)
//!     .with_init_seq_len(16)
//!     .init(&device);
//!
//! // Training: the whole sequence at once, with no cache.
//! let ids = Tensor::<B, 2, Int>::zeros([1, 8], &device);
//! assert_eq!(gpt.forward(ids, &mut None).dims(), [1, 8, 64]);
//!
//! // Decoding: the caller owns the cache, and passes it to every step.
//! let mut cache = gpt.new_kv_cache(1);
//! let prompt = Tensor::<B, 2, Int>::zeros([1, 4], &device);
//! let _ = gpt.forward(prompt, &mut Some(&mut cache));
//! let next = Tensor::<B, 2, Int>::zeros([1, 1], &device);
//! assert_eq!(gpt.forward(next, &mut Some(&mut cache)).dims(), [1, 1, 64]);
//! ```

pub mod blocks;
pub mod datasets;

pub use blocks::model::*;
