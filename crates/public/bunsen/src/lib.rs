#![cfg_attr(feature = "wgpu", recursion_limit = "512")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
//! # bunsen
//!
//! A "batteries included" community standard library for the
//! [burn](https://burn.dev) tensor framework: reusable modules, tensor
//! operations, shape contracts, pretrained-model machinery, and the
//! module-lifecycle utilities that fall outside burn's core.
//!
//! These API docs are the reference for every item. The
//! [bunsen book](https://zspacelabs.ai/bunsen/book/) explains how the crate is
//! organized, the systems that cut across modules, and how bunsen is
//! developed.
//!
//! ## Organization
//!
//! The crate is layered: a module builds on the layers below it, never on
//! those above.
//!
//! ```text
//!   kits        whole models and simulations
//!   blocks      torch.nn-like components: module roots and tree parts
//!   ops         operations; Module/Config only to hold cached tables or state
//!   foundation  burner, contracts, errors, rust_ext, support, zspace
//!   ───────────────────────────────────────────────────────────────────────
//!   burn
//!
//!   beside the stack: data (kits load through it), audit
//! ```
//!
//! The `ops` / `blocks` split is a rule, stated in [`ops`](ops#ops-and-blocks).
//!
//! ### Components
//!
//! * [`kits`]: complete models and simulations: image models, GPTs,
//!   simulations, speech, and the token seam the speech kits share.
//! * [`blocks`]: reusable [`Module`](burn::module::Module) components: conv
//!   blocks, image layers, recurrent cells, transformer parts.
//! * [`ops`]: tensor operations, including the signal-processing front ends,
//!   attention ops, and the KV cache.
//!
//! ### Foundation
//!
//! * [`burner`]: extensions to burn's module machinery: the config lifecycle
//!   ([`ModuleInit`](burner::module::ModuleInit),
//!   [`ToStructureConfig`](burner::module::ToStructureConfig)), module
//!   [`reflection`](burner::module::reflection), grouped optimizers
//!   ([`optim`](burner::optim)), `Tensor` extension traits
//!   ([`tensor`](burner::tensor)), and PyTorch-load repairs
//!   ([`store`](burner::store)).
//! * [`contracts`]: runtime tensor-shape contracts.
//! * [`errors`]: `BunsenError`, `BunsenResult`, and the `try_x` / `x`
//!   convention.
//! * [`rust_ext`]: Rust-language extensions with no tensor or burn dependency:
//!   [`CloneBox`](rust_ext::CloneBox) / [`CloneRef`](rust_ext::CloneRef), array
//!   and range helpers, and
//!   [`LocationDesc`](rust_ext::reflection::LocationDesc), a serializable
//!   source location.
//! * [`support`]: shared utilities, including the test backends and devices
//!   (`support::testing`, feature `testing`).
//! * [`zspace`]: integer-lattice index and shape helpers.
//!
//! ### Beside the stack
//!
//! * [`data`]: the disk cache, pretrained-model loading, and dataset shard
//!   sets.
#![cfg_attr(
    feature = "audit",
    doc = "* [`audit`]: audit probes: record a stream of tensor checkpoints from one run,"
)]
#![cfg_attr(
    not(feature = "audit"),
    doc = "* `audit` (feature `audit`): audit probes: record a stream of tensor checkpoints from one run,"
)]
//!   and verify another run (on another backend, or against a stored
//!   baseline) against it.
//!
//! ### Entry points
//!
//! * [`prelude`]: the traits most code needs in scope, for one glob import.
//! * [`public`]: re-exports of public dependencies (`burn`), so downstream code
//!   uses the same versions.
//!
//! ### Sibling crates
//!
//! `bunsen-firehose` and `bunsen-firehose-image` (a data-processing pipeline),
//! `bunsen-arrow-dataloaders` (preview: Parquet shards to training batches),
//! `bunsen-bundled-silero` and `bunsen-bundled-whisper` (model assets behind
//! the `*-weights` features), and `bunsen-contracts-macros` (the proc macro
//! behind [`contracts`]). The book's
//! [workspace](https://zspacelabs.ai/bunsen/book/organization/workspace.html)
//! page maps them.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

extern crate alloc;
extern crate core;

// Make the macro targets public.
// TODO: re-examine contracts publication.
#[doc(hidden)]
pub use bunsen_contracts_macros::shape_contract as __proc_shape_contract;

/// Re-exports public dependencies.
#[allow(unused_imports)]
#[allow(missing_docs)]
pub mod public {
    pub use burn;
    #[cfg(feature = "train")]
    pub use hashbrown;
}

#[cfg(feature = "audit")]
pub mod audit;
pub mod blocks;
pub mod burner;
pub mod contracts;
pub mod data;
pub mod errors;
pub mod kits;
pub mod ops;
pub mod prelude;
pub mod rust_ext;
pub mod support;
pub mod zspace;
