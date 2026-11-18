//! Shared command-line plumbing for bunsen's example binaries.
//!
//! - [`cache`]: the weights cache's flags, and its `path`, `list` and `clear`
//!   commands.
//! - [`catalog`]: the build catalogs' commands: import catalogs, then dump,
//!   browse, show, resolve and select over them.
//! - [`device`]: `--device` and `--device-index`, resolved into a configured
//!   burn `Device` (its default float f32).
//! - [`logging`]: log-level flags.
//! - [`precision`]: `--min-precision`, a floor under a loaded model's dtypes,
//!   and `--init-dtype`, the dtype a program creates its own tensors in.
//! - [`shards`]: dataset shard selection and fetch policy.
//!
//! An example's backend features forward here (`wgpu = ["bunsen-app/wgpu"]`),
//! as does `tui` for the training examples.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

pub mod cache;
pub mod catalog;
pub mod device;
pub mod logging;
pub mod precision;
pub mod shards;
