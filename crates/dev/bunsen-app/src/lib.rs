//! Shared command-line plumbing for bunsen's example binaries.
//!
//! - [`cache`]: the weights cache's flags, and its `path`, `list` and `clear`
//!   commands.
//! - [`device`]: `--device` and precision flags, resolved into a configured
//!   burn `Device`.
//! - [`logging`]: log-level flags.
//! - [`shards`]: dataset shard selection and fetch policy.
//!
//! An example's backend features forward here (`wgpu = ["bunsen-app/wgpu"]`),
//! as does `tui` for the training examples.
pub mod cache;
pub mod device;
pub mod logging;
pub mod shards;
