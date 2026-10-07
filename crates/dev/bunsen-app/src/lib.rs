//! Shared command-line plumbing for bunsen's example binaries.
//!
//! - [`device`]: `--device` and precision flags, resolved into a configured
//!   burn `Device`.
//! - [`logging`]: log-level flags.
//! - [`shards`]: dataset shard selection and fetch policy.
//!
//! An example's backend features forward here (`wgpu = ["bunsen-app/wgpu"]`),
//! as does `tui` for the training examples.
pub mod device;
pub mod logging;
pub mod shards;
