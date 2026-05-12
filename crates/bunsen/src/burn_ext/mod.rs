//! Burn Extension/Overlay Utilities
//!
//! This module contains utilities which are direct extensions
//! or overlay/replacements for existing `burn_ext` functionality.

pub mod descriptors;
pub mod distribution;
pub mod module;
pub mod record;

#[cfg(feature = "train")]
pub mod optim;
