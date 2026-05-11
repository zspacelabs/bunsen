//! Burn Extension/Overlay Utilities
//!
//! This module contains utilities which are direct extensions
//! or overlay/replacements for existing kit functionality.

pub mod descriptors;
pub mod distribution;
pub mod module;
pub mod record;

#[cfg(feature = "train")]
pub mod optim;
