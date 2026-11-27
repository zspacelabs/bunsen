//! # The image classification kits' pretrained pathways, as one table
//!
//! Every kit's factory is an [`ImageKitFactory`], type-erased:
//! [`image_kit_factories`] lists them, and [`image_kit_for`] picks the one a
//! checkpoint name loads through. A tool that loads any image kit's model
//! (`bunsen-tool classify`, the validation crate) drives from that table, so
//! a new kit is one entry in it.
//!
//! The table needs the `store` feature.

#[cfg(feature = "store")]
mod kit_table;

#[cfg(feature = "store")]
pub use kit_table::*;
