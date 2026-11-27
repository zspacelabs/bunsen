//! Image classification on bunsen's image kits ([`bunsen::kits::images`]), as
//! `bunsen-tool classify` runs it.
//!
//! - [`ClassifyCommandGroup`]: `classify <run|list|fetch>`: image files to
//!   their top classes, and the checkpoints.
//! - [`load_image_model`]: a checkpoint by name, as an
//!   [`ImageClassifier`](bunsen::kits::images::classification::driver::ImageClassifier)
//!   and its card, whatever its kit; `embed` loads through it too.
//!   [`fetch_image_model`] fetches one's files.

mod classify_command_group;
mod image_model;

pub use classify_command_group::*;
pub use image_model::*;
