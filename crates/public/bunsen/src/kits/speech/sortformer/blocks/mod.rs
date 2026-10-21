//! Sortformer component blocks: the model, its configs, its encoder and its
//! head.
//!
//! [`SortformerStructureConfig`] builds a [`Sortformer`]: a
//! [`SortformerEncoder`] (pre-encode, then the encoder layers), a projection,
//! an optional [`SortformerTransformer`] and sub-pixel upsample, and a
//! [`SortformerHead`]. A policy lowers to it: [`NemotronDiarizationConfig`],
//! [`SortformerV2Config`]. All answer [`SortformerMeta`].

mod encoder;
mod head;
mod model;
mod transformer;

pub use encoder::*;
pub use head::*;
pub use model::*;
pub use transformer::*;
