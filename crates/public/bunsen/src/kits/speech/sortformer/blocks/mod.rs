//! Sortformer component blocks: the model, its configs, its encoder and its
//! head.
//!
//! [`SortformerStructureConfig`] builds a [`Sortformer`]: a
//! [`SortformerEncoder`] (pre-encode, then the encoder layers), a projection,
//! an optional
//! [`ResidualEncoderTransformer`](crate::blocks::transformers::encoders::residual::ResidualEncoderTransformer)
//! and sub-pixel upsample, and a
//! [`SortformerHead`]. A policy lowers to it: [`NemotronDiarizationConfig`],
//! [`SortformerV2Config`]. All answer [`SortformerMeta`].

mod encoder;
mod head;
mod model;

pub use encoder::*;
pub use head::*;
pub use model::*;
