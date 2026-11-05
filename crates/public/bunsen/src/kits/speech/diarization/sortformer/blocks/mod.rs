//! Sortformer component blocks: the model, its configs, and its head.
//!
//! [`SortformerStructureConfig`] builds a [`Sortformer`]: a
//! [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)
//! (pre-encode, then the encoder layers: feature stacking and a rotary
//! encoder for Nemotron, the `FastConformer` for v2), a projection,
//! an optional
//! [`ResidualEncoderTransformer`](crate::blocks::transformers::encoders::residual::ResidualEncoderTransformer)
//! and sub-pixel upsample, and a
//! [`SortformerHead`]. A policy lowers to it: [`NemotronDiarizationConfig`],
//! [`SortformerV2Config`]. All answer [`SortformerMeta`].

mod head;
mod model;

pub use head::*;
pub use model::*;
