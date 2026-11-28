//! # Swin as timm publishes it
//!
//! timm's `SwinTransformer` (V1) and `SwinTransformerV2` state dicts, each
//! a hub repo's `model.safetensors`, with the patch merges at the start of
//! the stages they open:
//!
//! - **Names:** [`swin_row_splits`] (the fused `qkv`, split) and
//!   [`swin_rekey`], onto the
//!   [`StagedVisionEncoderModel`](crate::blocks::transformers::models::staged_vision_encoder::StagedVisionEncoderModel)'s
//!   canonical names.
//! - **Loading** (feature `store`): `load_swin_tensors`; with `cache`,
//!   `load_swin`, a resource set's checkpoint.

#[cfg(feature = "store")]
mod weights;
#[cfg(feature = "store")]
pub use weights::*;

mod config;
mod rekey;

pub(crate) use config::*;
pub use rekey::*;

#[cfg(test)]
mod tests;
