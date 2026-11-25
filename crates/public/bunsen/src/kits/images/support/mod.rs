//! # Shared parts of the image kits
//!
//! What the image kits share across their tasks, at the area's lowest
//! covering level (the [kits layout](crate::kits)):
//!
//! - [`transform`]: an image model's eval transform,
//!   [`ImageEvalTransformConfig`](transform::ImageEvalTransformConfig), with
//!   timm's settings and size arithmetic, and the tensor half that crops and
//!   normalises.

pub mod transform;
