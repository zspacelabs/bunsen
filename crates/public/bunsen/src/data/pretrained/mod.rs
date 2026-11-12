//! # Pretrained: prefabs and tags
//!
//! What is left of the kits' model vocabulary beside the
//! [`builds`](crate::data::builds) pathway: a *prefab* is a geometry by name,
//! with no weights, what a name means as a shape before any bytes are
//! fetched, kept by a kit in a [`StaticPreFabMap`]; and [`tags`] are the
//! facts about a model that listings show.

mod prefabs;
pub mod tags;

pub use prefabs::*;
pub use tags::Tags;
