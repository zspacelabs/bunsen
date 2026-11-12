//! # Pretrained: prefabs
//!
//! What is left of the kits' model vocabulary beside the
//! [`builds`](crate::data::builds) pathway: a *prefab* is a geometry by name,
//! with no weights, what a name means as a shape before any bytes are
//! fetched, kept by a kit in a [`StaticPreFabMap`]. A build states the
//! geometry it is built in, and a kit's tests check it is a prefab's.

mod prefabs;

pub use prefabs::*;
