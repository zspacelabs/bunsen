#![allow(unused)]
//! # Utility Support Functions
//!
//! This module exists to support developing `bimm` modules.
//! The API stability expectations are lower than for [`crate::layers`]
//! or [`crate::models`]; but it is not meant to be experimental code.

pub mod burn;
pub mod probability;

mod with_ok_or_panic;
#[doc(inline)]
pub use with_ok_or_panic::*;
