//!# bimm - Burn Image Models
//!
//! ## Notable Components
//!
//! * [`cache`] - weight loading cache.
//! * [`compat`] - compat code, ported or planned for an upcoming release of
//!   ``burn``.
//! * [`models`] - complete model families.
//!   * [`models::resnet`] - `ResNet`
//!   * [`imgs::swin`] - The SWIN Family.
//!     * [`imgs::swin::v2`] - The SWIN-V2 Model.
#![warn(missing_docs)]

extern crate alloc;

extern crate core;
/// Test-only macro import.
#[cfg(test)]
#[allow(unused_imports)]
#[macro_use]
extern crate hamcrest;

pub mod cache;
pub mod models;
