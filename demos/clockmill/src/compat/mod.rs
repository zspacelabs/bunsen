//! # Compat Mechanisms for upcoming Burn API Changes
pub mod data_view;
pub mod operations;
pub mod shape;

/// `1.0 / (3.0).sqrt()`
/// TODO: unstable feature: `f64::FRAC_1_SQRT_3`
pub const FRAC_1_SQRT_3: f64 = 0.577_350_269_189_625_7_f64;
