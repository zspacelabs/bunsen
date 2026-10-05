//! # Mathematical utilities.

mod iroot;
mod nan_utils;

pub use iroot::*;
pub use nan_utils::*;

/// `1.0 / 3.0.sqrt()`.
// TODO: use `f64::consts::FRAC_1_SQRT_3` once it is stable.
pub const FRAC_1_SQRT_3: f64 = 0.577_350_269_189_625_7_f64;
