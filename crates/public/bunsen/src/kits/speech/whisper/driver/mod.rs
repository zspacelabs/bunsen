//! The Whisper context driver.

pub mod support;

mod context;
mod driver_impl;

#[doc(inline)]
pub use context::*;
#[doc(inline)]
pub use driver_impl::*;

/// The sample rate Whisper's front end is defined at, in Hz.
/// TODO: This should not be hardcoded.
pub const SAMPLE_RATE: usize = 16_000;
