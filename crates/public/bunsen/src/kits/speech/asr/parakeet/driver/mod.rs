//! # The the TDT transducer stream driver
//!
//! The [`asr_driver`](crate::kits::speech::asr::driver) core's
//! [`AsrStreamDriver`](crate::kits::speech::asr::driver::AsrStreamDriver) and
//! [`AsrStreamContext`](crate::kits::speech::asr::driver::AsrStreamContext)
//! over a [`ParakeetEngine`]: [`ParakeetStreamDriver`] and
//! [`ParakeetStreamContext`].

mod engine;

pub use engine::*;
