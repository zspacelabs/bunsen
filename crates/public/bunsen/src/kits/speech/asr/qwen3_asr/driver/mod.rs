//! # The Qwen3-ASR stream driver
//!
//! The [`asr_driver`](crate::kits::speech::asr::driver) core's
//! [`AsrStreamDriver`](crate::kits::speech::asr::driver::AsrStreamDriver) and
//! [`AsrStreamContext`](crate::kits::speech::asr::driver::AsrStreamContext)
//! over a [`Qwen3AsrEngine`]: [`Qwen3AsrStreamDriver`] and
//! [`Qwen3AsrStreamContext`].

mod engine;

pub use engine::*;
