//! # Timing: one surface for every architecture's decode
//!
//! Not a model: what a throughput harness drives, so it knows no kit.
//! [`TimedDecode`] is a decode reduced to its phases (a state made for its
//! rows and positions, an encoder's pass where the model has one, a prompt
//! prefilled, then steps), each on the model's
//! own synthetic input ([`synthetic_ids`] for a token decode), each returning
//! what a real decode reads back, so a timing window can close on it.
//!
//! Every [`CachedLm`](crate::kits::language::generation::driver::CachedLm) is
//! one by a blanket impl (the language models: Qwen3, Qwen3.5, Gemma,
//! nanochat); a decode off that surface implements it beside its own loop
//! (Whisper's).
//! The timing itself is burn's: a harness opens
//! [`Device::profile`](burn::tensor::Device::profile) windows around the
//! phases.
mod timed_decode;

pub use timed_decode::*;
