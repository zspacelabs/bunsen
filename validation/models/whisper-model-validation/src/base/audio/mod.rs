//! End-to-end validation over the committed speech fixtures.
//!
//! [`staged`](super::staged) feeds each stage synthetic input, which isolates
//! it but says nothing about the pipeline a user drives. This runs the whole
//! thing — mp3, mel front end, encoder, KV-cached decoder, greedy search —
//! over the clips `bunsen-test-audio` commits, against their `openai-whisper`
//! references in `testdata/`, and judges it the way a transcription is
//! judged: **word error rate against a ground-truth transcript**.
//!
//! ## What is asserted
//!
//! The transcript is the authority. Every gate is on text, and every gate is
//! tunable per fixture:
//!
//! * **Accuracy** — WER against `{name}.txt` must be at or under
//!   [`Fixture::max_wer`]. This is what says the pipeline works.
//! * **Agreement** — WER against `{name}.reference.json`, what `openai-whisper`
//!   itself decodes, must be at or under [`Fixture::max_reference_wer`]. This
//!   is what says bunsen agrees with the implementation it transliterates.
//!
//! Nothing compares token ids. A greedy decode argmaxes over 51865 logits at
//! every step, so a backend differing in the last few digits can flip a token
//! and cascade — while the *text* barely moves.
//!
//! ## What runs when
//!
//! The fixture-integrity checks need no model at all and so run in an ordinary
//! `cargo test`. Everything that loads weights is behind `download`, which is
//! also what makes `build.rs` fetch the checkpoint — so within this crate a
//! gated test never has to skip.

/// Everything that loads weights, which the tests fetch (`checkpoints`).
#[cfg(feature = "checkpoints")]
mod download;
#[cfg(feature = "gpu-tests")]
mod gpu_tests;

mod reference;

pub use reference::*;

#[cfg(test)]
mod tests;
