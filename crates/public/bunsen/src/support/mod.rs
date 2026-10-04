#![allow(unused)]
//! # Support utilities
//!
//! Small helpers that the rest of bunsen is built from, and the test kit
//! that bunsen's own tests use and that tests of code built on bunsen can use
//! too. Nothing here is a model, an op or a block.
//!
//! Rust-language helpers with no tensor or burn dependency go in
//! [`rust_ext`](crate::rust_ext) instead.
//!
//! ## The submodules
//!
//! Two kinds live here. Some are **for users**: the test kit, audio loading,
//! and validators. The rest are **plumbing**: public so that bunsen's other
//! modules can share them, and shaped by those modules' needs rather than
//! designed as library API.
//!
//! | Module | What it holds | For |
//! |--------|---------------|-----|
//! | `testing` | Test backends and devices, seeded inputs, device-memory hygiene, float assertions; `testing::asr` scores transcripts. Needs the `testing` feature. | users' tests |
//! | `audio` | `load_audio_mono_sr`: decode a mono WAV or mp3 at a required sample rate, without resampling. Needs the `audio` feature. | users (speech kits) |
//! | [`validators`] | [`try_probability`](validators::try_probability) / [`expect_probability`](validators::expect_probability), a `try_x` / `x` pair of the [errors convention]. | users |
//! | [`geometry`] | [`GridShape2D`](geometry::GridShape2D), a `W,H` grid size that parses from a string; the sims kits' configs use it. | users, through the sims kits |
//! | [`math`] | [`maybe_iroot`](math::maybe_iroot), the exact integer root that shape contracts solve with; [`nan_to_num`](math::nan_to_num); [`FRAC_1_SQRT_3`](math::FRAC_1_SQRT_3). | plumbing |
//!
//! `testing` and `audio` are listed without links because they exist only
//! with their features; when they are on, rustdoc lists them under
//! "Modules" below. The `testing` feature turns on `audio`.
//!
//! [errors convention]: crate::errors#convention-try_x-and-x

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub mod geometry;
pub mod math;
pub mod validators;
