//! # Math operations
//!
//! [`LogBase`] names the base of a logarithm (ten, `e`, or any other) as a
//! value object, and [`LogBase::apply`] takes that log elementwise. burn's
//! `Tensor` has only the natural log ([`Tensor::log`]); other bases are
//! `ln(x) / ln(base)`.
//!
//! It exists so a config can carry the choice: the log-mel front end in
//! [`crate::ops::signal::perceptive_audio`] takes `log10` for Whisper /
//! `librosa` and the natural log for Kaldi-style features.
//!
//! [`Tensor::log`]: burn::Tensor::log

mod log;

pub use log::*;
