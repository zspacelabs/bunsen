//! # The engine and its driver
//!
//! [`Zonos3Engine`]: text and a speaker to codes, and codes to speech, one
//! take decoded whole, as the release generates it; the model's half of the
//! [TTS driver](crate::kits::speech::tts::driver). [`Zonos3Driver`] is the
//! driver over it ([`zonos3_driver`]).
//!
//! - [`Zonos3TakeRows`]: a take's generated rows, the stop rule (the end of
//!   audio, or the budget) and the frames a whole decode covers.
//! - [`zonos3_output_stage`]: the release's gain and soft limiter on the
//!   decoder's audio.
//!
//! A request's text is normalized by the driver: the release's
//! normalization ([`zonos3_release_normalization`]) is Zyphra's normalizer
//! (feature `zyphra-textnorm`).

mod engine;
mod generate;
mod output;
mod take;

pub use generate::*;
pub use output::*;
pub use take::*;

#[cfg(test)]
mod toy_tests;
