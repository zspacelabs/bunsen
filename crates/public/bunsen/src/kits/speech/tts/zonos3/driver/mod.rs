//! # The driver
//!
//! [`Zonos3Driver`]: normalized text and a speaker to speech, one take
//! decoded whole, as the release generates it.
//!
//! - [`Zonos3TakeRows`]: a take's generated rows, the stop rule (the end of
//!   audio, or the budget) and the frames a whole decode covers.
//! - [`zonos3_output_stage`]: the release's gain and soft limiter on the
//!   decoder's audio.
//! - [`Zonos3TextNormalization`]: how a request's text is normalized, as typed
//!   or by Zyphra's normalizer (feature `zyphra-textnorm`);
//!   [`Zonos3Normalizer`] applies it.

mod generate;
mod normalize;
mod output;
mod take;

pub use generate::*;
pub use normalize::*;
pub use output::*;
pub use take::*;

#[cfg(test)]
mod toy_tests;
