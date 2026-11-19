//! # The driver
//!
//! [`Zonos3Driver`]: normalized text and a speaker to speech, one take
//! decoded whole, as the release generates it.
//!
//! - [`Zonos3TakeRows`]: a take's generated rows, the stop rule (the end of
//!   audio, or the budget) and the frames a whole decode covers.
//! - [`zonos3_output_stage`]: the release's gain and soft limiter on the
//!   decoder's audio.
//!
//! A request's text is normalized by a
//! [`TextNormalizer`](crate::kits::support::text::TextNormalizer) of the
//! config's: by default the release's, Zyphra's normalizer (feature
//! `zyphra-textnorm`).

mod generate;
mod output;
mod take;

pub use generate::*;
pub use output::*;
pub use take::*;

#[cfg(test)]
mod toy_tests;
