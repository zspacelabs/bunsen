//! The Whisper context driver.
//!
//! What a caller touches lives here: the driver and its config, the stream
//! context, the emission policy and what it emits, the clock, the clamp
//! policies, the voice-activity filter config, the token layout and the
//! detokenizer. [`support`] holds the internals underneath, and nothing in
//! it is part of the driver's API.

pub mod support;

mod clamp;
mod clock;
mod context;
mod driver_impl;
mod emission;
mod mel;
mod text;
mod tokens;
mod va_filter;
mod vocab;

#[doc(inline)]
pub use clamp::*;
#[doc(inline)]
pub use clock::*;
#[doc(inline)]
pub use context::*;
#[doc(inline)]
pub use driver_impl::*;
#[doc(inline)]
pub use emission::*;
#[doc(inline)]
pub use mel::*;
#[doc(inline)]
pub use text::*;
#[doc(inline)]
pub use tokens::*;
#[doc(inline)]
pub use va_filter::*;
#[doc(inline)]
pub use vocab::*;
