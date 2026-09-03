//! The Whisper context driver.
//!
//! What a caller touches lives here: the driver and its config, the stream
//! context, the emission policy and what it emits, the clock, the clamp
//! policies, the voice-activity filter config, the token layout and the
//! detokenizer. [`support`] holds the internals underneath, and nothing in
//! it is part of the driver's API.

pub mod support;

mod stream_clamp_policy;
mod stream_clock;
mod voice_activity_filter;
mod whisper_emission;
mod whisper_stream_context;
mod whisper_stream_driver;
mod whisper_token_layout;

#[doc(inline)]
pub use stream_clamp_policy::*;
#[doc(inline)]
pub use stream_clock::*;
#[doc(inline)]
pub use voice_activity_filter::*;
#[doc(inline)]
pub use whisper_emission::*;
#[doc(inline)]
pub use whisper_stream_context::*;
#[doc(inline)]
pub use whisper_stream_driver::*;
#[doc(inline)]
pub use whisper_token_layout::*;
