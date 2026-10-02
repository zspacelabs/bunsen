//! The Whisper context driver.
//!
//! What a caller touches lives here: the driver and its config, the stream
//! context, the emission policy and what it emits, the clock, the clamp
//! policies, the voice-activity filter config, the token layout and the
//! detokenizer. [`support`] holds the internals underneath, and nothing in
//! it is part of the driver's API.

pub mod support;

mod emission_policy;
mod speech_region;
mod stream_clamp_policy;
mod stream_clock;
mod transcript;
mod voice_activity_filter;
mod whisper_bundle;
mod whisper_stream_context;
mod whisper_stream_driver;
mod whisper_token_layout;

pub use emission_policy::*;
pub use speech_region::*;
pub use stream_clamp_policy::*;
pub use stream_clock::*;
pub use transcript::*;
pub use voice_activity_filter::*;
pub use whisper_bundle::*;
pub use whisper_stream_context::*;
pub use whisper_stream_driver::*;
pub use whisper_token_layout::*;
