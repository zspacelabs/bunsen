//! Internals of the Whisper driver: the speech regions a voice-activity
//! filter produces and the segment splitting of a decoded window. Nothing
//! here is part of the driver's API.

pub(crate) mod segments;
mod util;

#[doc(inline)]
pub use util::*;
