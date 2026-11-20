//! ZONOS3-Mini's stock voices: each its reference codes and speaker
//! embedding, and what the release says of it.
//!
//! The release ships them as `Zyphra/ZONOS3-Mini-hf`'s `voices.safetensors`
//! and `voices.json`, which
//! [`interop::zonos`](super::interop::zonos) reads.

use alloc::vec::Vec;

use crate::kits::speech::tts::driver::TtsVoiceInfo;

/// One stock voice.
#[derive(Clone, Debug, PartialEq)]
pub struct Zonos3Voice {
    /// Its reference codes, `[F][n]`.
    pub codes: Vec<Vec<u32>>,

    /// Its speaker embedding, `[S]`.
    pub embedding: Vec<f32>,

    /// What the release says of it, when it says: its gender, its style,
    /// and its language as the release names it (`English`), as its
    /// description.
    pub info: Option<TtsVoiceInfo>,
}
