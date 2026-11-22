//! [`TtsVoicePrint`]: a voice built for one model, and its file.

use alloc::{
    collections::BTreeMap,
    format,
    string::String,
    vec::Vec,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The version of the voice file layout this reads and writes.
pub const TTS_VOICE_FILE_VERSION: u32 = 1;

/// The voice file's header keys, in its safetensors `__metadata__`: the
/// family, the encoder build, the mode, the reference's digest and the
/// version. A voice file holds no other metadata.
pub const TTS_VOICE_HEADER_KEYS: [&str; 5] = [
    "bunsen.voice.family",
    "bunsen.voice.encoder",
    "bunsen.voice.mode",
    "bunsen.voice.reference",
    "bunsen.voice.version",
];

/// What made a voice file: the model family, the encoder build (a
/// checkpoint's row, its pin and its encoder), the mode, and the reference it
/// was built from. A voice speaks only in the build that made it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtsVoiceIdentity {
    /// The model family (`zonos3`).
    pub family: String,

    /// The encoder build that made it.
    pub encoder: String,

    /// The mode it was made in (`embedding`).
    pub mode: String,

    /// The digest of the reference it was built from
    /// ([`TtsVoiceReference::digest`](super::TtsVoiceReference::digest)).
    pub reference: String,
}

impl TtsVoiceIdentity {
    /// Refuses a voice another build made.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) naming what made
    /// the voice and what `family`'s `encoder` takes.
    pub fn check(
        &self,
        family: &str,
        encoder: &str,
    ) -> BunsenResult<()> {
        if self.family != family || self.encoder != encoder {
            return Err(BunsenError::illegal(format!(
                "a voice made by {} {} (mode {}); this model speaks in voices of {family} \
                 {encoder}: build it again from its reference, for this model",
                self.family, self.encoder, self.mode
            )));
        }
        Ok(())
    }
}

/// One tensor of a voice file.
#[derive(Clone, Debug, PartialEq)]
pub enum TtsVoiceTensor {
    /// Floats.
    F32 {
        /// Its shape.
        shape: Vec<usize>,

        /// Its values, row-major.
        values: Vec<f32>,
    },

    /// Integers (codes, token ids).
    I32 {
        /// Its shape.
        shape: Vec<usize>,

        /// Its values, row-major.
        values: Vec<i32>,
    },
}

impl TtsVoiceTensor {
    /// Its shape.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32 { shape, .. } | Self::I32 { shape, .. } => shape,
        }
    }
}

/// A voice built for one model: tensors, which the engine that made it
/// defines and reads, and the identity that names that engine's build
/// ([`TtsVoiceIdentity`]).
///
/// Made by [`TtsEngine::build_voice`](super::TtsEngine::build_voice) from a
/// [`TtsVoiceReference`](super::TtsVoiceReference). Its file (feature `store`,
/// [`interop::safetensors`](super::interop::safetensors)) is one safetensors
/// file: the tensors, and the identity as the header's only
/// metadata ([`TTS_VOICE_HEADER_KEYS`]). Everything a person reads of the
/// voice is in its reference.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsVoicePrint {
    /// What made it.
    pub identity: TtsVoiceIdentity,

    /// The tensors, by name.
    pub tensors: BTreeMap<String, TtsVoiceTensor>,
}

impl TtsVoicePrint {
    /// The tensor `name`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// naming the tensor and the ones there are.
    pub fn tensor(
        &self,
        name: &str,
    ) -> BunsenResult<&TtsVoiceTensor> {
        self.tensors.get(name).ok_or_else(|| {
            BunsenError::invalid_resource(format!(
                "a voice with no `{name}` tensor (it has {:?})",
                self.tensors.keys().collect::<Vec<_>>()
            ))
        })
    }

    /// The float vector `name`, of `len` values.
    ///
    /// # Errors
    /// As [`tensor`](Self::tensor), and
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for one that is not floats of shape `[len]`.
    pub fn f32_vector(
        &self,
        name: &str,
        len: usize,
    ) -> BunsenResult<&[f32]> {
        match self.tensor(name)? {
            TtsVoiceTensor::F32 { shape, values } if shape.as_slice() == [len] => Ok(values),
            other => Err(BunsenError::invalid_resource(format!(
                "a voice's `{name}`: {other:?}; it is f32 [{len}]"
            ))),
        }
    }
}
