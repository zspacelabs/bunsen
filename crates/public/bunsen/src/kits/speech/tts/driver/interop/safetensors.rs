//! The voice file: a [`TtsVoicePrint`] as one safetensors file, its tensors
//! and its identity, the header's only metadata
//! ([`TTS_VOICE_HEADER_KEYS`]).

use alloc::{
    string::{
        String,
        ToString,
    },
    vec::Vec,
};
use std::{
    collections::HashMap,
    path::Path,
};

use safetensors::tensor::{
    Dtype,
    SafeTensors,
    TensorView,
    serialize,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
    kits::speech::tts::driver::{
        TTS_VOICE_FILE_VERSION,
        TTS_VOICE_HEADER_KEYS,
        TtsVoiceIdentity,
        TtsVoicePrint,
        TtsVoiceTensor,
    },
};

fn header(identity: &TtsVoiceIdentity) -> HashMap<String, String> {
    let values = [
        identity.family.clone(),
        identity.encoder.clone(),
        identity.mode.clone(),
        identity.reference.clone(),
        TTS_VOICE_FILE_VERSION.to_string(),
    ];
    TTS_VOICE_HEADER_KEYS
        .iter()
        .map(|k| k.to_string())
        .zip(values)
        .collect()
}

fn bad(what: impl core::fmt::Display) -> BunsenError {
    BunsenError::invalid_resource(format!("a voice file: {what}"))
}

impl TtsVoicePrint {
    /// The voice as a voice file's bytes.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for a tensor whose values do not fill its shape.
    pub fn to_bytes(&self) -> BunsenResult<Vec<u8>> {
        let bytes: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = self
            .tensors
            .iter()
            .map(|(name, tensor)| match tensor {
                TtsVoiceTensor::F32 { shape, values } => (
                    name.clone(),
                    Dtype::F32,
                    shape.clone(),
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
                TtsVoiceTensor::I32 { shape, values } => (
                    name.clone(),
                    Dtype::I32,
                    shape.clone(),
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            })
            .collect();
        let views = bytes
            .iter()
            .map(|(name, dtype, shape, data)| {
                TensorView::new(*dtype, shape.clone(), data)
                    .map(|view| (name.clone(), view))
                    .map_err(|e| bad(format!("tensor {name}: {e}")))
            })
            .collect::<BunsenResult<Vec<_>>>()?;
        serialize(views, Some(header(&self.identity))).map_err(bad)
    }

    /// The voice in a voice file's `bytes`.
    ///
    /// # Errors
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for bytes that are not safetensors; a header that lacks one of
    /// [`TTS_VOICE_HEADER_KEYS`], has another key, or is of another
    /// version; a tensor that is neither f32 nor i32.
    pub fn from_bytes(bytes: &[u8]) -> BunsenResult<Self> {
        let (_, metadata) = SafeTensors::read_metadata(bytes).map_err(bad)?;
        let header = metadata
            .metadata()
            .clone()
            .ok_or_else(|| bad("no header metadata"))?;
        if let Some(other) = header
            .keys()
            .find(|k| !TTS_VOICE_HEADER_KEYS.contains(&k.as_str()))
        {
            return Err(bad(format!(
                "the header key {other:?}; a voice file's are {TTS_VOICE_HEADER_KEYS:?}"
            )));
        }
        let get = |k: &str| {
            header
                .get(k)
                .cloned()
                .ok_or_else(|| bad(format!("no {k:?} in the header")))
        };
        let version = get("bunsen.voice.version")?;
        if version != TTS_VOICE_FILE_VERSION.to_string() {
            return Err(bad(format!(
                "version {version}; this reads version {TTS_VOICE_FILE_VERSION}"
            )));
        }
        let identity = TtsVoiceIdentity {
            family: get("bunsen.voice.family")?,
            encoder: get("bunsen.voice.encoder")?,
            mode: get("bunsen.voice.mode")?,
            reference: get("bunsen.voice.reference")?,
        };
        let file = SafeTensors::deserialize(bytes).map_err(bad)?;
        let tensors = file
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                let shape = view.shape().to_vec();
                let words = view.data().as_chunks::<4>().0.iter().copied();
                let tensor = match view.dtype() {
                    Dtype::F32 => TtsVoiceTensor::F32 {
                        shape,
                        values: words.map(f32::from_le_bytes).collect(),
                    },
                    Dtype::I32 => TtsVoiceTensor::I32 {
                        shape,
                        values: words.map(i32::from_le_bytes).collect(),
                    },
                    other => {
                        return Err(bad(format!(
                            "tensor {name} of {other:?}; a voice's are f32 or i32"
                        )));
                    }
                };
                Ok((name, tensor))
            })
            .collect::<BunsenResult<_>>()?;
        Ok(Self { identity, tensors })
    }

    /// Writes the voice file to `path`.
    ///
    /// # Errors
    /// As [`to_bytes`](Self::to_bytes); as [`sys_at`] sorts an io
    /// failure, naming the file.
    pub fn write(
        &self,
        path: impl AsRef<Path>,
    ) -> BunsenResult<()> {
        let path = path.as_ref();
        std::fs::write(path, self.to_bytes()?).map_err(sys_at("write", path))
    }

    /// Reads the voice file at `path`.
    ///
    /// # Errors
    /// As [`sys_at`] sorts an io failure, and as
    /// [`from_bytes`](Self::from_bytes); each names the file.
    pub fn read(path: impl AsRef<Path>) -> BunsenResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(sys_at("read", path))?;
        Self::from_bytes(&bytes).map_err(|e| e.context(path.display()))
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        collections::BTreeMap,
        vec,
    };

    use super::*;
    use crate::errors::testing::ErrorMatcher;

    fn voice() -> TtsVoicePrint {
        TtsVoicePrint {
            identity: TtsVoiceIdentity {
                family: "zonos3".into(),
                encoder: "zonos3-mini@73500c5a/ownmect_s2".into(),
                mode: "embedding".into(),
                reference: "ab12".into(),
            },
            tensors: BTreeMap::from([
                (
                    "embedding".to_string(),
                    TtsVoiceTensor::F32 {
                        shape: vec![3],
                        values: vec![0.5, -1.0, 2.0],
                    },
                ),
                (
                    "ref_codes".to_string(),
                    TtsVoiceTensor::I32 {
                        shape: vec![2, 2],
                        values: vec![1, 2, 3, 4],
                    },
                ),
            ]),
        }
    }

    /// A voice file round-trips; its header is the identity alone.
    #[test]
    fn test_round_trip() {
        let voice = voice();
        let bytes = voice.to_bytes().unwrap();
        assert_eq!(TtsVoicePrint::from_bytes(&bytes).unwrap(), voice);
        let (_, metadata) = SafeTensors::read_metadata(&bytes).unwrap();
        let mut keys: Vec<_> = metadata.metadata().clone().unwrap().into_keys().collect();
        keys.sort();
        let mut expected = TTS_VOICE_HEADER_KEYS.to_vec();
        expected.sort();
        assert_eq!(keys, expected);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Ada.voice.safetensors");
        voice.write(&path).unwrap();
        assert_eq!(TtsVoicePrint::read(&path).unwrap(), voice);
        assert_eq!(voice.f32_vector("embedding", 3).unwrap(), [0.5, -1.0, 2.0]);
        ErrorMatcher::invalid_resource().assert_err(&voice.f32_vector("embedding", 4));
        ErrorMatcher::invalid_resource().assert_err(&voice.f32_vector("ref_codes", 4));
        ErrorMatcher::invalid_resource().assert_err(&voice.tensor("xvector"));
    }

    /// A header with another key, without one, or of another version is
    /// refused; so is a voice of another build.
    #[test]
    fn test_refusals() {
        let voice = voice();
        let data = [0u8; 4];
        let view = || TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let mut extra = header(&voice.identity);
        extra.insert("name".into(), "Ada".into());
        let mut missing = header(&voice.identity);
        missing.remove("bunsen.voice.mode");
        let mut version = header(&voice.identity);
        version.insert("bunsen.voice.version".into(), "2".into());
        for meta in [Some(extra), Some(missing), Some(version), None] {
            let bytes = serialize([("x", view())], meta).unwrap();
            ErrorMatcher::invalid_resource().assert_err(&TtsVoicePrint::from_bytes(&bytes));
        }
        ErrorMatcher::invalid_resource().assert_err(&TtsVoicePrint::from_bytes(b"not a voice"));

        let identity = &voice.identity;
        identity
            .check("zonos3", "zonos3-mini@73500c5a/ownmect_s2")
            .unwrap();
        let err = identity.check("zonos3", "other").unwrap_err();
        assert!(
            format!("{err}").contains("build it again from its reference"),
            "{err}"
        );
        ErrorMatcher::illegal().assert_err(&identity.check("qwen3_tts", &identity.encoder));
    }
}
