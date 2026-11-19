//! The committed fixtures: `qwen_tts`' outputs over the checkpoint, as
//! `validation/tools/gen_qwen3_tts_fixtures.py` writes them (its docstring is
//! the layout's full description).
//!
//! Each is a safetensors file: its tensors by name, and its header's
//! `__metadata__`, a map of strings, holding the pins and, where a value is
//! structured, JSON.

use std::{
    collections::BTreeMap,
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use bunsen::data::formats::SafetensorsCheckpoint;
use burn::{
    prelude::TensorData,
    store::bridge,
};
use serde::Deserialize;

/// `qwen_tts`' float32 outputs, over each of [`REQUESTS`].
pub const REFERENCE: &str = "reference.safetensors";

/// `qwen_tts`' bfloat16 logits over [`REFERENCE`]'s takes, dictated their
/// codes.
pub const REFERENCE_BF16: &str = "reference_bf16.safetensors";

/// The codec's audio from fixed codes.
pub const CODEC: &str = "codec.safetensors";

/// The processor's tokenizer's ids for the test texts.
pub const TOKENIZER: &str = "tokenizer.safetensors";

/// The requests [`REFERENCE`] holds, each under `<name>.`.
pub const REQUESTS: [&str; 4] = [
    "ryan_english",
    "vivian_chinese_upfront",
    "eric_auto_dialect",
    "aiden_french_upfront",
];

/// The frames of each request's take.
pub const FRAMES: usize = 12;

/// The codebooks a frame holds.
pub const CODE_GROUPS: usize = 16;

/// The stride of the strided samples.
pub const SAMPLE_STRIDE: usize = 97;

/// The fixture directory: this crate's `testdata`.
pub fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata")
}

/// One fixture file, read whole.
pub struct FixtureFile {
    name: String,
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, TensorData>,
}

impl FixtureFile {
    /// The fixture `name` (one of [`REFERENCE`], [`REFERENCE_BF16`],
    /// [`CODEC`], [`TOKENIZER`]) in [`testdata`].
    ///
    /// # Panics
    /// When the file does not read.
    pub fn load(name: &str) -> Self {
        let path = testdata().join(name);
        let tensors = SafetensorsCheckpoint::single(&path)
            .tensors()
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
            .iter()
            .map(|t| {
                let data =
                    bridge::to_data(t).unwrap_or_else(|e| panic!("{name}: read {}: {e:?}", t.name));
                (t.name.clone(), data)
            })
            .collect();
        Self {
            name: name.to_string(),
            metadata: read_metadata(&path),
            tensors,
        }
    }

    /// The tensors' names.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Whether the file holds the tensor `key`.
    pub fn has(
        &self,
        key: &str,
    ) -> bool {
        self.tensors.contains_key(key)
    }

    /// The tensor `key`.
    ///
    /// # Panics
    /// When the file has none.
    pub fn tensor(
        &self,
        key: &str,
    ) -> &TensorData {
        self.tensors
            .get(key)
            .unwrap_or_else(|| panic!("{}: no tensor {key}", self.name))
    }

    /// The tensor `key`'s shape.
    pub fn shape(
        &self,
        key: &str,
    ) -> Vec<usize> {
        self.tensor(key).shape().as_slice().to_vec()
    }

    /// The tensor `key`'s values, as integers.
    pub fn ints(
        &self,
        key: &str,
    ) -> Vec<i64> {
        self.tensor(key)
            .clone()
            .convert::<i64>()
            .try_into_vec::<i64>()
            .unwrap()
    }

    /// The tensor `key`'s values, as `f32`.
    pub fn floats(
        &self,
        key: &str,
    ) -> Vec<f32> {
        self.tensor(key)
            .clone()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .unwrap()
    }

    /// The tensor `key`'s values, as `f64`.
    pub fn doubles(
        &self,
        key: &str,
    ) -> Vec<f64> {
        self.tensor(key)
            .clone()
            .convert::<f64>()
            .try_into_vec::<f64>()
            .unwrap()
    }

    /// The metadata entry `key`.
    ///
    /// # Panics
    /// When the file has none.
    pub fn metadata(
        &self,
        key: &str,
    ) -> &str {
        self.metadata
            .get(key)
            .unwrap_or_else(|| panic!("{}: no metadata {key}", self.name))
    }

    /// The metadata entry `key`, parsed as JSON.
    ///
    /// # Panics
    /// When the file has none, or it does not parse as a `T`.
    pub fn metadata_json<T: for<'de> Deserialize<'de>>(
        &self,
        key: &str,
    ) -> T {
        serde_json::from_str(self.metadata(key))
            .unwrap_or_else(|e| panic!("{}: metadata {key}: {e}", self.name))
    }

    /// The request `name`'s facts ([`REFERENCE`] only).
    pub fn request(
        &self,
        name: &str,
    ) -> RequestFacts {
        self.metadata_json(name)
    }
}

/// A request, as the generator made it.
#[derive(Clone, Debug, Deserialize)]
pub struct RequestFacts {
    /// The text.
    pub text: String,

    /// The stock speaker.
    pub speaker: String,

    /// The language key (`auto`, or one of the model's).
    pub language: String,

    /// Whether the text is upfront (upstream's non-streaming mode).
    pub non_streaming: bool,

    /// The take's frames.
    pub frames: usize,
}

/// A safetensors file's `__metadata__`.
///
/// # Panics
/// When the header does not read.
fn read_metadata(path: &Path) -> BTreeMap<String, String> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + len])
        .unwrap_or_else(|e| panic!("{}: header: {e}", path.display()));
    header
        .get("__metadata__")
        .map(|m| serde_json::from_value(m.clone()).unwrap())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINS: [&str; 3] = ["revision", "torch", "qwen_tts"];

    /// Every file reads, carries the same pins, and has the shapes the
    /// generator documents.
    #[test]
    fn test_fixtures_read() {
        let reference = FixtureFile::load(REFERENCE);
        let bf16 = FixtureFile::load(REFERENCE_BF16);
        let codec = FixtureFile::load(CODEC);
        let tokenizer = FixtureFile::load(TOKENIZER);
        assert_eq!(bf16.metadata("dtype"), "bf16");
        for file in [&bf16, &codec, &tokenizer] {
            for pin in PINS {
                assert_eq!(file.metadata(pin), reference.metadata(pin), "{pin}");
            }
        }
        assert_eq!(reference.metadata("frames"), FRAMES.to_string());
        assert_eq!(
            reference.metadata("sample_stride"),
            SAMPLE_STRIDE.to_string()
        );
        for name in REQUESTS {
            let facts = reference.request(name);
            assert_eq!(facts.frames, FRAMES);
            let at = |k: &str| format!("{name}.{k}");
            assert_eq!(reference.shape(&at("codes")), [FRAMES, CODE_GROUPS]);
            assert_eq!(reference.shape(&at("talker.top8.ids")), [FRAMES + 1, 8]);
            assert_eq!(reference.shape(&at("talker.sums")), [FRAMES + 1, 2]);
            assert_eq!(
                reference.shape(&at("predictor.top8.values")),
                [FRAMES, CODE_GROUPS - 1, 8]
            );
            for part in ["talker.top8.ids", "predictor.top8.values", "talker.sums"] {
                assert_eq!(bf16.shape(&at(part)), reference.shape(&at(part)), "{part}");
            }
            let rows = reference.shape(&at("prompt.norms"))[0];
            assert_eq!(
                reference.shape(&at("prompt.sample"))[0],
                (rows * 1024).div_ceil(SAMPLE_STRIDE)
            );
        }
        let cases: Vec<String> = codec.metadata_json("cases");
        assert_eq!(cases.len(), REQUESTS.len() + 1);
        for case in &cases {
            let at = |k: &str| format!("{case}.{k}");
            let frames = codec.shape(&at("codes"))[0];
            let stats = codec.doubles(&at("audio.stats"));
            assert_eq!(stats[0], (frames * 1920) as f64);
            let starts = codec.ints(&at("audio.window_starts"));
            assert_eq!(codec.shape(&at("audio.windows"))[0], starts.len());
        }
        assert_eq!(tokenizer.keys().count(), 5);
    }
}
