//! The committed fixtures: upstream's reference outputs, as
//! `validation/tools/gen_zonos3_fixtures.py` writes them (its docstring is
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

/// The upstream reference's float32 outputs, over each of [`REQUESTS`].
pub const REFERENCE: &str = "reference.safetensors";

/// The upstream reference's bfloat16 logits over [`REFERENCE`]'s prompts and
/// generated rows.
pub const REFERENCE_BF16: &str = "reference_bf16.safetensors";

/// The `-hf` codec's audio from fixed codes.
pub const CODEC: &str = "codec.safetensors";

/// The `-hf` tokenizer's ids for the `bunsen-test-audio` UDHR texts.
pub const TOKENIZER: &str = "tokenizer.safetensors";

/// The requests (upstream's `prompt_requests.json`) the reference files
/// hold, each under `<name>.`.
pub const REQUESTS: [&str; 8] = [
    "plain",
    "embedding",
    "reference_codes",
    "condition_values",
    "silence_prefill_5",
    "silence_prefill_max",
    "pass_through_language",
    "reference_codes_long",
];

/// The decode steps each request's reference holds.
pub const STEPS: usize = 12;

/// The codebooks: the logits' second axis, a frame's width.
pub const CODEBOOKS: usize = 10;

/// The prompt's row width: the codebooks, then the text.
pub const ROW_WIDTH: usize = CODEBOOKS + 1;

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

    /// The request `name`'s prompt facts ([`REFERENCE`] only).
    pub fn request(
        &self,
        name: &str,
    ) -> RequestFacts {
        self.metadata_json(name)
    }
}

/// A request's prompt facts, as the reference planned it.
#[derive(Clone, Debug, Deserialize)]
pub struct RequestFacts {
    /// The request, as upstream's `prompt_requests.json` gives it.
    pub request: serde_json::Value,

    /// Its text, normalized by upstream's normalizer.
    pub normalized_text: String,

    /// The row the speaker embedding replaces (embedding mode).
    pub speaker_row: Option<usize>,

    /// The first conditioned row.
    pub condition_row: Option<usize>,

    /// The conditioned rows' buckets, in row order.
    pub condition_buckets: Vec<u32>,

    /// The silence rows ending the prompt.
    pub silence_rows: usize,

    /// The stride of the prefill logits' `prefill.sample`.
    pub stats_stride: usize,
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

    const PINS: [&str; 3] = ["upstream", "hf_revision", "torch"];

    /// Every file reads, carries the same pins, and has the shapes the
    /// generator documents.
    #[test]
    fn test_fixtures_read() {
        let reference = FixtureFile::load(REFERENCE);
        let bf16 = FixtureFile::load(REFERENCE_BF16);
        let codec = FixtureFile::load(CODEC);
        let tokenizer = FixtureFile::load(TOKENIZER);
        for file in [&bf16, &codec, &tokenizer] {
            for pin in PINS {
                assert_eq!(file.metadata(pin), reference.metadata(pin), "{pin}");
            }
        }
        assert_eq!(reference.metadata("steps"), STEPS.to_string());
        assert_eq!(bf16.metadata("dtype"), "bf16");

        for name in REQUESTS {
            let facts = reference.request(name);
            assert!(!facts.normalized_text.is_empty());
            let at = |k: &str| format!("{name}.{k}");
            let rows = reference.shape(&at("rows"))[0];
            assert_eq!(reference.shape(&at("rows")), [rows, ROW_WIDTH]);
            assert!(facts.silence_rows < rows);
            assert_eq!(
                facts.condition_row.is_some(),
                !facts.condition_buckets.is_empty()
            );
            assert_eq!(
                facts.speaker_row.is_some(),
                reference.has(&at("speaker_embedding"))
            );
            assert_eq!(reference.shape(&at("generated")), [STEPS, ROW_WIDTH]);
            for file in [&reference, &bf16] {
                assert_eq!(file.shape(&at("prefill.argmax")), [rows, CODEBOOKS]);
                assert_eq!(file.shape(&at("prefill.margin")), [rows, CODEBOOKS]);
                assert_eq!(file.shape(&at("prefill.top8.ids")), [CODEBOOKS, 8]);
                assert_eq!(file.shape(&at("prefill.sums")), [2]);
                assert_eq!(file.shape(&at("steps.argmax")), [STEPS, CODEBOOKS]);
                assert_eq!(file.shape(&at("steps.top8.values")), [STEPS, CODEBOOKS, 8]);
            }
        }

        let cases: Vec<String> = codec.metadata_json("cases");
        assert_eq!(cases.len(), 5);
        for case in &cases {
            let at = |k: &str| format!("{case}.{k}");
            let frames = codec.shape(&at("codes"))[0];
            assert_eq!(codec.shape(&at("codes")), [frames, CODEBOOKS]);
            let stats = codec.doubles(&at("audio.stats"));
            let windows = codec.shape(&at("audio.windows"));
            let starts = codec.ints(&at("audio.window_starts"));
            assert_eq!(windows[0], starts.len());
            for start in starts {
                assert!(start as f64 + windows[1] as f64 <= stats[0]);
            }
        }

        assert_eq!(tokenizer.keys().count(), 6);
    }
}
