//! The catalog, `validation/models.yml`: every model a crate checks.

use std::path::Path;

use indexmap::IndexMap;
use serde::Deserialize;

/// The catalog: the crates, in the file's order.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    /// Each crate, by its name in the catalog (`qwen3`, `whisper`).
    pub crates: IndexMap<String, CrateSpec>,
}

/// One validation crate.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrateSpec {
    /// Its package.
    pub package: String,

    /// The features always on when it runs.
    #[serde(default)]
    pub features: Vec<String>,

    /// The feature that builds its ONNX reference, turned on when a chosen
    /// model's model checks need it.
    #[serde(default)]
    pub reference: Option<String>,

    /// Its models, by id (bunsen's row name), in the file's order.
    pub models: IndexMap<String, ModelSpec>,
}

/// One model.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    /// What it fetches, its reference included.
    pub size: Size,

    /// What it fetches on a cold cache, in GB.
    pub gb: f64,

    /// Whether it is a representative: the one model for an architecture
    /// variation.
    pub core: bool,

    /// Whether its model checks need the crate's reference.
    #[serde(default)]
    pub reference: bool,

    /// What it is checked in.
    pub dtypes: Vec<Dtype>,

    /// The variation it stands for, or repeats.
    pub varies: String,
}

/// A model's size, by what it fetches: `small` up to ~1 GB, `medium` up to
/// ~10 GB, `large` more.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Size {
    Small,
    Medium,
    Large,
}

impl Size {
    pub fn name(self) -> &'static str {
        match self {
            Size::Small => "small",
            Size::Medium => "medium",
            Size::Large => "large",
        }
    }
}

/// A dtype a model is checked in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
    Bf16,
}

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::Bf16 => "bf16",
        }
    }
}

impl Catalog {
    /// Reads the catalog at `path`.
    pub fn load(path: &Path) -> Result<Catalog, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_saphyr::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// A model id's segment in a test path: its last `/` part, `-` and `.` as
/// `_` (`qwen3-0.6b` is `qwen3_0_6b`, `openai/base` is `base`).
pub fn segment(model: &str) -> String {
    model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .replace(['-', '.'], "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment() {
        assert_eq!(segment("qwen3-0.6b"), "qwen3_0_6b");
        assert_eq!(segment("openai/base"), "base");
        assert_eq!(
            segment("mistralai/voxtral-mini-4b-realtime-2602"),
            "voxtral_mini_4b_realtime_2602"
        );
        assert_eq!(segment("sortformer-v2.1"), "sortformer_v2_1");
    }

    /// The committed catalog reads, and its segments are distinct within a
    /// crate.
    #[test]
    fn test_the_catalog_reads() {
        let catalog = Catalog::load(&crate::validation_dir().join("models.yml")).unwrap();
        assert!(catalog.crates.contains_key("qwen3"));
        for (name, spec) in &catalog.crates {
            let mut segments: Vec<String> = spec.models.keys().map(|m| segment(m)).collect();
            let n = segments.len();
            segments.sort();
            segments.dedup();
            assert_eq!(segments.len(), n, "{name}: two models share a segment");
        }
    }
}
