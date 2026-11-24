//! # timm model validation
//!
//! bunsen's image kits against timm itself, on `ImageNet`'s validation
//! images. One crate for every timm family, parameterised by catalog row.
//!
//! The images are `ImageNet`'s own: the first [`IMAGES`] samples of the
//! validation set's first shard (`timm/imagenet-1k-validation`, from the
//! gated `timm/imagenet-1k-wds` mirror), pulled into the cache at run time
//! with the user's Hub token, never committed. What is committed is what
//! timm computes from them, per row, in `testdata/<segment>.safetensors`:
//! the logits and the pre-logits features, in float64 cast to float32, and,
//! in the file's metadata, the digest of the preprocessed input they were
//! computed from, the sample keys and their classes. timm's preprocessed
//! inputs themselves are pixels, and stay in the local cache
//! ([`inputs_path`]).
//!
//! ## Making the fixtures
//!
//! A fresh checkout runs two steps before the checks can run:
//!
//! 1. `cargo run -p timm-model-validation --features checkpoints --bin
//!    timm-fixture-plan > plan.json`: resolves each row through bunsen, fetches
//!    its checkpoint and the validation shard into the cache, and writes the
//!    plan: each row's checkpoint path, timm architecture, eval transform, and
//!    where its inputs and fixture go.
//! 2. `python -I validation/tools/gen_timm_fixtures.py plan.json`: timm reads
//!    the same files, writes each transform's preprocessed inputs into the
//!    cache and each row's fixture into `testdata/`.
//!
//! Step 2 rewrites a committed fixture only when its numbers change; on
//! another machine it only fills the cache.
//!
//! ## The checks
//!
//! Per row (`models::<segment>`): bunsen's model, its card's transform the
//! plan's, on timm's preprocessed input (its digest the fixture's): its
//! logits and pooled features against timm's, and the same top class for
//! every image. bunsen's own preprocessing of the same images is held to
//! timm's input at a pixel tolerance, so a resize difference never hides in
//! the model check.

use std::path::{
    Path,
    PathBuf,
};

use bunsen::kits::images::support::transform::ImageEvalTransformConfig;
use sha2::{
    Digest,
    Sha256,
};

/// The rows with committed fixtures, by bunsen's row name.
pub const ROWS: &[&str] = &[
    "torchvision/resnet18",
    "torchvision/resnet50",
    "timm/resnet18.a1_in1k",
    "timm/resnet26.bt_in1k",
    "timm/resnet50.a1_in1k",
    "timm/resnet101.tv2_in1k",
];

/// How many validation images a fixture covers: the first of the shard.
pub const IMAGES: usize = 8;

/// A row's segment in a test path and its fixture's name: its last `/`
/// part, `-` and `.` as `_` (`timm/resnet50.a1_in1k` is `resnet50_a1_in1k`),
/// as `cargo validate` maps a model id.
pub fn segment(row: &str) -> String {
    row.rsplit('/')
        .next()
        .unwrap_or(row)
        .replace(['-', '.'], "_")
}

/// A row's committed fixture.
pub fn fixture_path(row: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(format!("{}.safetensors", segment(row)))
}

/// The key of an eval transform's preprocessed inputs: the first 16 hex
/// digits of the SHA-256 of its JSON. Rows of one transform share inputs.
pub fn input_key(transform: &ImageEvalTransformConfig) -> String {
    let json = serde_json::to_vec(transform).expect("a transform serializes");
    hex(&Sha256::digest(&json))[..16].to_string()
}

/// Lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Where the preprocessed inputs of `key` are kept, in the cache's data
/// directory: local, never committed.
pub fn inputs_path(
    cache: &bunsen::data::resources::ResourceCache,
    key: &str,
) -> PathBuf {
    cache
        .disk()
        .data_path(&["validation", "timm"], format!("inputs-{key}.safetensors"))
}

/// One row's committed fixture: timm's answers on the validation images.
#[derive(Debug, Clone)]
pub struct RowFixture {
    /// The `[IMAGES, classes]` logits, row-major.
    pub logits: Vec<f32>,
    /// The logits' class count.
    pub classes: usize,
    /// The `[IMAGES, d]` pre-logits features, row-major.
    pub pre_logits: Vec<f32>,
    /// The features' width.
    pub d: usize,
    /// The key of the inputs they were computed from.
    pub input_key: String,
    /// The SHA-256 of those inputs' float32 bytes.
    pub input_sha256: String,
    /// Each image's `ImageNet` class, as the shard labels it.
    pub labels: Vec<usize>,
    /// Each image's sample key in the shard.
    pub keys: Vec<String>,
}

/// A tensor of a safetensors file as `f32`s, with its shape.
fn f32_tensor(
    file: &safetensors::SafeTensors<'_>,
    name: &str,
    path: &Path,
) -> (Vec<f32>, Vec<usize>) {
    let view = file
        .tensor(name)
        .unwrap_or_else(|e| panic!("{}: {name}: {e}", path.display()));
    assert_eq!(
        view.dtype(),
        safetensors::Dtype::F32,
        "{}: {name}",
        path.display()
    );
    let values = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    (values, view.shape().to_vec())
}

/// A safetensors file's `__metadata__` entry `key`.
fn metadata(
    bytes: &[u8],
    key: &str,
    path: &Path,
) -> String {
    let (_, header) = safetensors::SafeTensors::read_metadata(bytes)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    header
        .metadata()
        .as_ref()
        .and_then(|m| m.get(key).cloned())
        .unwrap_or_else(|| panic!("{}: no metadata {key}", path.display()))
}

/// Reads a row's committed fixture.
///
/// # Panics
/// When it is missing or malformed.
pub fn read_fixture(row: &str) -> RowFixture {
    let path = fixture_path(row);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let file = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (logits, logits_shape) = f32_tensor(&file, "logits", &path);
    let (pre_logits, pre_shape) = f32_tensor(&file, "pre_logits", &path);
    let list = |key: &str| -> Vec<String> {
        serde_json::from_str(&metadata(&bytes, key, &path))
            .unwrap_or_else(|e| panic!("{}: {key}: {e}", path.display()))
    };
    assert_eq!(metadata(&bytes, "row", &path), row, "{}", path.display());
    RowFixture {
        logits,
        classes: logits_shape[1],
        pre_logits,
        d: pre_shape[1],
        input_key: metadata(&bytes, "input_key", &path),
        input_sha256: metadata(&bytes, "input_sha256", &path),
        labels: list("labels")
            .iter()
            .map(|c| c.parse().expect("a class index"))
            .collect(),
        keys: list("keys"),
    }
}

/// Reads preprocessed inputs, `[IMAGES, channels, height, width]`, and the
/// SHA-256 of their float32 bytes.
///
/// # Panics
/// When the file is missing (with how to make it) or malformed.
pub fn read_inputs(path: &Path) -> (Vec<f32>, [usize; 4], String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}: the preprocessed inputs are made locally: run \
             `timm-fixture-plan`, then `validation/tools/gen_timm_fixtures.py` \
             (the crate's docs)",
            path.display()
        )
    });
    let file = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let view = file.tensor("inputs").expect("an inputs tensor");
    let shape: [usize; 4] = view.shape().try_into().expect("a 4D batch");
    let sha = hex(&Sha256::digest(view.data()));
    let (values, _) = f32_tensor(&file, "inputs", path);
    (values, shape, sha)
}

/// The weights cache, offline under `BUNSEN_VALIDATE_OFFLINE`.
pub fn weights_cache() -> bunsen::data::resources::ResourceCache {
    use bunsen::data::resources::{
        ResourceCache,
        ResourceCacheOptions,
    };
    let offline = std::env::var_os("BUNSEN_VALIDATE_OFFLINE").is_some();
    ResourceCache::new(ResourceCacheOptions::default().with_offline(offline))
        .expect("open the resource cache")
}

/// The validation shard's path, fetched into the cache's data directory with
/// the user's Hub token.
///
/// # Panics
/// When the shard cannot be fetched: an account that has not accepted
/// `ImageNet`'s terms, say.
#[cfg(feature = "checkpoints")]
pub fn validation_shard(cache: &bunsen::data::resources::ResourceCache) -> PathBuf {
    use bunsen::{
        data::datasets::{
            ShardId,
            ShardSet,
            ShardSetDescriptor,
        },
        kits::images::support::imagenet::IMAGENET_1K_VALIDATION,
    };
    let desc =
        ShardSetDescriptor::from_builtin(IMAGENET_1K_VALIDATION).expect("the validation set");
    ShardSet::in_cache(cache.disk(), desc)
        .fetch(ShardId(0))
        .unwrap_or_else(|e| panic!("fetch ImageNet's validation shard: {e:#}"))
}

#[cfg(all(test, feature = "checkpoints", feature = "gpu-tests"))]
mod models;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row has its fixture, of [`IMAGES`] images, readable.
    #[test]
    fn test_fixture_integrity() {
        for row in ROWS {
            let fixture = read_fixture(row);
            assert_eq!(fixture.labels.len(), IMAGES, "{row}");
            assert_eq!(fixture.keys.len(), IMAGES, "{row}");
            assert_eq!(fixture.logits.len(), IMAGES * fixture.classes, "{row}");
            assert_eq!(fixture.pre_logits.len(), IMAGES * fixture.d, "{row}");
            assert_eq!(fixture.input_sha256.len(), 64, "{row}");
        }
    }

    /// Segments are distinct, as `cargo validate` needs them.
    #[test]
    fn test_segments_are_distinct() {
        let mut segments: Vec<_> = ROWS.iter().map(|r| segment(r)).collect();
        segments.sort();
        segments.dedup();
        assert_eq!(segments.len(), ROWS.len());
        assert_eq!(segment("timm/resnet50.a1_in1k"), "resnet50_a1_in1k");
    }
}
