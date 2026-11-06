//! Toy-model fixtures and the decoder toy checks.

use alloc::collections::BTreeMap;
use core::ops::Range;
use std::path::Path;

use burn::{
    prelude::*,
    store::{
        ModuleStore,
        SafetensorsStore,
        bridge,
        burn_pack::Tensor as PackTensor,
    },
    tensor::Tolerance,
};

use crate::ops::transformers::attention::{
    KVCache,
    KVCacheConfig,
    KVCacheStorage,
};

/// A toy model's fixture, read from a safetensors file: the reference's
/// inputs and outputs (the `io.*` tensors, by the name after `io.`) and the
/// model's weights (the rest, under their checkpoint names), read onto a
/// device.
///
/// `tools/toy_fixture.py` writes them.
pub struct ToyFixture {
    io: BTreeMap<String, PackTensor>,
    weights: Vec<PackTensor>,
    device: Device,
}

impl ToyFixture {
    /// The fixture at `path`, its `io.*` tensors apart from its weights.
    ///
    /// # Panics
    /// When the file does not read.
    pub fn load(
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Self {
        let (io, weights): (Vec<_>, Vec<_>) = Self::read(path.as_ref())
            .into_iter()
            .partition(|(name, _)| name.starts_with("io."));
        Self {
            io: io
                .into_iter()
                .map(|(name, t)| (name["io.".len()..].to_string(), t))
                .collect(),
            weights: weights.into_iter().map(|(_, t)| t).collect(),
            device: device.clone(),
        }
    }

    /// Answers alone, at `path`: every tensor is an io tensor, under its own
    /// name (a fixture whose weights are a checkpoint of their own).
    ///
    /// # Panics
    /// When the file does not read.
    pub fn load_io(
        path: impl AsRef<Path>,
        device: &Device,
    ) -> Self {
        Self {
            io: Self::read(path.as_ref()).into_iter().collect(),
            weights: Vec::new(),
            device: device.clone(),
        }
    }

    fn read(path: &Path) -> BTreeMap<String, PackTensor> {
        SafetensorsStore::from_file(path)
            .get_all_tensors()
            .unwrap_or_else(|e| panic!("read the toy fixture {}: {e}", path.display()))
            .clone()
    }

    /// The device the io tensors are read onto.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The weights, for a loader; leaves none behind.
    pub fn take_weights(&mut self) -> Vec<PackTensor> {
        core::mem::take(&mut self.weights)
    }

    /// Whether the fixture has the io tensor `key`.
    pub fn has(
        &self,
        key: &str,
    ) -> bool {
        self.io.contains_key(key)
    }

    /// The io tensor `key`, on the fixture's device, in its default float.
    ///
    /// # Panics
    /// When there is no such tensor, or it is not of rank `D`.
    pub fn io<const D: usize>(
        &self,
        key: &str,
    ) -> Tensor<D> {
        let tensor = self
            .io
            .get(key)
            .unwrap_or_else(|| panic!("the toy fixture has no io tensor {key:?}"));
        Tensor::from_data(
            bridge::to_data(tensor).expect("read an io tensor"),
            &self.device,
        )
    }

    /// The `input_ids` io tensor, which a fixture stores as floats, as ids.
    pub fn ids(&self) -> Tensor<2, Int> {
        self.io::<2>("input_ids").int()
    }
}

/// `actual` is the fixture's `expected`: the same shape, and every value
/// within the toy checks' tolerance, relative `1e-3` and absolute `1e-4` (a
/// float64 reference, stored as float32, against a run on a test device).
///
/// # Panics
/// When they differ; the message names `what`.
pub fn assert_io_close<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
) {
    assert_io_close_within(what, actual, expected, Tolerance::rel_abs(1e-3, 1e-4));
}

/// [`assert_io_close`] within `tolerance`: for a toy whose reference is
/// computed partly in float32 (a `transformers` model that widens some steps
/// to float32 only). Prints the largest absolute difference, the measure a
/// tolerance is set from.
///
/// # Returns
/// The largest absolute difference.
///
/// # Panics
/// When they differ; the message names `what`.
pub fn assert_io_close_within<const D: usize>(
    what: &str,
    actual: Tensor<D>,
    expected: Tensor<D>,
    tolerance: Tolerance<f32>,
) -> f32 {
    assert_eq!(actual.dims(), expected.dims(), "{what}: shape");
    let (actual, expected) = (actual.to_data_as::<f32>(), expected.to_data_as::<f32>());
    let gap = actual
        .iter::<f32>()
        .zip(expected.iter::<f32>())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max);
    eprintln!("{what}: off by at most {gap:.2e}");
    actual.assert_approx_eq::<f32>(&expected, tolerance);
    gap
}

/// A decoder with its toy input, as [`check_decoder_toy`] runs it: the
/// logits of any range of its positions, through a KV cache that holds the
/// positions before the range, or through none.
pub trait DecoderToy {
    /// How many positions the input has.
    fn positions(&self) -> usize;

    /// The config of a cache for the model, at batch 1.
    fn kv_cache_config(&self) -> KVCacheConfig;

    /// The logits of positions `range`, `[1, range.len(), vocab]`. With a
    /// cache, the cache holds positions `..range.start` and takes the range's;
    /// without, `range` starts at 0.
    fn logits(
        &self,
        range: Range<usize>,
        cache: Option<&mut KVCache>,
    ) -> Tensor<3>;
}

/// Every [`KVCacheStorage`], which a cached decode is checked through.
pub const KV_CACHE_STORAGES: [KVCacheStorage; 4] = [
    KVCacheStorage::Auto,
    KVCacheStorage::Concat,
    KVCacheStorage::PerLayer,
    KVCacheStorage::MultiLayer,
];

/// The checks every decoder toy shares:
///
/// - an uncached pass over the whole input is `uncached`;
/// - a prefill of `prefill` positions and then one position per step, through a
///   cache of every [`KVCacheStorage`] that starts too small (two positions,
///   growing two at a time), is `cached`, and leaves the cache at the input's
///   length.
///
/// `uncached` and `cached` are the same logits for a model whose cached step
/// sees what its uncached pass does; a model whose step sees less (a
/// previous-token input it is not given at the prefill's start) records
/// them apart.
///
/// # Panics
/// When a check fails; the message names the storage.
pub fn check_decoder_toy(
    toy: &impl DecoderToy,
    uncached: Tensor<3>,
    cached: Tensor<3>,
    prefill: usize,
) {
    let n = toy.positions();
    assert!(
        0 < prefill && prefill < n,
        "a prefill inside the input: {prefill} of {n}"
    );
    assert_io_close("uncached logits", toy.logits(0..n, None), uncached);

    for storage in KV_CACHE_STORAGES {
        let mut cache = toy
            .kv_cache_config()
            .with_seq_len(2)
            .with_chunk_size(2)
            .with_storage(storage)
            .init();
        let mut parts = vec![toy.logits(0..prefill, Some(&mut cache))];
        for pos in prefill..n {
            parts.push(toy.logits(pos..pos + 1, Some(&mut cache)));
        }
        assert_eq!(cache.pos(), n, "{storage:?}: the cache's length");
        assert_io_close(
            &format!("{storage:?} cached logits"),
            Tensor::cat(parts, 1),
            cached.clone(),
        );
    }
}
