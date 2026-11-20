//! The decoder toy checks: a decoder's logits, uncached and through every KV
//! cache storage.

use core::ops::Range;

use burn::prelude::*;

use super::{
    KVCache,
    KVCacheConfig,
    KVCacheStorage,
};
use crate::{
    burner::module::ModuleInit,
    support::testing::assert_io_close,
};

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
    let device = uncached.device();
    assert_io_close("uncached logits", toy.logits(0..n, None), uncached);

    for storage in KV_CACHE_STORAGES {
        let mut cache = toy
            .kv_cache_config()
            .with_seq_len(2)
            .with_chunk_size(2)
            .with_storage(storage)
            .init(&device);
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
