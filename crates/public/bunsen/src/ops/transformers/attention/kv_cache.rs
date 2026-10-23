//! A preallocated multi-layer key/value cache.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::s,
    tensor::{
        DType,
        Device,
    },
};

use crate::contracts::{
    assert_shape_contract_periodically,
    unpack_shape_contract,
};

/// Common meta trait for [`KVCache`] and [`KVCacheConfig`].
pub trait KVCacheMeta {
    /// Batch size.
    fn batch_size(&self) -> usize;

    /// Number of attention heads.
    fn num_heads(&self) -> usize;

    /// Initial target sequence length.
    fn seq_len(&self) -> usize;

    /// Dimension of each head.
    fn head_dim(&self) -> usize;

    /// Number of layers.
    fn num_layers(&self) -> usize;

    /// The positions the storage grows by when a step runs past its end.
    fn chunk_size(&self) -> usize;

    /// The most positions the cache may hold, if it is capped.
    fn max_seq_len(&self) -> Option<usize>;
}

/// A model, or its config, that decodes through a [`KVCache`], and knows the
/// cache it needs.
///
/// The model gives the cache's geometry (layers, KV heads, head width) and
/// its sizing: a default pre-size, and a cap where the model has a hard
/// positional limit. A caller asks the model for a cache rather than
/// assembling one, and overrides what it knows better on the config:
///
/// ```rust,ignore
/// let mut cache = model.new_kv_cache(batch);
/// // Or, knowing the decode's length:
/// let mut cache = model.kv_cache_config().with_batch_size(batch).with_seq_len(n).init();
/// ```
///
/// A model's config implements it too, so a cache can be sized (or its
/// memory estimated) before the model is built.
pub trait KVCacheModel {
    /// The config of a cache for this model, at batch 1: its geometry, its
    /// default pre-size, and its cap.
    fn kv_cache_config(&self) -> KVCacheConfig;

    /// A cache for `batch_size` decodes, sized as
    /// [`kv_cache_config`](Self::kv_cache_config) says.
    fn new_kv_cache(
        &self,
        batch_size: usize,
    ) -> KVCache {
        self.kv_cache_config().with_batch_size(batch_size).init()
    }
}

/// Config for [`KVCache`].
///
/// Declares the cache geometry up front, and its sizing: `seq_len` positions
/// allocated on first use, grown by `chunk_size` positions as a decode runs
/// past them, up to `max_seq_len` when the cache is capped. Implements
/// [`KVCacheMeta`]. A model hands out its own, through [`KVCacheModel`].
#[derive(Config, Debug)]
pub struct KVCacheConfig {
    /// Configured batch size.
    pub batch_size: usize,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Initial target sequence length.
    pub seq_len: usize,

    /// Dimension of each head.
    pub head_dim: usize,

    /// Number of layers.
    pub num_layers: usize,

    /// The positions the storage grows by when a step runs past its end.
    #[config(default = "1024")]
    pub chunk_size: usize,

    /// The most positions the cache may hold; uncapped when `None`.
    ///
    /// Growth stops at the cap, and a step past it panics: for a model with
    /// a hard positional limit, running past it is a bug, not a reason to
    /// reallocate.
    #[config(default = "None")]
    pub max_seq_len: Option<usize>,
}

impl KVCacheConfig {
    /// Sets the `batch_size`.
    pub fn with_batch_size(
        self,
        batch_size: usize,
    ) -> Self {
        Self { batch_size, ..self }
    }

    /// Sets the `seq_len`, the positions allocated up front.
    pub fn with_seq_len(
        self,
        seq_len: usize,
    ) -> Self {
        Self { seq_len, ..self }
    }
}

impl KVCacheMeta for KVCacheConfig {
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn max_seq_len(&self) -> Option<usize> {
        self.max_seq_len
    }
}

impl KVCacheConfig {
    /// Initializes a [`KVCache`].
    ///
    /// # Panics
    /// When `chunk_size` is 0, or `seq_len` exceeds `max_seq_len`.
    pub fn init(self) -> KVCache {
        assert!(
            self.chunk_size > 0,
            "a KV cache's chunk_size must be positive"
        );
        if let Some(max) = self.max_seq_len {
            assert!(
                self.seq_len <= max,
                "a KV cache's seq_len ({}) exceeds its max_seq_len ({max})",
                self.seq_len
            );
        }
        KVCache {
            batch_size: self.batch_size,
            num_heads: self.num_heads,
            seq_len: self.seq_len,
            head_dim: self.head_dim,
            num_layers: self.num_layers,
            pos: 0,
            cache: None,
            chunk_size: self.chunk_size,
            extra_chunks: 1,
            max_seq_len: self.max_seq_len,
        }
    }
}

/// A preallocated, multi-layer key/value cache for autoregressive decoding.
///
/// One `[layers, 2, batch, heads, seq, head_dim]` tensor holds the keys and
/// values of every layer. It is allocated on first use, on the device of the
/// first keys, and grows in chunks (`chunk_size` positions, 1024 by default)
/// when a step runs past its end, up to its `max_seq_len` when it is capped.
///
/// Each layer writes its step with [`insert_kv`](KVCache::insert_kv), which
/// returns that layer's whole `(k, v)` history. The shared position
/// ([`pos`](KVCache::pos)) advances once the last layer has written, so one
/// cache serves a whole stack of layers. [`reset`](KVCache::reset) rewinds the
/// position without freeing the storage, and [`prefill`](KVCache::prefill)
/// seeds an empty cache from another one, broadcasting a batch of 1.
///
/// The cache is injected state: the caller builds one per decode and passes
/// it to the model's forward, so one model can run several decodes at once.
/// It is a `Module` over a bare tensor, not parameters.
/// [`CausalSelfAttention`] is its consumer;
/// [`ops::transformers::attention`](crate::ops::transformers::attention)
/// compares it with the per-layer [`AttnKvPair`](super::AttnKvPair).
///
/// Built by [`KVCacheConfig`], whose `init` takes no device; a model hands
/// out its own through [`KVCacheModel`].
///
/// [`CausalSelfAttention`]: crate::blocks::transformers::attention::csa::CausalSelfAttention
#[derive(Module, Debug)]
pub struct KVCache {
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    num_layers: usize,

    chunk_size: usize,
    extra_chunks: usize,
    max_seq_len: Option<usize>,

    pos: usize,
    cache: Option<Tensor<6>>,
}

impl KVCacheMeta for KVCache {
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn max_seq_len(&self) -> Option<usize> {
        self.max_seq_len
    }
}

impl KVCache {
    /// Resets the current position.
    ///
    /// Does not drop/re-allocate the cache.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Returns the current position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Prefill given another `KVCache`.
    ///
    /// Copies the other cache's history (its first [`pos`](Self::pos)
    /// positions, whatever the length of its storage) into fresh storage,
    /// and takes its position. The storage is this cache's `seq_len`, grown
    /// in chunks if the history is longer.
    ///
    /// - This cache must be `None`.
    /// - The other cache must be `Some`.
    /// - The `num_layers`, `num_heads`, and `head_dim` must match.
    /// - The `batch_size` must match, or `other.batch_size` must be 1.
    pub fn prefill(
        &mut self,
        other: &KVCache,
    ) {
        assert!(self.cache.is_none(), "Cannot prefill a non-empty KV cache.");
        assert!(
            other.cache.is_some(),
            "Cannot prefill from a None KV cache."
        );

        assert_eq!(self.num_layers, other.num_layers);
        assert_eq!(self.num_heads, other.num_heads);
        assert_eq!(self.head_dim, other.head_dim);

        if self.batch_size != other.batch_size && other.batch_size != 1 {
            panic!(
                "Incompatible pre-fill batch size: {} vs {}",
                self.batch_size, other.batch_size
            );
        }
        assert!(self.seq_len >= other.seq_len);

        let other_cache = other.cache.as_ref().unwrap();

        let seq_len = if other.pos > self.seq_len {
            self.allocation_size(other.pos)
        } else {
            self.seq_len
        };
        let cache = self.allocate(seq_len, other_cache.dtype(), &other_cache.device());

        // The source's storage can be longer than its history.
        let source = other_cache
            .clone()
            .slice(s![.., .., .., .., ..other.pos, ..]);
        let mut source_shape = source.dims();
        source_shape[2] = self.batch_size;
        let other_cache = source.expand(source_shape);

        self.cache = cache
            .slice_assign(s![.., .., .., .., ..other.pos, ..], other_cache)
            .into();
        self.pos = other.pos;
    }

    /// Inserts and extends a `(k, v)` pair.
    ///
    /// # Arguments
    /// - `layer_idx`: the block layer index.
    /// - `k`: the `[B, H_kv, T, D]` key tensor.
    /// - `v`: the `[B, H_kv, T, D]` value tensor.
    ///
    /// # Returns
    /// - the extended (k, v) `[B, H_kv, T, D]` pair.
    ///
    /// # Panics
    /// When the step would take the cache past its `max_seq_len`.
    pub fn insert_kv(
        &mut self,
        layer_idx: usize,
        k: Tensor<4>,
        v: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        let [t_add] = unpack_shape_contract!(
            ["B", "H_kv", "T_add", "D"],
            &k.dims(),
            &["T_add"],
            &[
                ("B", self.batch_size),
                ("H_kv", self.num_heads),
                ("D", self.head_dim)
            ]
        );
        assert_shape_contract_periodically!(
            ["B", "H_kv", "T_add", "D"],
            &v.dims(),
            &[
                ("B", self.batch_size),
                ("H_kv", self.num_heads),
                ("T_add", t_add),
                ("D", self.head_dim)
            ]
        );

        let dtype = k.dtype();
        let device = k.device();

        // Release or allocate the cache.
        let mut cache = if let Some(cache) = self.cache.take() {
            cache
        } else {
            self.allocate(self.seq_len, dtype, &device)
        };

        let t0 = self.pos;
        let t1 = t0 + t_add;
        if let Some(max) = self.max_seq_len {
            assert!(
                t1 <= max,
                "a KV cache step past its max_seq_len: {t0} + {t_add} > {max}"
            );
        }

        // Grow the cache if needed.
        let cached_size = cache.dims()[4];
        if t1 > cached_size {
            let needed_t = self.allocation_size(t1);

            cache = self
                .allocate(needed_t, dtype, &device)
                .slice_assign(s![.., .., .., .., ..cached_size, ..], cache);
        }

        // Insert k, v into the cache.
        cache = cache
            .slice_assign(s![layer_idx, 0, .., .., t0..t1], k.unsqueeze())
            .slice_assign(s![layer_idx, 1, .., .., t0..t1], v.unsqueeze());

        // Get a full key/value slice view up to the current position.
        let k = cache
            .clone()
            .slice(s![layer_idx, 0, .., .., ..t1])
            .squeeze_dims::<4>(&[0, 1]);
        let v = cache
            .clone()
            .slice(s![layer_idx, 1, .., .., ..t1])
            .squeeze_dims::<4>(&[0, 1]);

        // Reattach the cache.
        self.cache = Some(cache);

        // Increment pos after the last layer.
        if layer_idx == self.num_layers - 1 {
            // TODO: consider reifying this as a public API, rather than layer
            // magic.
            self.pos = t1;
        }

        (k, v)
    }

    fn allocate(
        &self,
        seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Tensor<6> {
        Tensor::<6>::empty(
            [
                self.num_layers,
                2,
                self.batch_size,
                self.num_heads,
                seq_len,
                self.head_dim,
            ],
            device,
        )
        .cast(dtype)
    }

    /// Computes the target allocation size for a given required size: whole
    /// chunks, with spare ones, but never past `max_seq_len`.
    pub fn allocation_size(
        &self,
        required_size: usize,
    ) -> usize {
        let size = (required_size.div_ceil(self.chunk_size) + self.extra_chunks) * self.chunk_size;
        match self.max_seq_len {
            Some(max) => size.min(max).max(required_size),
            None => size,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        performance_device,
        seeded_tensor,
    };

    /// Writes `t_add` seeded positions to every layer of `cache`, and returns
    /// each layer's `(k, v)` step.
    fn fill(
        cache: &mut KVCache,
        t_add: usize,
        device: &Device,
    ) -> Vec<(Tensor<4>, Tensor<4>)> {
        let shape = [
            cache.batch_size(),
            cache.num_heads(),
            t_add,
            cache.head_dim(),
        ];
        (0..cache.num_layers())
            .map(|layer| {
                let seed = 2 * layer as u64;
                let k = seeded_tensor::<4>(seed, shape, Distribution::Default, device);
                let v = seeded_tensor::<4>(seed + 1, shape, Distribution::Default, device);
                cache.insert_kv(layer, k.clone(), v.clone());
                (k, v)
            })
            .collect()
    }

    /// Prefills a batch-3 cache from a batch-1 source holding `filled`
    /// positions, and checks that every layer reads the source's history
    /// back, broadcast over the batch.
    fn check_prefill(
        source_seq_len: usize,
        filled: usize,
    ) {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let [num_heads, head_dim, num_layers] = [2, 4, 2];
        let mut source: KVCache =
            KVCacheConfig::new(1, num_heads, source_seq_len, head_dim, num_layers).init();
        let steps = fill(&mut source, filled, &device);
        assert_eq!(source.pos(), filled);

        let batch = 3;
        let mut cache: KVCache =
            KVCacheConfig::new(batch, num_heads, source_seq_len, head_dim, num_layers).init();
        cache.prefill(&source);
        assert_eq!(cache.pos(), filled);

        // One more position on every layer returns the prefilled history.
        let step = [batch, num_heads, 1, head_dim];
        let history = [batch, num_heads, filled, head_dim];
        for (layer, (k, v)) in steps.into_iter().enumerate() {
            let (k_all, v_all) = cache.insert_kv(
                layer,
                Tensor::zeros(step, &device),
                Tensor::zeros(step, &device),
            );
            assert_eq!(k_all.dims(), [batch, num_heads, filled + 1, head_dim]);
            k_all
                .slice(s![.., .., ..filled])
                .into_data()
                .assert_eq(&k.expand(history).into_data(), true);
            v_all
                .slice(s![.., .., ..filled])
                .into_data()
                .assert_eq(&v.expand(history).into_data(), true);
        }
    }

    /// The source's storage is longer than its position: 3 of 8 filled.
    #[test]
    #[serial]
    fn test_prefill_from_partly_filled_cache() {
        check_prefill(8, 3);
    }

    /// The source grew past its configured length: 6 positions in a cache
    /// configured for 4, so its storage was reallocated in chunks.
    #[test]
    #[serial]
    fn test_prefill_from_grown_cache() {
        check_prefill(4, 6);
    }

    /// The source's storage is exactly its position.
    #[test]
    #[serial]
    fn test_prefill_from_full_cache() {
        check_prefill(5, 5);
    }

    #[test]
    fn test_config_sizing() {
        let config = KVCacheConfig::new(1, 2, 16, 4, 3);
        assert_eq!(config.chunk_size(), 1024);
        assert_eq!(config.max_seq_len(), None);

        let config = config
            .with_batch_size(5)
            .with_seq_len(32)
            .with_chunk_size(8)
            .with_max_seq_len(Some(40));
        let cache = config.init();
        assert_eq!(cache.batch_size(), 5);
        assert_eq!(cache.seq_len(), 32);
        assert_eq!(cache.chunk_size(), 8);
        assert_eq!(cache.max_seq_len(), Some(40));

        // Whole chunks, a spare one, and never past the cap.
        assert_eq!(cache.allocation_size(33), 40);
        assert_eq!(cache.allocation_size(17), 32);
        let uncapped = KVCacheConfig::new(1, 2, 16, 4, 3).with_chunk_size(8).init();
        assert_eq!(uncapped.allocation_size(33), 48);
    }

    #[test]
    #[should_panic(expected = "exceeds its max_seq_len")]
    fn test_config_rejects_a_pre_size_past_its_cap() {
        KVCacheConfig::new(1, 2, 16, 4, 3)
            .with_max_seq_len(Some(8))
            .init();
    }

    /// A capped cache grows to its cap, and no further: a decode may fill it
    /// exactly.
    #[test]
    #[serial]
    fn test_growth_stops_at_the_cap() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let mut cache = KVCacheConfig::new(1, 2, 2, 4, 2)
            .with_chunk_size(4)
            .with_max_seq_len(Some(6))
            .init();
        fill(&mut cache, 4, &device);
        fill(&mut cache, 2, &device);
        assert_eq!(cache.pos(), 6);
        assert_eq!(cache.cache.as_ref().unwrap().dims()[4], 6);
    }

    #[test]
    #[serial]
    #[should_panic(expected = "past its max_seq_len")]
    fn test_a_step_past_the_cap_panics() {
        let device = performance_device();
        let mut cache = KVCacheConfig::new(1, 2, 2, 4, 2)
            .with_max_seq_len(Some(4))
            .init();
        fill(&mut cache, 3, &device);
        fill(&mut cache, 2, &device);
    }

    #[test]
    fn test_a_model_hands_out_its_cache() {
        struct Model;
        impl KVCacheModel for Model {
            fn kv_cache_config(&self) -> KVCacheConfig {
                KVCacheConfig::new(1, 2, 16, 4, 3).with_max_seq_len(Some(64))
            }
        }
        let cache = Model.new_kv_cache(7);
        assert_eq!(cache.batch_size(), 7);
        assert_eq!(cache.num_layers(), 3);
        assert_eq!(cache.seq_len(), 16);
        assert_eq!(cache.max_seq_len(), Some(64));
    }
}
