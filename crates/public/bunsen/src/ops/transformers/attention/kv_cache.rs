//! A multi-layer key/value cache, stored as its config's policy says.

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

/// How a [`KVCache`] stores each layer's history.
///
/// A policy, not a fixed design: which storage is fastest depends on the
/// backend, on whether burn's fusion is on, and on burn's version, and will
/// change as they do. [`Auto`](Self::Auto), the default, picks the best known
/// for the device; the others pin a storage, as a benchmark does
/// (`crates/dev/kv-cache-bench`, which is how [`auto_for`](Self::auto_for)'s
/// table is kept current). Every storage behaves the same through
/// [`KVCache`]'s API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KVCacheStorage {
    /// The best known storage for the device the cache first writes on:
    /// [`auto_for`](Self::auto_for).
    #[default]
    Auto,

    /// Each layer's history grows by concatenation, one `Tensor::cat` per
    /// step. It copies the history every step, but a fused backend keeps the
    /// growth in its graph, where it beats an in-place write.
    Concat,

    /// Each layer's keys and values in their own preallocated buffers,
    /// written in place: the buffer is taken out of the cache to be written,
    /// so nothing else references it, and the backend can reuse it.
    PerLayer,

    /// One preallocated `[layers, 2, batch, heads, seq, head_dim]` buffer for
    /// every layer. Its writes are in place, but each layer's history is a
    /// strided view of it, which the attention after it pays to read.
    MultiLayer,
}

impl KVCacheStorage {
    /// This storage, with [`Auto`](Self::Auto) resolved for `device`.
    pub fn resolve(
        self,
        device: &Device,
    ) -> Self {
        match self {
            Self::Auto => Self::auto_for(device),
            storage => storage,
        }
    }

    /// The best known storage for `device`: a best-shot default, aware of
    /// the backend and of fusion.
    ///
    /// From `kv-cache-bench` (2026-10-23, burn 0.22: an Apple M-series on
    /// Metal and its CPU; two RTX 3090s on Vulkan and CUDA), a decode over
    /// Whisper `base` and `large-v3` geometry:
    ///
    /// | Fusion | Runtime | Storage |
    /// |---|---|---|
    /// | off | any | [`PerLayer`](Self::PerLayer): 13% to 1.9x faster than concatenation, with up to half the memory |
    /// | on | wgpu (Vulkan, Metal, WebGPU) | [`Concat`](Self::Concat): fusion makes in-place writes 25-30% slower |
    /// | on | CUDA | [`PerLayer`](Self::PerLayer) |
    ///
    /// Anything else (another runtime, the CPU) gets
    /// [`PerLayer`](Self::PerLayer), the best across everything measured.
    ///
    /// Both facts come from the device, not from bunsen's build: whether
    /// fusion is on is burn's feature, which any crate in the build may
    /// enable, so it is read, with the runtime, from the device's backend
    /// name, which spells out the stack
    /// (`dispatch<fusion<cubecl<wgpu<msl>>>>`).
    pub fn auto_for(device: &Device) -> Self {
        let stack = BackendStack::of(device);
        match (stack.fused, stack.runtime) {
            (true, Runtime::Wgpu) => Self::Concat,
            _ => Self::PerLayer,
        }
    }
}

/// What a device runs on, read from its dispatch backend's name, which
/// spells out the backend stack: `dispatch<fusion<cubecl<wgpu<msl>>>>`,
/// `dispatch<cubecl<cuda...>>`, `dispatch<flex>`.
///
/// burn has no query for whether fusion is on: it is a compile-time feature
/// of burn's, unified across the build, which bunsen's own features can't
/// see. The name is the one runtime account of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BackendStack {
    /// burn's fusion wraps the backend.
    fused: bool,
    runtime: Runtime,
}

/// The runtime under a [`BackendStack`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Runtime {
    /// cubecl's wgpu runtime: Vulkan, Metal, WebGPU.
    Wgpu,
    /// cubecl's CUDA runtime.
    Cuda,
    /// Anything else: another cubecl runtime, the CPU.
    Other,
}

impl BackendStack {
    fn of(device: &Device) -> Self {
        use burn_backend::Backend;
        Self::parse(&burn_dispatch::Dispatch::name(device.as_dispatch()))
    }

    fn parse(name: &str) -> Self {
        let runtime = if name.contains("cubecl<wgpu") {
            Runtime::Wgpu
        } else if name.contains("cubecl<cuda") {
            Runtime::Cuda
        } else {
            Runtime::Other
        };
        Self {
            fused: name.contains("fusion<"),
            runtime,
        }
    }
}

/// Config for [`KVCache`].
///
/// Declares the cache geometry up front, and its sizing: `seq_len` positions
/// allocated on first use, grown by `chunk_size` positions as a decode runs
/// past them, up to `max_seq_len` when the cache is capped; and its
/// [`storage`](Self::storage) policy. Implements [`KVCacheMeta`]. A model
/// hands out its own, through [`KVCacheModel`].
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

    /// How the cache stores its history; [`KVCacheStorage::Auto`], the
    /// best-known storage for the device, by default.
    #[config(default = "KVCacheStorage::Auto")]
    pub storage: KVCacheStorage,
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
            storage: self.storage,
            resolved: None,
            keys: vec![None; self.num_layers],
            values: vec![None; self.num_layers],
        }
    }
}

/// A multi-layer key/value cache for autoregressive decoding.
///
/// Each layer writes its step with [`insert_kv`](KVCache::insert_kv), which
/// returns that layer's whole `(k, v)` history. The shared position
/// ([`pos`](KVCache::pos)) advances once the last layer has written, so one
/// cache serves a whole stack of layers. [`reset`](KVCache::reset) rewinds the
/// position without freeing the storage, and [`prefill`](KVCache::prefill)
/// seeds an empty cache from another one, broadcasting a batch of 1.
///
/// How it stores the history is its config's [`KVCacheStorage`] policy,
/// resolved on the first write, on the device of the first keys
/// ([`storage`](KVCache::storage) reports it). The preallocated storages
/// allocate `seq_len` positions, and grow in chunks (`chunk_size` positions,
/// 1024 by default) when a step runs past their end, up to `max_seq_len`
/// when the cache is capped.
///
/// The cache is injected state: the caller builds one per decode and passes
/// it to the model's forward, so one model can run several decodes at once.
/// It is a `Module` over bare tensors, not parameters.
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

    /// The configured policy.
    #[module(skip)]
    storage: KVCacheStorage,
    /// The storage in use, once the first write resolved it.
    #[module(skip)]
    resolved: Option<KVCacheStorage>,

    pos: usize,
    /// [`KVCacheStorage::MultiLayer`]'s buffer.
    cache: Option<Tensor<6>>,
    /// [`KVCacheStorage::PerLayer`]'s buffers, or
    /// [`KVCacheStorage::Concat`]'s histories, one per layer.
    keys: Vec<Option<Tensor<4>>>,
    /// As `keys`, for the values.
    values: Vec<Option<Tensor<4>>>,
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

    /// The storage in use: the configured policy, with
    /// [`Auto`](KVCacheStorage::Auto) resolved once the first write has
    /// told it the device.
    pub fn storage(&self) -> KVCacheStorage {
        self.resolved.unwrap_or(self.storage)
    }

    /// Whether any storage is allocated.
    fn is_allocated(&self) -> bool {
        self.cache.is_some() || self.keys.iter().any(Option::is_some)
    }

    /// Prefill given another `KVCache`.
    ///
    /// Copies the other cache's history (its first [`pos`](Self::pos)
    /// positions, whatever the length of its storage) into this one's own
    /// storage, and takes its position. A preallocated storage holds this
    /// cache's `seq_len`, grown in chunks if the history is longer. The two
    /// caches' storages need not match.
    ///
    /// - This cache must be empty.
    /// - The other cache must not be.
    /// - The `num_layers`, `num_heads`, and `head_dim` must match.
    /// - The `batch_size` must match, or `other.batch_size` must be 1.
    pub fn prefill(
        &mut self,
        other: &KVCache,
    ) {
        assert!(!self.is_allocated(), "Cannot prefill a non-empty KV cache.");
        assert!(other.is_allocated(), "Cannot prefill from a None KV cache.");

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

        if other.pos > 0 {
            for layer in 0..self.num_layers {
                let (k, v) = other.history(layer);
                let shape = [self.batch_size, self.num_heads, other.pos, self.head_dim];
                self.write(layer, 0, k.expand(shape), v.expand(shape));
            }
        }
        self.pos = other.pos;
    }

    /// `layer`'s `(k, v)` history, its first [`pos`](Self::pos) positions.
    fn history(
        &self,
        layer: usize,
    ) -> (Tensor<4>, Tensor<4>) {
        let t = self.pos;
        match &self.cache {
            Some(cache) => (
                cache
                    .clone()
                    .slice(s![layer, 0, .., .., ..t])
                    .squeeze_dims::<4>(&[0, 1]),
                cache
                    .clone()
                    .slice(s![layer, 1, .., .., ..t])
                    .squeeze_dims::<4>(&[0, 1]),
            ),
            None => {
                let held = |slot: &Option<Tensor<4>>| {
                    slot.clone()
                        .expect("an allocated cache holds every layer")
                        .slice(s![.., .., ..t])
                };
                (held(&self.keys[layer]), held(&self.values[layer]))
            }
        }
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

        let t0 = self.pos;
        let t1 = t0 + t_add;
        if let Some(max) = self.max_seq_len {
            assert!(
                t1 <= max,
                "a KV cache step past its max_seq_len: {t0} + {t_add} > {max}"
            );
        }

        let history = self.write(layer_idx, t0, k, v);

        // Increment pos after the last layer.
        if layer_idx == self.num_layers - 1 {
            // TODO: consider reifying this as a public API, rather than layer
            // magic.
            self.pos = t1;
        }

        history
    }

    /// Writes `(k, v)` at `layer`'s positions `t0..`, in this cache's
    /// storage, and returns the layer's history through them. A write at
    /// `t0` drops anything the layer held past `t0` (after a
    /// [`reset`](Self::reset)).
    fn write(
        &mut self,
        layer: usize,
        t0: usize,
        k: Tensor<4>,
        v: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        let t1 = t0 + k.dims()[2];
        let configured = self.storage;
        let storage = *self
            .resolved
            .get_or_insert_with(|| configured.resolve(&k.device()));
        match storage {
            KVCacheStorage::MultiLayer => self.write_multi_layer(layer, t0, t1, k, v),
            KVCacheStorage::PerLayer => {
                let key = self.keys[layer].take();
                let value = self.values[layer].take();
                let (key, k) = self.write_buffer(key, t0, t1, k);
                let (value, v) = self.write_buffer(value, t0, t1, v);
                self.keys[layer] = Some(key);
                self.values[layer] = Some(value);
                (k, v)
            }
            KVCacheStorage::Concat => {
                let k = Self::concat(self.keys[layer].take(), t0, k);
                let v = Self::concat(self.values[layer].take(), t0, v);
                self.keys[layer] = Some(k.clone());
                self.values[layer] = Some(v.clone());
                (k, v)
            }
            KVCacheStorage::Auto => unreachable!("resolved above"),
        }
    }

    /// [`KVCacheStorage::PerLayer`]'s write: `x` into `buffer` (allocated,
    /// or grown, to hold `t1` positions) at `t0..t1`; the buffer, and its
    /// history through `t1`.
    ///
    /// The buffer is owned here, taken out of the cache, so the write sees
    /// no other reference to it and the backend can write it in place.
    fn write_buffer(
        &self,
        buffer: Option<Tensor<4>>,
        t0: usize,
        t1: usize,
        x: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        let [batch, heads, _, head_dim] = x.dims();
        let allocate = |len: usize| {
            Tensor::<4>::empty([batch, heads, len, head_dim], &x.device()).cast(x.dtype())
        };
        let buffer = match buffer {
            None if t1 <= self.seq_len => allocate(self.seq_len),
            None => allocate(self.allocation_size(t1)),
            Some(buffer) if buffer.dims()[2] >= t1 => buffer,
            Some(buffer) => {
                let held = buffer.dims()[2];
                allocate(self.allocation_size(t1)).slice_assign(s![.., .., ..held], buffer)
            }
        };
        let buffer = buffer.slice_assign(s![.., .., t0..t1], x);
        let history = buffer.clone().slice(s![.., .., ..t1]);
        (buffer, history)
    }

    /// [`KVCacheStorage::Concat`]'s write: `history`, cut back to `t0`, with
    /// `x` appended.
    fn concat(
        history: Option<Tensor<4>>,
        t0: usize,
        x: Tensor<4>,
    ) -> Tensor<4> {
        match history {
            Some(history) if t0 > 0 => {
                let history = if history.dims()[2] > t0 {
                    history.slice(s![.., .., ..t0])
                } else {
                    history
                };
                Tensor::cat(vec![history, x], 2)
            }
            None if t0 > 0 => panic!("a KV cache layer has no history to extend at {t0}"),
            _ => x,
        }
    }

    /// [`KVCacheStorage::MultiLayer`]'s write.
    fn write_multi_layer(
        &mut self,
        layer: usize,
        t0: usize,
        t1: usize,
        k: Tensor<4>,
        v: Tensor<4>,
    ) -> (Tensor<4>, Tensor<4>) {
        let dtype = k.dtype();
        let device = k.device();

        // Release or allocate the cache.
        let mut cache = match self.cache.take() {
            Some(cache) => cache,
            None if t1 <= self.seq_len => self.allocate(self.seq_len, dtype, &device),
            None => self.allocate(self.allocation_size(t1), dtype, &device),
        };

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
            .slice_assign(s![layer, 0, .., .., t0..t1], k.unsqueeze())
            .slice_assign(s![layer, 1, .., .., t0..t1], v.unsqueeze());

        // Get a full key/value slice view up to the current position.
        let k = cache
            .clone()
            .slice(s![layer, 0, .., .., ..t1])
            .squeeze_dims::<4>(&[0, 1]);
        let v = cache
            .clone()
            .slice(s![layer, 1, .., .., ..t1])
            .squeeze_dims::<4>(&[0, 1]);

        // Reattach the cache.
        self.cache = Some(cache);
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

    /// The positions the storage holds (allocated, or concatenated); `None`
    /// before the first write.
    #[cfg(test)]
    fn held_len(&self) -> Option<usize> {
        match &self.cache {
            Some(cache) => Some(cache.dims()[4]),
            None => self.keys[0].as_ref().map(|k| k.dims()[2]),
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
        cpu_device,
        performance_device,
        seeded_tensor,
    };

    /// The storages a cache can be pinned to.
    const STORAGES: [KVCacheStorage; 3] = [
        KVCacheStorage::Concat,
        KVCacheStorage::PerLayer,
        KVCacheStorage::MultiLayer,
    ];

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
    /// positions, for every pair of storages, and checks that every layer
    /// reads the source's history back, broadcast over the batch.
    fn check_prefill(
        source_seq_len: usize,
        filled: usize,
    ) {
        for from in STORAGES {
            for to in STORAGES {
                check_prefill_between(source_seq_len, filled, from, to);
            }
        }
    }

    fn check_prefill_between(
        source_seq_len: usize,
        filled: usize,
        from: KVCacheStorage,
        to: KVCacheStorage,
    ) {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let [num_heads, head_dim, num_layers] = [2, 4, 2];
        let mut source: KVCache =
            KVCacheConfig::new(1, num_heads, source_seq_len, head_dim, num_layers)
                .with_storage(from)
                .init();
        let steps = fill(&mut source, filled, &device);
        assert_eq!(source.pos(), filled);

        let batch = 3;
        let mut cache: KVCache =
            KVCacheConfig::new(batch, num_heads, source_seq_len, head_dim, num_layers)
                .with_storage(to)
                .init();
        cache.prefill(&source);
        assert_eq!(cache.storage(), to, "{from:?} into {to:?}");
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
        for storage in STORAGES {
            let mut cache = KVCacheConfig::new(1, 2, 2, 4, 2)
                .with_chunk_size(4)
                .with_max_seq_len(Some(6))
                .with_storage(storage)
                .init();
            fill(&mut cache, 4, &device);
            fill(&mut cache, 2, &device);
            assert_eq!(cache.pos(), 6);
            assert_eq!(cache.held_len(), Some(6), "{storage:?}");
        }
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
        assert_eq!(cache.storage(), KVCacheStorage::Auto);
    }

    /// Every storage returns the same history through the same writes: a
    /// prompt, steps past the pre-size (growth), a reset and rewrite, and a
    /// step after it.
    #[test]
    #[serial]
    fn test_the_storages_agree() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let [batch, heads, head_dim, layers] = [2, 3, 4, 2];
        let step = |seed: u64, t: usize| {
            seeded_tensor::<4>(
                seed,
                [batch, heads, t, head_dim],
                Distribution::Default,
                &device,
            )
        };
        // (positions written, then whether to reset first).
        let script = [(3, false), (1, false), (4, false), (2, true), (1, false)];
        let run = |storage: KVCacheStorage| {
            let mut cache = KVCacheConfig::new(batch, heads, 4, head_dim, layers)
                .with_chunk_size(2)
                .with_storage(storage)
                .init();
            let mut histories = Vec::new();
            for (i, &(t, reset)) in script.iter().enumerate() {
                if reset {
                    cache.reset();
                }
                for layer in 0..layers {
                    let seed = 10 * i as u64 + 2 * layer as u64;
                    let (k, v) = cache.insert_kv(layer, step(seed, t), step(seed + 1, t));
                    histories.push((k.into_data(), v.into_data()));
                }
            }
            assert_eq!(cache.storage(), storage);
            histories
        };
        let want = run(KVCacheStorage::MultiLayer);
        for storage in [KVCacheStorage::Concat, KVCacheStorage::PerLayer] {
            for ((k, v), (want_k, want_v)) in run(storage).iter().zip(&want) {
                k.assert_eq(want_k, true);
                v.assert_eq(want_v, true);
            }
        }
    }

    /// The backend names burn 0.22's dispatch reports, as seen on the CPU,
    /// Metal, Vulkan and CUDA, with and without fusion.
    #[test]
    fn test_backend_stack_parses_backend_names() {
        let parse = BackendStack::parse;
        let stack = |fused, runtime| BackendStack { fused, runtime };
        assert_eq!(parse("dispatch<flex>"), stack(false, Runtime::Other));
        assert_eq!(
            parse("dispatch<cubecl<wgpu<spirv>>>"),
            stack(false, Runtime::Wgpu)
        );
        assert_eq!(
            parse("dispatch<fusion<cubecl<wgpu<spirv>>>>"),
            stack(true, Runtime::Wgpu)
        );
        assert_eq!(parse("dispatch<cubecl<cuda>>"), stack(false, Runtime::Cuda));
        assert_eq!(
            parse("dispatch<cubecl<wgpu<msl>>>"),
            stack(false, Runtime::Wgpu)
        );
        assert_eq!(
            parse("dispatch<fusion<cubecl<wgpu<msl>>>>"),
            stack(true, Runtime::Wgpu)
        );
        assert_eq!(
            parse("dispatch<fusion<cubecl<cuda>>>"),
            stack(true, Runtime::Cuda)
        );
        assert_eq!(
            parse("dispatch<autodiff<fusion<cubecl<wgpu<spirv>>>>>"),
            stack(true, Runtime::Wgpu)
        );
    }

    /// The device this test runs on reports a backend name the parser
    /// understands: a cubecl device's names its runtime.
    #[test]
    #[serial]
    fn test_backend_stack_reads_the_device() {
        use burn_backend::Backend;
        let device = performance_device();
        let name = burn_dispatch::Dispatch::name(device.as_dispatch());
        let stack = BackendStack::of(&device);
        println!(
            "backend {name}: {stack:?}, Auto is {:?}",
            KVCacheStorage::auto_for(&device)
        );
        assert_eq!(stack.fused, name.contains("fusion<"), "{name}");
        if name.contains("cubecl<") {
            assert!(
                name.contains("cubecl<wgpu")
                    || name.contains("cubecl<cuda")
                    || stack.runtime == Runtime::Other,
                "{name}"
            );
        }
    }

    /// `Auto` resolves on the first write, on the device of the first keys:
    /// on the CPU, to `PerLayer`.
    #[test]
    fn test_auto_resolves_on_the_first_write() {
        let device = cpu_device();
        assert_eq!(KVCacheStorage::auto_for(&device), KVCacheStorage::PerLayer);
        assert_eq!(
            KVCacheStorage::Concat.resolve(&device),
            KVCacheStorage::Concat
        );

        let mut cache = KVCacheConfig::new(1, 2, 4, 4, 1).init();
        assert_eq!(cache.storage(), KVCacheStorage::Auto);
        cache.insert_kv(
            0,
            Tensor::zeros([1, 2, 1, 4], &device),
            Tensor::zeros([1, 2, 1, 4], &device),
        );
        assert_eq!(cache.storage(), KVCacheStorage::PerLayer);
    }
}
