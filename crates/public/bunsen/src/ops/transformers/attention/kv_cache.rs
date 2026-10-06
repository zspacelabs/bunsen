//! A preallocated multi-layer key/value cache.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Backend,
        s,
    },
    tensor::DType,
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
}

/// Config for [`KVCache`].
///
/// Declares the cache geometry up front. Implements [`KVCacheMeta`].
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
}

impl KVCacheConfig {
    /// Sets the `batch_size`.
    pub fn with_batch_size(
        self,
        batch_size: usize,
    ) -> Self {
        Self { batch_size, ..self }
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
}

impl KVCacheConfig {
    /// Initializes a [`KVCache`].
    pub fn init<B: Backend>(self) -> KVCache<B> {
        KVCache {
            batch_size: self.batch_size,
            num_heads: self.num_heads,
            seq_len: self.seq_len,
            head_dim: self.head_dim,
            num_layers: self.num_layers,
            pos: 0,
            cache: None,
            chunk_size: 1024,
            extra_chunks: 1,
        }
    }
}

/// A preallocated, multi-layer key/value cache for autoregressive decoding.
///
/// One `[layers, 2, batch, heads, seq, head_dim]` tensor holds the keys and
/// values of every layer. It is allocated on first use, on the device of the
/// first keys, and grows in 1024-position chunks when a step runs past its
/// end.
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
/// Built by [`KVCacheConfig`], whose `init` takes no device.
///
/// [`CausalSelfAttention`]: crate::blocks::transformers::attention::csa::CausalSelfAttention
#[derive(Module, Debug)]
pub struct KVCache<B: Backend> {
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    num_layers: usize,

    chunk_size: usize,
    extra_chunks: usize,

    pos: usize,
    cache: Option<Tensor<B, 6>>,
}

impl<B: Backend> KVCacheMeta for KVCache<B> {
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
}

impl<B: Backend> KVCache<B> {
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
        other: &KVCache<B>,
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
    pub fn insert_kv(
        &mut self,
        layer_idx: usize,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
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
        device: &B::Device,
    ) -> Tensor<B, 6> {
        Tensor::<B, 6>::empty(
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

    /// Computes the target allocation size for a given required size.
    pub fn allocation_size(
        &self,
        required_size: usize,
    ) -> usize {
        (required_size.div_ceil(self.chunk_size) + self.extra_chunks) * self.chunk_size
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        PerformanceBackend,
        performance_device,
        seeded_tensor,
    };

    /// Writes `t_add` seeded positions to every layer of `cache`, and returns
    /// each layer's `(k, v)` step.
    fn fill<B: Backend>(
        cache: &mut KVCache<B>,
        t_add: usize,
        device: &B::Device,
    ) -> Vec<(Tensor<B, 4>, Tensor<B, 4>)> {
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
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let [num_heads, head_dim, num_layers] = [2, 4, 2];
        let mut source: KVCache<B> =
            KVCacheConfig::new(1, num_heads, source_seq_len, head_dim, num_layers).init();
        let steps = fill(&mut source, filled, &device);
        assert_eq!(source.pos(), filled);

        let batch = 3;
        let mut cache: KVCache<B> =
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
}
