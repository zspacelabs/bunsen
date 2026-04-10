//! # KV Cache

use bimm_contracts::{
    assert_shape_contract_periodically,
    unpack_shape_contract,
};
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
    /// Set the `batch_size`.
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
    /// Initialize a [`KVCache`].
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

/// KV Cache
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
    /// Reset the current position.
    ///
    /// Does not drop/re-allocate the cache.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Get the current position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Prefill given another `KVCache`.
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

        let cache = self.allocate(other.seq_len, other_cache.dtype(), &other_cache.device());

        let source = other_cache.clone();
        let mut source_shape = source.dims();
        source_shape[2] = self.batch_size;
        let other_cache = source.expand(source_shape);

        self.cache = cache
            .slice_assign(s![.., .., .., .., ..other.pos, ..], other_cache)
            .into();
        self.pos = other.pos;
    }

    /// Insert and extend a (k, v) pair.
    ///
    /// # Arguments
    /// - `layer_idx`: the block layer index.
    /// - `k`: the ``[B, H_kv, T, D]`` key tensor.
    /// - `v`: the ``[B, H_kv, T, D]`` value tensor.
    ///
    /// # Returns
    /// - the extended (k, v) ``[B, H_kv, T, D]`` pair.
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
            // TODO: consider reifying this as a public API, rather than layer magic.
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

    /// Compute the target allocation size for a given required size.
    pub fn allocation_size(
        &self,
        required_size: usize,
    ) -> usize {
        (required_size.div_ceil(self.chunk_size) + self.extra_chunks) * self.chunk_size
    }
}
