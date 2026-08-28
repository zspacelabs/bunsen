use burn::{
    Tensor,
    prelude::Backend,
};

/// Projected, head-split keys and values: `[batch, heads, seq, d_k]`.
///
/// Built by [`project_kv_pair`](super::project_kv_pair). Self-attention grows
/// one of these a step at a time; cross-attention builds one per layer and
/// reuses it untouched.
#[derive(Clone, Debug)]
pub struct AttnKvPair<B: Backend> {
    /// `[batch, heads, seq, d_k]` keys.
    pub key: Tensor<B, 4>,

    /// `[batch, heads, seq, d_k]` values.
    pub value: Tensor<B, 4>,
}

impl<B: Backend> AttnKvPair<B> {
    /// The number of cached positions.
    pub fn seq_len(&self) -> usize {
        self.key.dims()[2]
    }

    /// The batch size.
    pub fn batch_size(&self) -> usize {
        self.key.dims()[0]
    }

    /// Appends `next` along the sequence axis.
    ///
    /// This reallocates: every call copies the whole cache, so growing a
    /// self-attention cache one token at a time costs O(T^2) copying across a
    /// decode of length T. That is not worth avoiding at a few hundred
    /// positions, but a long-context decode wants a preallocated cache
    /// instead; see [`KVCache`].
    ///
    /// [`KVCache`]: crate::blocks::transformers::attention::kvcache::KVCache
    pub fn concat(
        self,
        next: Self,
    ) -> Self {
        Self {
            key: Tensor::cat(vec![self.key, next.key], 2),
            value: Tensor::cat(vec![self.value, next.value], 2),
        }
    }
}
