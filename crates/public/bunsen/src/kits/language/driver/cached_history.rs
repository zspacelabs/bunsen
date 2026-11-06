//! The tokens a decode's cache holds.
use crate::{
    blocks::transformers::decoders::stack::DecoderStackCache,
    ops::transformers::attention::KVCache,
};

/// A decode's cache as [`CachedHistory`] rewinds it: its position, and
/// whether and how it rewinds to an earlier one.
pub trait RewindableCache {
    /// The position the next step writes at: the tokens held.
    fn pos(&self) -> usize;

    /// Whether [`truncate`](Self::truncate) can rewind to `pos`.
    fn can_truncate(
        &self,
        pos: usize,
    ) -> bool;

    /// Rewinds to `pos`, which [`can_truncate`](Self::can_truncate) allows.
    fn truncate(
        &mut self,
        pos: usize,
    );
}

/// A KV cache rewinds within what its sliding windows retain, or resets
/// ([`KVCache::truncate`]).
impl RewindableCache for KVCache {
    fn pos(&self) -> usize {
        KVCache::pos(self)
    }

    fn can_truncate(
        &self,
        pos: usize,
    ) -> bool {
        KVCache::can_truncate(self, pos)
    }

    fn truncate(
        &mut self,
        pos: usize,
    ) {
        KVCache::truncate(self, pos);
    }
}

/// A decoder stack's cache rewinds as its KV cache does, but its
/// linear-attention states only reset ([`DecoderStackCache::truncate`]).
impl RewindableCache for DecoderStackCache {
    fn pos(&self) -> usize {
        DecoderStackCache::pos(self)
    }

    fn can_truncate(
        &self,
        pos: usize,
    ) -> bool {
        DecoderStackCache::can_truncate(self, pos)
    }

    fn truncate(
        &mut self,
        pos: usize,
    ) {
        DecoderStackCache::truncate(self, pos);
    }
}

/// The tokens a decode's cache holds ([`RewindableCache`]: a [`KVCache`], or
/// a [`DecoderStackCache`]), in order: what a session has run through the
/// model so far.
///
/// A multi-turn session renders its whole conversation for each turn, and
/// most of it is what the cache already holds. [`rewind`](Self::rewind)
/// keeps the prefix the new rendering shares with the history, truncates
/// the cache to it ([`KVCache::truncate`]), and returns the rest, which is
/// all the session has to prefill. Where the cache cannot rewind that far
/// (a sliding window has dropped the positions it would keep, or a hybrid
/// model's linear-attention states, which summarize the whole prefix, would
/// have to rewind), it resets, and the whole rendering is left to prefill. At
/// least one token is always left to prefill, since a decode starts from the
/// logits of the last prompt position.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CachedHistory {
    tokens: Vec<i64>,
}

impl CachedHistory {
    /// An empty history, for an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The tokens held.
    pub fn tokens(&self) -> &[i64] {
        &self.tokens
    }

    /// Records tokens run through the model into the cache.
    pub fn extend(
        &mut self,
        tokens: &[i64],
    ) {
        self.tokens.extend_from_slice(tokens);
    }

    /// Rewinds `cache` and the history to the prefix `next` shares with the
    /// history, short of `next`'s last token, and returns what of `next` is
    /// left to prefill. The history then holds that prefix; record the
    /// prefill with [`extend`](Self::extend).
    ///
    /// # Panics
    /// If `next` is empty, or the cache is not at the history's length.
    pub fn rewind<'a>(
        &mut self,
        cache: &mut impl RewindableCache,
        next: &'a [i64],
    ) -> &'a [i64] {
        assert!(!next.is_empty(), "a prompt to prefill");
        assert_eq!(
            cache.pos(),
            self.tokens.len(),
            "the cache holds the history"
        );
        let shared = self
            .tokens
            .iter()
            .zip(next)
            .take_while(|(a, b)| a == b)
            .count()
            .min(next.len() - 1);
        let shared = if cache.can_truncate(shared) {
            shared
        } else {
            0
        };
        cache.truncate(shared);
        self.tokens.truncate(shared);
        &next[shared..]
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    use super::*;
    use crate::{
        ops::transformers::attention::KVCacheConfig,
        support::testing::cpu_device,
    };

    fn advance(
        cache: &mut KVCache,
        n: usize,
    ) {
        let device = cpu_device();
        let t = Tensor::<4>::zeros([1, 1, n, 2], &device);
        cache.insert_kv(0, t.clone(), t);
    }

    #[test]
    fn test_rewind_keeps_the_shared_prefix() {
        let mut cache = KVCacheConfig::new(1, 1, 8, 2, 1).init();
        let mut history = CachedHistory::new();

        let first = [1, 2, 3, 4];
        let todo = history.rewind(&mut cache, &first);
        assert_eq!(todo, first, "nothing cached yet");
        advance(&mut cache, todo.len());
        history.extend(todo);

        // The next turn re-renders the end of the last.
        let next = [1, 2, 9, 9, 9];
        let todo = history.rewind(&mut cache, &next);
        assert_eq!(todo, [9, 9, 9]);
        assert_eq!(cache.pos(), 2);
        assert_eq!(history.tokens(), [1, 2]);
        advance(&mut cache, todo.len());
        history.extend(todo);

        // The same prompt again leaves its last token to prefill.
        let todo = history.rewind(&mut cache, &next);
        assert_eq!(todo, [9]);
        assert_eq!(cache.pos(), 4);
    }

    /// A hybrid model's decode cache: one attention layer's KV cache, and one
    /// linear layer's state, made non-zero as a prefill would.
    fn hybrid_cache() -> DecoderStackCache {
        use crate::{
            blocks::transformers::embedding::{
                RotaryEmbeddingConfig,
                RotarySet,
            },
            burner::module::ModuleInit,
            ops::transformers::linear_attention::{
                DeltaNetCache,
                DeltaNetStateConfig,
            },
        };
        let device = cpu_device();
        DecoderStackCache {
            cache: KVCacheConfig::new(1, 1, 8, 2, 1).init(),
            linear: DeltaNetCache::new(vec![
                DeltaNetStateConfig::new(1, 3, 2, 1, 2, 2).init(&device),
            ]),
            rotary: RotarySet::from(ModuleInit::init(&RotaryEmbeddingConfig::new(8, 2), &device)),
        }
    }

    fn fill_linear(cache: &mut DecoderStackCache) {
        let state = cache.linear.layer_mut(0);
        state.recurrent = state.recurrent.ones_like();
    }

    fn linear_is_zero(cache: &mut DecoderStackCache) -> bool {
        let state = cache.linear.layer_mut(0);
        state.recurrent.clone().abs().max().into_scalar::<f32>() == 0.0
    }

    /// A hybrid model's linear states cannot rewind: a next prompt that
    /// extends the history keeps them, and one that parts from it resets
    /// the whole cache, and leaves the whole rendering to prefill.
    #[test]
    fn test_rewind_a_hybrid_cache() {
        let mut cache = hybrid_cache();
        let mut history = CachedHistory::new();
        let first = [1, 2, 3, 4];
        let todo = history.rewind(&mut cache, &first);
        advance(&mut cache.cache, todo.len());
        fill_linear(&mut cache);
        history.extend(todo);

        let longer = [1, 2, 3, 4, 5, 6];
        let todo = history.rewind(&mut cache, &longer);
        assert_eq!(todo, [5, 6], "the history extended: nothing rewinds");
        assert_eq!(cache.pos(), 4);
        assert!(!linear_is_zero(&mut cache), "the states kept");

        let parted = [1, 2, 9, 9];
        assert!(!cache.can_truncate(2) && cache.can_truncate(4) && cache.can_truncate(0));
        let todo = history.rewind(&mut cache, &parted);
        assert_eq!(todo, parted, "the linear states cannot rewind to 2");
        assert_eq!(cache.pos(), 0);
        assert!(linear_is_zero(&mut cache), "the states reset");
        assert!(history.tokens().is_empty());
    }

    /// Where a sliding window has dropped the shared prefix, the cache
    /// resets and the whole rendering is left to prefill.
    #[test]
    fn test_rewind_past_a_window_resets() {
        let mut cache = KVCacheConfig::new(1, 1, 8, 2, 1)
            .with_layer_windows(vec![Some(2)])
            .init();
        let mut history = CachedHistory::new();
        let first = [1, 2, 3, 4];
        let todo = history.rewind(&mut cache, &first);
        advance(&mut cache, todo.len());
        history.extend(todo);
        assert_eq!(cache.retained_from(), 3);

        let next = [1, 2, 9, 9];
        let todo = history.rewind(&mut cache, &next);
        assert_eq!(todo, next, "position 2 is gone");
        assert_eq!(cache.pos(), 0);
        assert!(history.tokens().is_empty());
    }
}
