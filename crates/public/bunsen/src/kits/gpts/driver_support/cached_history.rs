//! The tokens a KV cache holds.
use crate::ops::transformers::attention::KVCache;

/// The tokens a [`KVCache`] holds, in order: what a session has run through
/// the model so far.
///
/// A multi-turn session renders its whole conversation for each turn, and
/// most of it is what the cache already holds. [`rewind`](Self::rewind)
/// keeps the prefix the new rendering shares with the history, truncates
/// the cache to it ([`KVCache::truncate`]), and returns the rest, which is
/// all the session has to prefill. At least one token is always left to
/// prefill, since a decode starts from the logits of the last prompt
/// position.
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
        cache: &mut KVCache,
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
}
