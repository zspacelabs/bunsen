#![allow(clippy::type_complexity)]

/// An `Iterator` wrapper with a callback watcher for each iteration.
pub struct IterWatcher<I>
where
    I: Iterator,
{
    inner: I,
    watcher: Box<dyn FnMut(&I::Item)>,
}

impl<I> IterWatcher<I>
where
    I: Iterator,
{
    pub fn new(
        inner: I,
        watcher: Box<dyn FnMut(&I::Item)>,
    ) -> Self {
        Self { inner, watcher }
    }
}

impl<I> Iterator for IterWatcher<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next();
        if let Some(ref item) = item {
            (self.watcher)(item);
        }
        item
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::AtomicUsize,
    };

    use super::*;

    #[test]
    fn test_iter_watcher() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let iter = IterWatcher::new(
            vec![1, 2, 3].into_iter(),
            Box::new(move |_| {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }),
        );

        let res = iter.collect::<Vec<u32>>();
        assert_eq!(res, vec![1, 2, 3]);

        assert_eq!(counter_clone.load(std::sync::atomic::Ordering::Relaxed), 3);
    }
}
