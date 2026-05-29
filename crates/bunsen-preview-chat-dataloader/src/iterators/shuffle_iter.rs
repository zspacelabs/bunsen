use rand::{
    RngExt,
    SeedableRng,
    rngs::StdRng,
};

/// Options for [`ShuffleIter`].
#[derive(Debug, Clone, Copy)]
pub struct ShuffleIterOptions {
    fill_rate: usize,
    buffer_size: usize,
    seed: u64,
}

impl Default for ShuffleIterOptions {
    fn default() -> Self {
        Self {
            fill_rate: 8,
            buffer_size: 128,
            seed: 0,
        }
    }
}

impl ShuffleIterOptions {
    pub fn fill_rate(&self) -> usize {
        self.fill_rate
    }

    pub fn with_fill_rate(
        mut self,
        fill_rate: usize,
    ) -> Self {
        self.fill_rate = fill_rate;
        self
    }

    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    pub fn with_buffer_size(
        mut self,
        buffer_size: usize,
    ) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn with_seed(
        mut self,
        seed: u64,
    ) -> Self {
        self.seed = seed;
        self
    }

    pub fn init<I: Iterator>(
        self,
        inner: I,
    ) -> ShuffleIter<I> {
        ShuffleIter::new(inner, self)
    }
}

/// An iterator that shuffles the items it iterates over.
pub struct ShuffleIter<I>
where
    I: Iterator,
{
    inner: I,
    options: ShuffleIterOptions,
    done: bool,
    buffer: Vec<I::Item>,
    rng: StdRng,
}

impl<I> ShuffleIter<I>
where
    I: Iterator,
{
    /// Creates a new `ShuffleIter`.
    ///
    /// Using `fill_rate === buffer_size` will produce the common semantics of
    /// af streaming shuffle. Using a smaller `fill_rate` can reduce latency
    /// at the begining of iteration, at the cost of shuffle bias.
    ///
    /// ## Arguments
    /// * `inner` - The iterator to shuffle.
    /// * `fill_rate` - The upper limit on calls to `inner.next()` per step.
    /// * `buffer_size` - The size of the shuffle buffer.
    /// * `seed` - The rng seed.
    pub fn new(
        inner: I,
        options: ShuffleIterOptions,
    ) -> Self {
        assert!(options.fill_rate > 0);
        assert!(options.fill_rate <= options.buffer_size);
        assert!(options.buffer_size > 1);

        let buffer = Vec::with_capacity(options.buffer_size);
        let rng = StdRng::seed_from_u64(options.seed);

        Self {
            inner,
            options,
            done: false,
            buffer,
            rng,
        }
    }
}

impl<I> Iterator for ShuffleIter<I>
where
    I: Iterator,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        for _ in 0..self.options.fill_rate {
            if !self.done && self.buffer.len() < self.options.buffer_size {
                if let Some(item) = self.inner.next() {
                    self.buffer.push(item);
                    continue;
                } else {
                    self.done = true;
                }
            }
            break;
        }

        if self.buffer.is_empty() {
            return None;
        }

        let idx = self.rng.random_range(0..self.buffer.len());
        let val = self.buffer.swap_remove(idx);

        if !self.done
            && let Some(new_value) = self.inner.next()
        {
            self.buffer.push(new_value);
        }

        Some(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_iter() {
        let source_items = (0..10).collect::<Vec<u32>>();

        let options = ShuffleIterOptions::default();

        let pass1 =
            ShuffleIter::new(source_items.clone().into_iter(), options).collect::<Vec<u32>>();

        let pass2 =
            ShuffleIter::new(source_items.clone().into_iter(), options).collect::<Vec<u32>>();

        assert_eq!(&pass1, &pass2);

        let mut pass1 = pass1;
        pass1.sort_unstable();

        assert_eq!(pass1, source_items);
    }
}
