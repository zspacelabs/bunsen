//! A sliding buffer of feature frames.

use alloc::{
    vec,
    vec::Vec,
};
use core::marker::PhantomData;

use burn::Tensor;

/// The frames of one stream from the seek pointer onward, in a buffer that is
/// written in place.
///
/// The live frames are `buf[head .. head + len]` along dim 1. A push writes
/// after the last live frame with `slice_assign`, which runs in place while
/// the ring owns the only handle to the buffer. Dropping frames from the front
/// moves `head` and copies nothing. Only when a push reaches the end of the
/// buffer are the live frames copied, to the start of a new buffer, and the
/// buffer doubles if they fill more than half of it. So each frame is copied
/// at most once more on average, however the stream is chunked; appending by
/// `cat` copies the whole ring on every push.
///
/// The tail of the buffer past the live frames is uninitialized, and nothing
/// reads it.
#[derive(Clone, Debug)]
pub(crate) struct FrameRing {
    /// `[1, capacity, features]`; `None` until the first push.
    buf: Option<Tensor<3>>,

    /// The buffer index of the first live frame.
    head: usize,

    /// The number of live frames.
    len: usize,

    /// The capacity of the first buffer, in frames.
    min_capacity: usize,
}

impl FrameRing {
    /// An empty ring; its first buffer holds at least `min_capacity` frames.
    pub(crate) fn new(min_capacity: usize) -> Self {
        Self {
            buf: None,
            head: 0,
            len: 0,
            min_capacity: min_capacity.max(1),
        }
    }

    /// The number of live frames.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Appends `[1, count, features]` frames after the last live frame.
    pub(crate) fn push(
        &mut self,
        frames: Tensor<3>,
    ) {
        let [_, count, features] = frames.dims();
        if count == 0 {
            return;
        }

        let capacity = self.buf.as_ref().map_or(0, |buf| buf.dims()[1]);
        if self.head + self.len + count > capacity {
            let needed = self.len + count;
            let capacity = if needed * 2 <= capacity {
                capacity
            } else {
                (capacity * 2).max(needed).max(self.min_capacity)
            };
            self.reallocate(capacity, features, &frames);
        }

        let at = self.head + self.len;
        let buf = self.buf.take().expect("reallocate made a buffer");
        self.buf = Some(buf.slice_assign([0..1, at..at + count, 0..features], frames));
        self.len += count;
    }

    /// Drops the first `count` live frames, or all of them if there are
    /// fewer.
    pub(crate) fn drop_front(
        &mut self,
        count: usize,
    ) {
        let count = count.min(self.len);
        self.head += count;
        self.len -= count;
    }

    /// Drops the last `count` live frames, or all of them if there are fewer.
    pub(crate) fn drop_back(
        &mut self,
        count: usize,
    ) {
        self.len -= count.min(self.len);
    }

    /// The live frames `start .. start + count`, joined: `[1, count,
    /// features]`.
    ///
    /// [`frame_window`](Self::frame_window) as one tensor. While the window
    /// is one piece, this is a view of the buffer, not a copy. Drop it before
    /// the next [`push`](Self::push), or that push copies the buffer.
    ///
    /// # Panics
    /// If the range is not inside the live frames.
    pub(crate) fn window(
        &self,
        start: usize,
        count: usize,
    ) -> Tensor<3> {
        self.frame_window(start, count).into_tensor()
    }

    /// The live frames `start .. start + count`, as the pieces of the buffer
    /// they lie in.
    ///
    /// # Panics
    /// If the range is not inside the live frames.
    pub(crate) fn frame_window(
        &self,
        start: usize,
        count: usize,
    ) -> FrameWindow<'_> {
        assert!(
            start + count <= self.len,
            "window {start}..{} is past the {} live frames",
            start + count,
            self.len,
        );
        let at = self.head + start;
        let piece = self
            .buf
            .as_ref()
            .expect("live frames have a buffer")
            .clone()
            .slice_dim(1, at..at + count);
        FrameWindow::new(vec![piece])
    }

    /// Moves the live frames to the start of a new buffer of `capacity`
    /// frames, with the dtype and device of `source`.
    fn reallocate(
        &mut self,
        capacity: usize,
        features: usize,
        source: &Tensor<3>,
    ) {
        let fresh = Tensor::empty([1, capacity, features], (&source.device(), source.dtype()));
        self.buf = Some(match self.buf.take() {
            Some(old) if self.len > 0 => {
                let kept = old.slice_dim(1, self.head..self.head + self.len);
                fresh.slice_assign([0..1, 0..self.len, 0..features], kept)
            }
            _ => fresh,
        });
        self.head = 0;
    }
}

/// Some live frames of a [`FrameRing`]: the pieces of the buffer they lie
/// in, in order, each `[1, frames, features]`.
///
/// The pieces agree in every dim but the frames, so they join along dim 1.
/// The ring never wraps today, so there is one piece; a caller that walks
/// [`iter`](Self::iter) rather than taking
/// [`into_tensor`](Self::into_tensor) keeps working if it ever does. The
/// window borrows the ring, so nothing can push to the ring while it lives.
#[derive(Clone, Debug)]
pub(crate) struct FrameWindow<'a> {
    /// The pieces, in order; never empty.
    pieces: Vec<Tensor<3>>,

    ring: PhantomData<&'a FrameRing>,
}

impl FrameWindow<'_> {
    /// # Panics
    /// If there are no pieces, or a piece disagrees with the first in a dim
    /// other than the frames.
    fn new(pieces: Vec<Tensor<3>>) -> Self {
        let [batch, _, features] = pieces
            .first()
            .expect("a window has at least one piece")
            .dims();
        for piece in &pieces[1..] {
            let [b, _, m] = piece.dims();
            assert!(
                b == batch && m == features,
                "piece [{b}, _, {m}] does not join [{batch}, _, {features}]",
            );
        }
        Self {
            pieces,
            ring: PhantomData,
        }
    }

    /// The pieces, in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Tensor<3>> {
        self.pieces.iter()
    }

    /// The pieces joined along dim 1. One piece is returned as it is, with
    /// no copy; more are concatenated.
    pub(crate) fn into_tensor(mut self) -> Tensor<3> {
        if self.pieces.len() == 1 {
            return self.pieces.pop().expect("one piece");
        }
        Tensor::cat(self.iter().cloned().collect(), 1)
    }
}

#[cfg(test)]
mod test {
    use burn::{
        Tensor,
        prelude::TensorData,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    /// `[1, count, 2]` frames whose values are their stream index.
    fn frames(
        from: usize,
        count: usize,
    ) -> Tensor<3> {
        let values: Vec<f32> = (from..from + count)
            .flat_map(|i| [i as f32, i as f32])
            .collect();
        Tensor::from_data(TensorData::new(values, [1, count, 2]), &cpu_device())
    }

    fn indices(window: Tensor<3>) -> Vec<f32> {
        window
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .step_by(2)
            .collect()
    }

    #[test]
    fn test_push_and_window() {
        let mut ring = FrameRing::new(4);
        ring.push(frames(0, 3));
        ring.push(frames(3, 2));

        assert_eq!(ring.len(), 5);
        assert_eq!(indices(ring.window(0, 5)), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(indices(ring.window(2, 2)), vec![2.0, 3.0]);
    }

    #[test]
    fn test_drop_front_then_push_compacts() {
        let mut ring = FrameRing::new(4);
        let mut next = 0;
        let mut first = 0;
        // Many more frames than the capacity: the ring compacts and grows,
        // and always reads back the live frames in order.
        for step in 0..40 {
            let count = step % 3 + 1;
            ring.push(frames(next, count));
            next += count;

            let drop = (step % 4).min(ring.len());
            ring.drop_front(drop);
            first += drop;

            let expected: Vec<f32> = (first..next).map(|i| i as f32).collect();
            assert_eq!(ring.len(), next - first);
            if ring.len() > 0 {
                assert_eq!(indices(ring.window(0, ring.len())), expected);
            }
        }
    }

    #[test]
    fn test_drop_back() {
        let mut ring = FrameRing::new(4);
        ring.push(frames(0, 3));
        ring.drop_back(1);
        assert_eq!(indices(ring.window(0, ring.len())), vec![0.0, 1.0]);

        ring.drop_back(5);
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn test_frame_window_one_piece() {
        let mut ring = FrameRing::new(4);
        ring.push(frames(0, 3));

        let window = ring.frame_window(1, 2);
        assert_eq!(window.iter().count(), 1);
        assert_eq!(indices(window.into_tensor()), vec![1.0, 2.0]);
    }

    #[test]
    fn test_frame_window_joins_pieces() {
        let window = FrameWindow::new(vec![frames(0, 2), frames(2, 1), frames(3, 2)]);
        assert_eq!(window.iter().count(), 3);
        assert_eq!(indices(window.into_tensor()), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "does not join")]
    fn test_frame_window_rejects_mismatched_pieces() {
        let other = Tensor::<3>::zeros([1, 2, 3], &cpu_device());
        let _ = FrameWindow::new(vec![frames(0, 2), other]);
    }

    #[test]
    #[should_panic(expected = "past the")]
    fn test_window_past_the_end_panics() {
        let mut ring = FrameRing::new(4);
        ring.push(frames(0, 2));
        let _ = ring.window(1, 2);
    }
}
