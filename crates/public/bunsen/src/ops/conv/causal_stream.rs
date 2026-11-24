//! A causal convolution's stream context, and its padding.

use alloc::{
    vec,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
    prelude::*,
};

use crate::contracts::assert_shape_contract_periodically;

/// What a causal convolution pads its input with.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum CausalPadding {
    /// Zeros (`torch`'s `constant`).
    Zeros,

    /// The edge's own frame, repeated (`torch`'s `replicate`).
    Replicate,
}

impl CausalPadding {
    /// `frames` frames of padding for `x` (`[B, C, T]`, `T ≥ 1`): zeros, or
    /// frame `at` of `x`, repeated.
    pub fn edge(
        self,
        x: &Tensor<3>,
        at: usize,
        frames: usize,
    ) -> Tensor<3> {
        let [b, c, _] = x.dims();
        match self {
            Self::Zeros => Tensor::zeros([b, c, frames], (&x.device(), x.dtype())),
            Self::Replicate => x.clone().slice(s![.., .., at..=at]).repeat_dim(2, frames),
        }
    }
}

/// One stream through a causal convolution of receptive field `R` and
/// stride `s`: the inputs a later call still needs.
///
/// Its windowing is the whole forward's, so that the chunking is not
/// observable: the stream starts with the left padding, `R − s` frames
/// (zeros, or the stream's first frame repeated); each
/// [`push`](Self::push) returns the window of every whole output the inputs
/// so far complete, and keeps what is left; [`finish`](Self::finish) pads a
/// partial last stride as the whole forward does (zeros, or the last frame
/// repeated) and returns its window. The windows of a stream, convolved
/// unpadded and joined, are the whole forward's output, whatever the chunks.
///
/// The convolution's own module makes and drives it
/// ([`CausalConv1d::forward_stream`](crate::blocks::conv::CausalConv1d::forward_stream)).
/// The model never owns one, so one model serves any number of streams;
/// each batch row is one stream.
///
/// A `Module` over bare tensors: the pending inputs move with `to_device`,
/// and are neither recorded nor visited (a `ModuleMapper`'s cast skips
/// them).
#[derive(Module, Debug)]
pub struct CausalConvStreamContext {
    /// `[B, C, P]`: the inputs not yet consumed, oldest first (the left
    /// padding, then the frames past the last whole output's window start);
    /// `None` when none are.
    pub pending: Option<Tensor<3>>,

    /// The frames pushed so far, the padding not counted.
    pub frames: usize,

    /// The outputs returned so far.
    pub outputs: usize,

    /// Whether [`finish`](Self::finish) has run.
    pub finished: bool,

    /// The receptive field, `R = (K − 1) · d + 1`.
    pub receptive_field: usize,

    /// The stride, `s`.
    pub stride: usize,

    /// What the stream's edges are padded with.
    #[module(skip)]
    pub padding: CausalPadding,
}

impl CausalConvStreamContext {
    /// A stream not yet started.
    ///
    /// # Panics
    /// For a stride of 0, or over the receptive field.
    pub fn new(
        receptive_field: usize,
        stride: usize,
        padding: CausalPadding,
    ) -> Self {
        assert!(
            stride >= 1 && stride <= receptive_field,
            "a causal convolution's stride is in 1..={receptive_field}; got {stride}"
        );
        Self {
            pending: None,
            frames: 0,
            outputs: 0,
            finished: false,
            receptive_field,
            stride,
            padding,
        }
    }

    /// The frames padded before the stream, `R − s`.
    pub fn left_padding(&self) -> usize {
        self.receptive_field - self.stride
    }

    /// Back to a stream not yet started.
    pub fn reset(&mut self) {
        self.pending = None;
        self.frames = 0;
        self.outputs = 0;
        self.finished = false;
    }

    /// The next inputs, and the window of the outputs they complete.
    ///
    /// # Arguments
    /// - `x`: `[B, C, T]`, `T ≥ 1`: the stream's next frames.
    ///
    /// # Returns
    /// `[B, C, (n − 1) · s + R]`, the window of the `n` outputs now complete,
    /// for an unpadded convolution of stride `s`; `None` when none is.
    ///
    /// # Panics
    /// For no frames, a batch or width unlike the stream's, or a finished
    /// stream.
    pub fn push(
        &mut self,
        x: Tensor<3>,
    ) -> Option<Tensor<3>> {
        assert!(!self.finished, "a push to a finished causal conv stream");
        let [b, c, t] = x.dims();
        assert!(t >= 1, "a causal conv stream's push of at least one frame");
        let mut parts = Vec::with_capacity(2);
        match self.pending.take() {
            Some(pending) => {
                assert_shape_contract_periodically!(
                    ["B", "C", "P"],
                    &pending.dims(),
                    &[("B", b), ("C", c)]
                );
                parts.push(pending);
            }
            None if self.frames == 0 && self.left_padding() > 0 => {
                parts.push(self.padding.edge(&x, 0, self.left_padding()));
            }
            None => {}
        }
        parts.push(x);
        self.frames += t;
        self.take_windows(Tensor::cat(parts, 2))
    }

    /// The end of the stream: the window of the last output, when a partial
    /// stride is left, padded to a whole one.
    ///
    /// # Returns
    /// `[B, C, R]`, the last output's window; `None` when the pushed frames
    /// were a whole number of strides, or none were pushed, or the stream was
    /// finished already.
    pub fn finish(&mut self) -> Option<Tensor<3>> {
        if self.finished {
            return None;
        }
        self.finished = true;
        let partial = self.frames % self.stride;
        if self.frames == 0 || partial == 0 {
            self.pending = None;
            return None;
        }
        let pending = self.pending.take().expect("a partial stride's inputs");
        let p = pending.dims()[2];
        let pad = self.padding.edge(&pending, p - 1, self.stride - partial);
        let window = self.take_windows(Tensor::cat(vec![pending, pad], 2));
        self.pending = None;
        window
    }

    /// The window of every whole output in `buffer` (`[B, C, L]`); the rest
    /// kept as `pending`.
    fn take_windows(
        &mut self,
        buffer: Tensor<3>,
    ) -> Option<Tensor<3>> {
        let (r, s) = (self.receptive_field, self.stride);
        let l = buffer.dims()[2];
        if l < r {
            self.pending = Some(buffer);
            return None;
        }
        let n = (l - r) / s + 1;
        let window = buffer.clone().slice(s![.., .., 0..(n - 1) * s + r]);
        self.pending = (n * s < l).then(|| buffer.slice(s![.., .., n * s..l]));
        self.outputs += n;
        Some(window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// `[1, 1, T]`: the frames `start..start + T`, each its own index.
    fn frames(
        start: usize,
        t: usize,
        device: &Device,
    ) -> Tensor<3> {
        let values: Vec<f32> = (start..start + t).map(|i| i as f32).collect();
        Tensor::<1>::from_floats(values.as_slice(), device).reshape([1, 1, t])
    }

    fn values(window: Option<Tensor<3>>) -> Vec<f32> {
        window.map_or_else(Vec::new, |w| w.into_data().try_to_vec::<f32>().unwrap())
    }

    /// `R = 4`, `s = 2`, zeros: 2 padding frames, then windows of whole
    /// strides; the partial last stride padded at the end.
    #[test]
    fn test_windows_strided() {
        let device = cpu_device();
        let mut ctx = CausalConvStreamContext::new(4, 2, CausalPadding::Zeros);
        assert_eq!(ctx.left_padding(), 2);
        // [0, 0, 0]: 3 frames, under R.
        assert_eq!(values(ctx.push(frames(0, 1, &device))), Vec::<f32>::new());
        // [0, 0, 0, 1, 2]: one output, its window [0, 0, 0, 1]; [2] kept.
        assert_eq!(
            values(ctx.push(frames(1, 2, &device))),
            [0.0, 0.0, 0.0, 1.0]
        );
        // [0, 1, 2, 3, 4]: one more output, its window [0, 1, 2, 3]; [2, 3, 4]
        // kept.
        assert_eq!(
            values(ctx.push(frames(3, 2, &device))),
            [0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(ctx.outputs, 2);
        // 5 frames: a partial stride, padded with a zero.
        assert_eq!(values(ctx.finish()), [2.0, 3.0, 4.0, 0.0]);
        assert_eq!((ctx.outputs, ctx.frames), (3, 5));
        assert!(ctx.finish().is_none());
    }

    /// Replicate: the stream's first frame before it, its last after it.
    #[test]
    fn test_windows_replicate() {
        let device = cpu_device();
        let mut ctx = CausalConvStreamContext::new(4, 2, CausalPadding::Replicate);
        let y = values(ctx.push(frames(5, 3, &device)));
        assert_eq!(y, [5.0, 5.0, 5.0, 6.0]);
        assert_eq!(values(ctx.finish()), [5.0, 6.0, 7.0, 7.0]);
    }

    /// A whole number of strides leaves nothing for `finish`; `reset` starts
    /// over, padding again.
    #[test]
    fn test_finish_whole_and_reset() {
        let device = cpu_device();
        let mut ctx = CausalConvStreamContext::new(3, 1, CausalPadding::Zeros);
        assert_eq!(
            values(ctx.push(frames(1, 2, &device))),
            [0.0, 0.0, 1.0, 2.0]
        );
        assert!(ctx.finish().is_none());
        ctx.reset();
        assert_eq!(values(ctx.push(frames(1, 1, &device))), [0.0, 0.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "stride is in 1..=3")]
    fn test_stride_past_field() {
        let _ = CausalConvStreamContext::new(3, 4, CausalPadding::Zeros);
    }

    #[test]
    #[should_panic(expected = "a push to a finished causal conv stream")]
    fn test_push_after_finish() {
        let device = cpu_device();
        let mut ctx = CausalConvStreamContext::new(3, 1, CausalPadding::Zeros);
        let _ = ctx.push(frames(0, 1, &device));
        let _ = ctx.finish();
        let _ = ctx.push(frames(1, 1, &device));
    }
}
