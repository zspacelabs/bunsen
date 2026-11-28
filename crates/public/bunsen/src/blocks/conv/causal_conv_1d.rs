//! [`CausalConv1d`] and [`CausalConvTranspose1d`]: convolutions over the past.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    nn::{
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
            ConvTranspose1d,
            ConvTranspose1dConfig,
        },
    },
    prelude::*,
};

pub use crate::ops::conv::CausalPadding;
use crate::{
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
    },
    ops::conv::{
        CausalConvStreamContext,
        CausalConvTransposeStreamContext,
    },
};

/// Common meta for [`CausalConv1d`] and [`CausalConv1dConfig`].
pub trait CausalConv1dMeta {
    /// The input and output channels.
    fn channels(&self) -> [usize; 2];

    /// The kernel's taps.
    fn kernel_size(&self) -> usize;

    /// The stride.
    fn stride(&self) -> usize;

    /// The spacing between taps.
    fn dilation(&self) -> usize;

    /// The channel groups.
    fn groups(&self) -> usize;

    /// What the input is padded with.
    fn padding(&self) -> CausalPadding;

    /// The span the kernel reads, `(K - 1) · d + 1`.
    fn receptive_field(&self) -> usize {
        (self.kernel_size() - 1) * self.dilation() + 1
    }

    /// The frames padded before the input: the receptive field less the
    /// stride, `(K - 1) · d + 1 - s`.
    fn left_padding(&self) -> usize {
        self.receptive_field() - self.stride()
    }

    /// The frames padded after an input of `frames`, so that the last stride
    /// is whole: `⌈T / s⌉ · s - T`.
    fn right_padding(
        &self,
        frames: usize,
    ) -> usize {
        frames.div_ceil(self.stride()) * self.stride() - frames
    }

    /// The output frames of an input of `frames`: `⌈T / s⌉`.
    fn output_frames(
        &self,
        frames: usize,
    ) -> usize {
        frames.div_ceil(self.stride())
    }
}

/// Config for [`CausalConv1d`].
#[derive(Config, Debug)]
pub struct CausalConv1dConfig {
    /// The input and output channels.
    pub channels: [usize; 2],

    /// The kernel's taps.
    pub kernel_size: usize,

    /// The stride; at most the receptive field.
    #[config(default = 1)]
    pub stride: usize,

    /// The spacing between taps.
    #[config(default = 1)]
    pub dilation: usize,

    /// The channel groups; each divides both channel counts.
    #[config(default = 1)]
    pub groups: usize,

    /// Whether the convolution adds a bias.
    #[config(default = true)]
    pub bias: bool,

    /// What the input is padded with.
    #[config(default = "CausalPadding::Zeros")]
    pub padding: CausalPadding,
}

impl CausalConv1dMeta for CausalConv1dConfig {
    fn channels(&self) -> [usize; 2] {
        self.channels
    }

    fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    fn stride(&self) -> usize {
        self.stride
    }

    fn dilation(&self) -> usize {
        self.dilation
    }

    fn groups(&self) -> usize {
        self.groups
    }

    fn padding(&self) -> CausalPadding {
        self.padding
    }
}

/// The convolution, its weights burn's default initialization.
impl ModuleInit<CausalConv1d> for CausalConv1dConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a stride of 0
    /// or over the receptive field.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<CausalConv1d> {
        if self.stride == 0 || self.stride > self.receptive_field() {
            return Err(ConstraintError::out_of_range(
                "CausalConv1dConfig",
                "stride",
                self.stride,
                format!("1..={}, the receptive field", self.receptive_field()),
            )
            .into());
        }
        let [c_in, c_out] = self.channels;
        Ok(CausalConv1d {
            conv: Conv1dConfig::new(c_in, c_out, self.kernel_size)
                .with_stride(self.stride)
                .with_dilation(self.dilation)
                .with_groups(self.groups)
                .with_bias(self.bias)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            padding: self.padding,
        })
    }
}

/// A 1-d convolution over each frame's past: output frame `t` reads the input
/// up to frame `t · s` and no later, so a model built of them can run as a
/// stream.
///
/// The input is padded by [`left_padding`](CausalConv1dMeta::left_padding)
/// frames before and [`right_padding`](CausalConv1dMeta::right_padding) after
/// (the last stride made whole), then convolved without padding: `[B, C_in,
/// T]` to `[B, C_out, ⌈T / s⌉]`. With a stride of 1 the right padding is
/// none, and the output is as long as the input. That is
/// the rule of the causal convolutions in Encodec, DAC, Mimi and Qwen3-TTS's
/// codec.
///
/// The convolution is burn's [`Conv1d`], as `conv`: checkpoints that hold it
/// as `conv.weight` and `conv.bias` load as they are.
///
/// # Streams
///
/// A stream runs the convolution over its input in chunks of any size:
/// [`new_stream_context`](Self::new_stream_context) opens one,
/// [`forward_stream`](Self::forward_stream) takes each chunk and returns the
/// outputs it completes, and [`finish_stream`](Self::finish_stream) ends it.
/// The outputs, joined, are [`forward`](Self::forward)'s over the whole
/// input: the left padding is the stream's start (zeros, or its first frame
/// repeated), and a partial last stride is padded only at the end. A strided
/// stream holds back the frames of a partial stride, so a chunk can complete
/// no output.
///
/// Built by [`CausalConv1dConfig`].
#[derive(Module, Debug)]
pub struct CausalConv1d {
    /// The convolution, unpadded.
    pub conv: Conv1d,

    /// What the input is padded with.
    #[module(skip)]
    pub padding: CausalPadding,
}

impl CausalConv1dMeta for CausalConv1d {
    fn channels(&self) -> [usize; 2] {
        let [c_out, c_in_per_group, _] = self.conv.weight.dims();
        [c_in_per_group * self.conv.groups, c_out]
    }

    fn kernel_size(&self) -> usize {
        self.conv.kernel_size
    }

    fn stride(&self) -> usize {
        self.conv.stride
    }

    fn dilation(&self) -> usize {
        self.conv.dilation
    }

    fn groups(&self) -> usize {
        self.conv.groups
    }

    fn padding(&self) -> CausalPadding {
        self.padding
    }
}

impl CausalConv1d {
    /// `x`, padded by `before` and `after` frames on its last axis.
    fn pad(
        &self,
        x: Tensor<3>,
        before: usize,
        after: usize,
    ) -> Tensor<3> {
        let [t] = crate::contracts::unpack_shape_contract!(["B", "C", "T"], &x.dims(), &["T"]);
        let mut parts = Vec::with_capacity(3);
        if before > 0 {
            parts.push(self.padding.edge(&x, 0, before));
        }
        parts.push(x.clone());
        if after > 0 {
            parts.push(self.padding.edge(&x, t - 1, after));
        }
        if parts.len() == 1 {
            x
        } else {
            Tensor::cat(parts, 2)
        }
    }

    /// The convolution over each frame's past.
    ///
    /// # Arguments
    /// - `x`: `[B, C_in, T]`, `T ≥ 1`.
    ///
    /// # Returns
    /// `[B, C_out, ⌈T / s⌉]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [c_in, c_out] = self.channels();
        #[cfg(not(any(debug_assertions, test)))]
        let _ = c_out;
        let [b, t] = unpack_shape_contract!(
            ["B", "C_in", "T"],
            &x.dims(),
            &["B", "T"],
            &[("C_in", c_in)]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = b;
        assert!(t >= 1, "a causal convolution over at least one frame");
        let x = self.pad(x, self.left_padding(), self.right_padding(t));
        let y = self.conv.forward(x);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "C_out", "T_out"],
            &y.dims(),
            &[("B", b), ("C_out", c_out), ("T_out", self.output_frames(t))]
        );
        y
    }
}

impl CausalConv1d {
    /// A stream through this convolution, not yet started.
    pub fn new_stream_context(&self) -> CausalConvStreamContext {
        CausalConvStreamContext::new(self.receptive_field(), self.stride(), self.padding)
    }

    /// The stream's next chunk: the outputs it completes.
    ///
    /// # Arguments
    /// - `x`: `[B, C_in, T]`, `T ≥ 1`, the stream's next frames.
    /// - `ctx`: the stream, from
    ///   [`new_stream_context`](Self::new_stream_context).
    ///
    /// # Returns
    /// `[B, C_out, n]`, the `n ≥ 1` outputs now complete; `None` when none
    /// is.
    ///
    /// # Panics
    /// As [`CausalConvStreamContext::push`]; or for a context of another
    /// convolution's geometry.
    pub fn forward_stream(
        &self,
        x: Tensor<3>,
        ctx: &mut CausalConvStreamContext,
    ) -> Option<Tensor<3>> {
        #[cfg(any(debug_assertions, test))]
        let [c_in, _] = self.channels();
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "C_in", "T"],
            &x.dims(),
            &[("C_in", c_in)]
        );
        self.check_stream(ctx);
        ctx.push(x).map(|window| self.conv.forward(window))
    }

    /// The end of the stream: its last output, when a partial stride was
    /// left.
    ///
    /// # Returns
    /// `[B, C_out, 1]`; `None` when the stream's frames were a whole number
    /// of strides (always, at a stride of 1).
    ///
    /// # Panics
    /// For a context of another convolution's geometry.
    pub fn finish_stream(
        &self,
        ctx: &mut CausalConvStreamContext,
    ) -> Option<Tensor<3>> {
        self.check_stream(ctx);
        ctx.finish().map(|window| self.conv.forward(window))
    }

    fn check_stream(
        &self,
        ctx: &CausalConvStreamContext,
    ) {
        assert!(
            ctx.receptive_field == self.receptive_field()
                && ctx.stride == self.stride()
                && ctx.padding == self.padding,
            "a causal conv stream of another geometry"
        );
    }
}

/// Common meta for [`CausalConvTranspose1d`] and
/// [`CausalConvTranspose1dConfig`].
pub trait CausalConvTranspose1dMeta {
    /// The input and output channels.
    fn channels(&self) -> [usize; 2];

    /// The kernel's taps; at least the stride.
    fn kernel_size(&self) -> usize;

    /// The stride: output frames per input frame.
    fn stride(&self) -> usize;

    /// The channel groups.
    fn groups(&self) -> usize;

    /// The frames trimmed from the end: `K - s`.
    fn right_trim(&self) -> usize {
        self.kernel_size() - self.stride()
    }
}

/// Config for [`CausalConvTranspose1d`].
#[derive(Config, Debug)]
pub struct CausalConvTranspose1dConfig {
    /// The input and output channels.
    pub channels: [usize; 2],

    /// The kernel's taps; at least the stride.
    pub kernel_size: usize,

    /// The stride: output frames per input frame.
    #[config(default = 1)]
    pub stride: usize,

    /// The channel groups; each divides both channel counts (their common
    /// count is a depthwise upsampler, Mimi's).
    #[config(default = 1)]
    pub groups: usize,

    /// Whether the convolution adds a bias.
    #[config(default = true)]
    pub bias: bool,
}

impl CausalConvTranspose1dMeta for CausalConvTranspose1dConfig {
    fn channels(&self) -> [usize; 2] {
        self.channels
    }

    fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    fn stride(&self) -> usize {
        self.stride
    }

    fn groups(&self) -> usize {
        self.groups
    }
}

/// The convolution, its weights burn's default initialization.
impl ModuleInit<CausalConvTranspose1d> for CausalConvTranspose1dConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a stride of 0
    /// or over the kernel.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<CausalConvTranspose1d> {
        if self.stride == 0 || self.stride > self.kernel_size {
            return Err(ConstraintError::out_of_range(
                "CausalConvTranspose1dConfig",
                "stride",
                self.stride,
                format!("1..={}, the kernel size", self.kernel_size),
            )
            .into());
        }
        Ok(CausalConvTranspose1d {
            conv: ConvTranspose1dConfig::new(self.channels, self.kernel_size)
                .with_stride(self.stride)
                .with_groups(self.groups)
                .with_bias(self.bias)
                .init(device),
        })
    }
}

/// A 1-d transposed convolution (an upsampler) over each frame's past: each
/// input frame becomes `s` output frames, and output frame `t` reads the input
/// up to frame `⌊t / s⌋` and no later.
///
/// burn's unpadded [`ConvTranspose1d`] gives `(T - 1) · s + K` frames; the
/// last [`right_trim`](CausalConvTranspose1dMeta::right_trim), `K - s`, are
/// trimmed, leaving `T · s`: the overlap a later input would add to. That is
/// the rule of the causal upsamplers in Encodec, DAC and Qwen3-TTS's codec.
///
/// The convolution is held as `conv`: checkpoints that hold it as
/// `conv.weight` and `conv.bias` load as they are.
///
/// # Streams
///
/// A stream runs it over its input in chunks of any size:
/// [`new_stream_context`](Self::new_stream_context) opens one, and
/// [`forward_stream`](Self::forward_stream) returns each chunk's `T · s`
/// frames, the `K - s` the chunks before it reached into added in. The
/// outputs, joined, are [`forward`](Self::forward)'s over the whole input.
///
/// Built by [`CausalConvTranspose1dConfig`].
#[derive(Module, Debug)]
pub struct CausalConvTranspose1d {
    /// The transposed convolution, unpadded.
    pub conv: ConvTranspose1d,
}

impl CausalConvTranspose1dMeta for CausalConvTranspose1d {
    fn channels(&self) -> [usize; 2] {
        self.conv.channels
    }

    fn kernel_size(&self) -> usize {
        self.conv.kernel_size
    }

    fn stride(&self) -> usize {
        self.conv.stride
    }

    fn groups(&self) -> usize {
        self.conv.groups
    }
}

impl CausalConvTranspose1d {
    /// The transposed convolution over each frame's past.
    ///
    /// # Arguments
    /// - `x`: `[B, C_in, T]`.
    ///
    /// # Returns
    /// `[B, C_out, T · s]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [c_in, c_out] = self.channels();
        #[cfg(not(any(debug_assertions, test)))]
        let _ = c_out;
        let [b, t] = unpack_shape_contract!(
            ["B", "C_in", "T"],
            &x.dims(),
            &["B", "T"],
            &[("C_in", c_in)]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = b;
        let y = self.conv.forward(x);
        let frames = t * self.stride();
        let y = if self.right_trim() > 0 {
            y.slice(s![.., .., 0..frames])
        } else {
            y
        };
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "C_out", "T_out"],
            &y.dims(),
            &[("B", b), ("C_out", c_out), ("T_out", frames)]
        );
        y
    }

    /// A stream through this convolution, not yet started.
    pub fn new_stream_context(&self) -> CausalConvTransposeStreamContext {
        CausalConvTransposeStreamContext::new(self.kernel_size(), self.stride())
    }

    /// The stream's next chunk: its `T · s` output frames, the overlap of
    /// the chunks before it added in. A stream ends with nothing more to
    /// return: [`forward`](Self::forward) trims what reaches past the input.
    ///
    /// # Arguments
    /// - `x`: `[B, C_in, T]`, `T ≥ 1`, the stream's next frames.
    /// - `ctx`: the stream, from
    ///   [`new_stream_context`](Self::new_stream_context).
    ///
    /// # Returns
    /// `[B, C_out, T · s]`.
    ///
    /// # Panics
    /// As [`CausalConvTransposeStreamContext::fold`]; or for a context of
    /// another convolution's geometry.
    pub fn forward_stream(
        &self,
        x: Tensor<3>,
        ctx: &mut CausalConvTransposeStreamContext,
    ) -> Tensor<3> {
        #[cfg(any(debug_assertions, test))]
        let [c_in, _] = self.channels();
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "C_in", "T"],
            &x.dims(),
            &[("C_in", c_in)]
        );
        assert!(
            ctx.kernel_size == self.kernel_size() && ctx.stride == self.stride(),
            "a causal transposed conv stream of another geometry"
        );
        let bias = self.conv.bias.as_ref().map(|bias| bias.val());
        ctx.fold(self.conv.forward(x), bias)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        tensor::{
            Distribution,
            TensorData,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::{
            assert_tensors_close,
            cpu_device,
            seeded_tensor,
        },
    };

    /// The oracle's input, `[1, 2, 7]`: `x[i] = ((5 i mod 13) - 6) / 4`.
    fn x(
        frames: usize,
        device: &Device,
    ) -> Tensor<3> {
        let values: Vec<f32> = (0..14).map(|i| ((i * 5 % 13) as f32 - 6.0) / 4.0).collect();
        Tensor::<1>::from_floats(values.as_slice(), device)
            .reshape([1, 2, 7])
            .slice(s![.., .., 0..frames])
    }

    /// The oracle's weights, by flat index: `w[i] = ((7 i mod 11) - 5) / 10`,
    /// `b[i] = i / 10 - 0.05`.
    fn fill(
        weight: &mut Param<Tensor<3>>,
        bias: &mut Option<Param<Tensor<1>>>,
    ) {
        let device = weight.device();
        let dims = weight.dims();
        let n = dims.iter().product::<usize>();
        let w: Vec<f32> = (0..n).map(|i| ((i * 7 % 11) as f32 - 5.0) / 10.0).collect();
        *weight = Param::from_tensor(Tensor::<1>::from_floats(w.as_slice(), &device).reshape(dims));
        let c = bias.as_ref().expect("a bias").dims()[0];
        let b: Vec<f32> = (0..c).map(|i| i as f32 / 10.0 - 0.05).collect();
        *bias = Some(Param::from_tensor(Tensor::from_floats(
            b.as_slice(),
            &device,
        )));
    }

    fn conv(config: CausalConv1dConfig) -> CausalConv1d {
        let mut conv = config.init(&cpu_device());
        fill(&mut conv.conv.weight, &mut conv.conv.bias);
        conv
    }

    fn check<const N: usize>(
        y: Tensor<3>,
        dims: [usize; 3],
        expected: [f32; N],
    ) {
        assert_eq!(y.dims(), dims);
        y.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected.to_vec(), dims),
            Tolerance::rel_abs(1e-5, 1e-6),
        );
    }

    /// The config and the module built from it answer the meta alike.
    #[test]
    fn test_meta() {
        let config = CausalConv1dConfig::new([4, 6], 3)
            .with_stride(2)
            .with_dilation(2)
            .with_groups(2)
            .with_padding(CausalPadding::Replicate);
        let module = config.init(&cpu_device());
        for meta in [&config as &dyn CausalConv1dMeta, &module] {
            assert_eq!(meta.channels(), [4, 6]);
            assert_eq!(
                [
                    meta.kernel_size(),
                    meta.stride(),
                    meta.dilation(),
                    meta.groups()
                ],
                [3, 2, 2, 2]
            );
            assert_eq!(meta.padding(), CausalPadding::Replicate);
            assert_eq!(meta.receptive_field(), 5);
            assert_eq!(meta.left_padding(), 3);
            assert_eq!([meta.right_padding(7), meta.right_padding(8)], [1, 0]);
            assert_eq!(meta.output_frames(7), 4);
        }

        let config = CausalConvTranspose1dConfig::new([4, 2], 6)
            .with_stride(3)
            .with_groups(2);
        let module = config.init(&cpu_device());
        for meta in [&config as &dyn CausalConvTranspose1dMeta, &module] {
            assert_eq!(meta.channels(), [4, 2]);
            assert_eq!(
                [meta.kernel_size(), meta.stride(), meta.groups()],
                [6, 3, 2]
            );
            assert_eq!(meta.right_trim(), 3);
        }
    }

    // The expected values: `qwen_tts` 0.1.1's own
    // `Qwen3TTSTokenizerV2CausalConvNet` and
    // `Qwen3TTSTokenizerV2CausalTransConvNet` in float64, with these weights.

    /// Dilated: `K = 3`, `d = 2`, 2 to 3 channels.
    #[test]
    fn test_dilated() {
        let conv = conv(CausalConv1dConfig::new([2, 3], 3).with_dilation(2));
        check(
            conv.forward(x(7, &cpu_device())),
            [1, 3, 7],
            [
                0.025, 0.375, -0.475, -0.4, 1.45, -1.075, -0.025, 0.275, 0.775, -0.425, -0.05,
                -0.05, -1.4, 1.475, -0.3, -0.475, 0.725, 0.575, -0.45, 0.75, -0.325,
            ],
        );
    }

    /// Strided: `K = 4`, `s = 2`, over 7 frames, so the last stride is padded
    /// whole.
    #[test]
    fn test_strided() {
        let conv = conv(CausalConv1dConfig::new([2, 2], 4).with_stride(2));
        check(
            conv.forward(x(7, &cpu_device())),
            [1, 2, 4],
            [0.425, 0.4, -0.725, -0.6, 0.575, 1.5, -2.0, -0.55],
        );
    }

    /// Grouped: depthwise, `K = 3`.
    #[test]
    fn test_grouped() {
        let conv = conv(CausalConv1dConfig::new([2, 2], 3).with_groups(2));
        check(
            conv.forward(x(7, &cpu_device())),
            [1, 2, 7],
            [
                0.25, -0.3, 0.45, 0.475, -0.8, 0.2, 0.225, -0.175, 0.5, 0.3, -0.95, 0.4, 0.45,
                0.175,
            ],
        );
    }

    /// Replicate padding: the first frame, repeated, before the input
    /// (`torch`'s `F.pad(mode="replicate")`, then the unpadded convolution).
    #[test]
    fn test_replicate() {
        let conv = conv(CausalConv1dConfig::new([2, 2], 3).with_padding(CausalPadding::Replicate));
        check(
            conv.forward(x(7, &cpu_device())),
            [1, 2, 7],
            [
                0.925, 1.275, 0.7, -0.525, -0.45, 0.6, 0.35, -0.175, 0.325, -0.6, -0.65, 0.6,
                -0.75, 0.825,
            ],
        );
    }

    /// Replicate pads the end with the last frame: a strided convolution of
    /// ones is ones' convolution, whatever the padding reads.
    #[test]
    fn test_replicate_end() {
        let device = cpu_device();
        let mut conv = CausalConv1dConfig::new([1, 1], 4)
            .with_stride(2)
            .with_bias(false)
            .with_padding(CausalPadding::Replicate)
            .init(&device);
        conv.conv.weight = Param::from_tensor(Tensor::ones([1, 1, 4], &device));
        check(
            conv.forward(Tensor::ones([1, 1, 5], &device)),
            [1, 1, 3],
            [4.0, 4.0, 4.0],
        );
    }

    /// The output reads only the past: a change at frame `j` changes no
    /// output before it.
    #[test]
    fn test_causal() {
        let device = cpu_device();
        let conv = conv(CausalConv1dConfig::new([2, 3], 3).with_dilation(2));
        let base = conv.forward(x(7, &device));
        let changed = conv.forward(
            x(7, &device).slice_assign(s![.., .., 4..5], Tensor::full([1, 2, 1], 9.0, &device)),
        );
        base.clone()
            .slice(s![.., .., 0..4])
            .into_data()
            .assert_eq(&changed.clone().slice(s![.., .., 0..4]).into_data(), true);
        assert_ne!(
            base.slice(s![.., .., 4..5]).into_data(),
            changed.slice(s![.., .., 4..5]).into_data()
        );
    }

    /// Transposed: `K = 4`, `s = 2`, so `K - s = 2` trimmed.
    #[test]
    fn test_transposed() {
        let mut conv = CausalConvTranspose1dConfig::new([2, 2], 4)
            .with_stride(2)
            .init(&cpu_device());
        fill(&mut conv.conv.weight, &mut conv.conv.bias);
        check(
            conv.forward(x(7, &cpu_device()).slice(s![.., .., 0..3])),
            [1, 2, 6],
            [
                0.4, -0.125, 0.8, -1.6, -0.375, 0.65, 0.05, 0.35, -0.45, 0.45, -0.575, -0.375,
            ],
        );
    }

    /// The Qwen3-TTS toy codec's convolutions
    /// (`tools/gen_qwen3_tts_toy_fixtures.py`), each on its own input: the
    /// pre-conv (`K = 3`) and the vocoder's first (`K = 7`).
    #[cfg(feature = "store")]
    #[test]
    fn test_qwen3_tts_toy_convs() {
        use crate::support::testing::{
            ToyFixture,
            assert_io_close,
        };

        let device = cpu_device();
        let path = crate::support::testing::testdata!("qwen3_tts/toy.safetensors");
        let fixture = ToyFixture::load_io(path, &device);
        for (name, input, output) in [
            (
                "decoder.pre_conv",
                "io.codec.quantized",
                "io.codec.pre_conv",
            ),
            (
                "decoder.decoder.0",
                "io.codec.upsample_1",
                "io.codec.decoder_0",
            ),
        ] {
            let weight = fixture.io::<3>(&format!("{name}.conv.weight"));
            let [c_out, c_in, k] = weight.dims();
            let mut conv = CausalConv1dConfig::new([c_in, c_out], k).init(&device);
            conv.conv.weight = Param::from_tensor(weight);
            conv.conv.bias = Some(Param::from_tensor(
                fixture.io::<1>(&format!("{name}.conv.bias")),
            ));
            assert_io_close(name, conv.forward(fixture.io(input)), fixture.io(output));
        }
    }

    /// `x` through `conv` as a stream, in chunks of `sizes` (cycled), then
    /// finished: the outputs joined.
    fn streamed(
        conv: &CausalConv1d,
        x: Tensor<3>,
        sizes: &[usize],
    ) -> Tensor<3> {
        let t = x.dims()[2];
        let mut ctx = conv.new_stream_context();
        let mut outputs = Vec::new();
        let (mut at, mut i) = (0, 0);
        while at < t {
            let n = sizes[i % sizes.len()].min(t - at);
            outputs.extend(conv.forward_stream(x.clone().slice(s![.., .., at..at + n]), &mut ctx));
            at += n;
            i += 1;
        }
        outputs.extend(conv.finish_stream(&mut ctx));
        Tensor::cat(outputs, 2)
    }

    /// A stream's outputs are the whole forward's, whatever the chunks: plain,
    /// dilated and grouped, strided with each padding (Mimi's `k4 s2`
    /// replicate downsample; a `SEANet` `k16 s8` stage), over lengths that end
    /// on a partial stride and that do not.
    #[test]
    fn test_stream_matches_forward() {
        let device = cpu_device();
        let configs = [
            CausalConv1dConfig::new([3, 4], 7),
            CausalConv1dConfig::new([4, 6], 3)
                .with_dilation(2)
                .with_groups(2),
            CausalConv1dConfig::new([3, 3], 4)
                .with_stride(2)
                .with_padding(CausalPadding::Replicate),
            CausalConv1dConfig::new([2, 4], 16).with_stride(8),
            CausalConv1dConfig::new([2, 2], 3)
                .with_stride(3)
                .with_padding(CausalPadding::Replicate),
            CausalConv1dConfig::new([2, 2], 1),
        ];
        for (i, config) in configs.iter().enumerate() {
            let conv = config.init(&device);
            for t in [1, 5, 12, 13, 24] {
                let x = seeded_tensor(
                    (10 * i + t) as u64,
                    [2, config.channels[0], t],
                    Distribution::Normal(0.0, 1.0),
                    &device,
                );
                let whole = conv.forward(x.clone());
                for sizes in [&[1][..], &[2], &[3], &[5], &[1, 4, 2], &[t]] {
                    assert_tensors_close(
                        &streamed(&conv, x.clone(), sizes),
                        &whole,
                        Tolerance::rel_abs(1e-5, 1e-6),
                    );
                }
            }
        }
    }

    /// A chunk short of a stride completes no output; the stream holds it.
    #[test]
    fn test_stream_holds_partial_strides() {
        let device = cpu_device();
        let conv = CausalConv1dConfig::new([1, 1], 4)
            .with_stride(2)
            .init(&device);
        let mut ctx = conv.new_stream_context();
        let x = Tensor::<3>::ones([1, 1, 1], &device);
        assert!(conv.forward_stream(x.clone(), &mut ctx).is_none());
        assert_eq!(
            conv.forward_stream(x.clone(), &mut ctx)
                .expect("a whole stride")
                .dims(),
            [1, 1, 1]
        );
        assert!(conv.forward_stream(x, &mut ctx).is_none());
        assert_eq!(
            conv.finish_stream(&mut ctx)
                .expect("the padded stride")
                .dims(),
            [1, 1, 1]
        );
    }

    /// A transposed stream's outputs are the whole forward's, whatever the
    /// chunks: overlapping (`K = 2s`, `K = s + 3`), one stride (`K = s`, no
    /// overlap) and depthwise (Mimi's upsampler).
    #[test]
    fn test_transposed_stream_matches_forward() {
        let device = cpu_device();
        let configs = [
            CausalConvTranspose1dConfig::new([3, 2], 4).with_stride(2),
            CausalConvTranspose1dConfig::new([2, 3], 4),
            CausalConvTranspose1dConfig::new([2, 2], 3).with_stride(3),
            CausalConvTranspose1dConfig::new([4, 4], 4)
                .with_stride(2)
                .with_groups(4),
            CausalConvTranspose1dConfig::new([2, 2], 16)
                .with_stride(8)
                .with_bias(false),
        ];
        for (i, config) in configs.iter().enumerate() {
            let conv = config.init(&device);
            for t in [1, 5, 12] {
                let x = seeded_tensor(
                    (10 * i + t) as u64,
                    [2, config.channels[0], t],
                    Distribution::Normal(0.0, 1.0),
                    &device,
                );
                let whole = conv.forward(x.clone());
                for sizes in [&[1][..], &[2], &[3], &[1, 4, 2], &[t]] {
                    let mut ctx = conv.new_stream_context();
                    let mut outputs = Vec::new();
                    let (mut at, mut j) = (0, 0);
                    while at < t {
                        let n = sizes[j % sizes.len()].min(t - at);
                        outputs.push(
                            conv.forward_stream(x.clone().slice(s![.., .., at..at + n]), &mut ctx),
                        );
                        at += n;
                        j += 1;
                    }
                    assert_tensors_close(
                        &Tensor::cat(outputs, 2),
                        &whole,
                        Tolerance::rel_abs(1e-5, 1e-6),
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "a causal conv stream of another geometry")]
    fn test_stream_of_another_geometry() {
        let device = cpu_device();
        let conv = CausalConv1dConfig::new([1, 1], 4).init(&device);
        let other = CausalConv1dConfig::new([1, 1], 3).init(&device);
        let mut ctx = other.new_stream_context();
        let _ = conv.forward_stream(Tensor::ones([1, 1, 2], &device), &mut ctx);
    }

    /// A stride of 0, or past what the kernel covers, is refused.
    #[test]
    fn test_illegal_strides() {
        let device = cpu_device();
        for stride in [0, 6] {
            ErrorMatcher::illegal()
                .constraint("CausalConv1dConfig", "stride")
                .message_contains("outside 1..=5")
                .assert_err(
                    &ModuleInit::<CausalConv1d>::try_init(
                        &CausalConv1dConfig::new([2, 2], 3)
                            .with_dilation(2)
                            .with_stride(stride),
                        &device,
                    )
                    .map(|_| ()),
                );
        }
        for stride in [0, 5] {
            ErrorMatcher::illegal()
                .constraint("CausalConvTranspose1dConfig", "stride")
                .message_contains("outside 1..=4")
                .assert_err(
                    &ModuleInit::<CausalConvTranspose1d>::try_init(
                        &CausalConvTranspose1dConfig::new([2, 2], 4).with_stride(stride),
                        &device,
                    )
                    .map(|_| ()),
                );
        }
    }
}
