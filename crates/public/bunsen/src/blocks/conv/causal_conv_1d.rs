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

use crate::{
    burner::module::ModuleInit,
    contracts::{
        assert_shape_contract_periodically,
        unpack_shape_contract,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// What a [`CausalConv1d`] pads its input with.
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum CausalPadding {
    /// Zeros (`torch`'s `constant`).
    Zeros,

    /// The edge's own frame, repeated (`torch`'s `replicate`).
    Replicate,
}

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
            return Err(BunsenError::illegal(format!(
                "a causal convolution's stride is in 1..={}; got {}",
                self.receptive_field(),
                self.stride
            )));
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
        let [b, c, t] = x.dims();
        let edge = |at: usize, frames: usize| match self.padding {
            CausalPadding::Zeros => Tensor::zeros([b, c, frames], (&x.device(), x.dtype())),
            CausalPadding::Replicate => x.clone().slice(s![.., .., at..=at]).repeat_dim(2, frames),
        };
        let mut parts = Vec::with_capacity(3);
        if before > 0 {
            parts.push(edge(0, before));
        }
        parts.push(x.clone());
        if after > 0 {
            parts.push(edge(t - 1, after));
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
        let [b, t] = unpack_shape_contract!(
            ["B", "C_in", "T"],
            &x.dims(),
            &["B", "T"],
            &[("C_in", c_in)]
        );
        assert!(t >= 1, "a causal convolution over at least one frame");
        let x = self.pad(x, self.left_padding(), self.right_padding(t));
        let y = self.conv.forward(x);
        assert_shape_contract_periodically!(
            ["B", "C_out", "T_out"],
            &y.dims(),
            &[("B", b), ("C_out", c_out), ("T_out", self.output_frames(t))]
        );
        y
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
            return Err(BunsenError::illegal(format!(
                "a causal transposed convolution's stride is in 1..={}; got {}",
                self.kernel_size, self.stride
            )));
        }
        Ok(CausalConvTranspose1d {
            conv: ConvTranspose1dConfig::new(self.channels, self.kernel_size)
                .with_stride(self.stride)
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
        let [b, t] = unpack_shape_contract!(
            ["B", "C_in", "T"],
            &x.dims(),
            &["B", "T"],
            &[("C_in", c_in)]
        );
        let y = self.conv.forward(x);
        let frames = t * self.stride();
        let y = if self.right_trim() > 0 {
            y.slice(s![.., .., 0..frames])
        } else {
            y
        };
        assert_shape_contract_periodically!(
            ["B", "C_out", "T_out"],
            &y.dims(),
            &[("B", b), ("C_out", c_out), ("T_out", frames)]
        );
        y
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        tensor::{
            TensorData,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::cpu_device,
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

        let config = CausalConvTranspose1dConfig::new([4, 2], 6).with_stride(3);
        let module = config.init(&cpu_device());
        for meta in [&config as &dyn CausalConvTranspose1dMeta, &module] {
            assert_eq!(meta.channels(), [4, 2]);
            assert_eq!([meta.kernel_size(), meta.stride()], [6, 3]);
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
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/qwen3_tts/toy.safetensors");
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

    /// A stride of 0, or past what the kernel covers, is refused.
    #[test]
    fn test_illegal_strides() {
        let device = cpu_device();
        for stride in [0, 6] {
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .message_contains("stride is in 1..=5")
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
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .message_contains("stride is in 1..=4")
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
