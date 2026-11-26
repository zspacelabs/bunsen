//! `SKNet`'s selective kernel.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{
            Conv2d,
            Conv2dConfig,
        },
    },
    prelude::Tensor,
    tensor::{
        Device,
        activation::softmax,
    },
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        conv::{
            ConvBlock2d,
            ConvBlock2dConfig,
            ConvBlock2dMeta,
        },
        norm::BunsenNormalizationConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
    support::math::make_divisible,
};

/// `SKNet`'s selective kernel, the config of a [`SelectiveKernel`] (timm's
/// `SelectiveKernel` and `SelectiveKernelAttn`).
///
/// Each path is a conv block of its own receptive field ([`paths`](Self::paths):
/// timm's are `3x3` convs dilated 1 and 2, its `[3, 5]` kernels kept `3x3`),
/// over the whole input or, under [`split_input`](Self::split_input), its own
/// share of the channels. The paths' sum, averaged over the map, goes through
/// a `1x1` conv down to the attention width, a norm and the activation
/// ([`reduce`](Self::reduce): timm's `fc_reduce`, `bn`, `act`), and a `1x1`
/// conv to every path's channels ([`select`](Self::select): `fc_select`). A
/// softmax across the paths of each channel weighs them, and the output is
/// their weighted sum.
#[derive(Config, Debug)]
pub struct SelectiveKernelConfig {
    /// The paths: each `in_channels` (or its share, under `split_input`) to
    /// `out_channels`, all of one stride.
    pub paths: Vec<ConvBlock2dConfig>,

    /// Whether the input's channels split evenly across the paths (timm's
    /// `split_input`); else every path reads all of them.
    pub split_input: bool,

    /// The attention's reduction: a `1x1` conv (no bias), its norm and its
    /// activation, `out_channels` to the attention width.
    pub reduce: ConvBlock2dConfig,

    /// The attention's selection: a `1x1` conv (no bias), the attention width
    /// to `out_channels` per path.
    pub select: Conv2dConfig,
}

impl SelectiveKernelConfig {
    /// timm's `SelectiveKernel` from `channels[0]` to `channels[1]`: a path
    /// per kernel of `kernel_size` (timm's `[3, 5]`), each kept `3x3` and
    /// dilated `dilation * (k - 1) / 2` under `keep_3x3` (else `k x k`,
    /// dilated `dilation`), padded as timm's `get_padding`, at `stride`, in
    /// `min(out_channels, groups)` groups; the paths' and the attention's norm
    /// and activation `norm` and `act`; the input split across the paths under
    /// `split_input`; and an attention width of
    /// `make_divisible(out_channels * rd_ratio, rd_divisor)` (timm's
    /// `rd_ratio`, `rd_divisor`; the rounding limit of 0.9 is timm's own).
    #[allow(clippy::too_many_arguments)]
    pub fn timm(
        channels: [usize; 2],
        kernel_size: [usize; 2],
        keep_3x3: bool,
        stride: usize,
        dilation: usize,
        groups: usize,
        rd_ratio: f64,
        rd_divisor: usize,
        split_input: bool,
        norm: BunsenNormalizationConfig,
        act: BunsenActivationConfig,
    ) -> Self {
        let [in_channels, out_channels] = channels;
        let num_paths = kernel_size.len();
        let path_in = match split_input {
            true => in_channels / num_paths,
            false => in_channels,
        };
        let groups = groups.min(out_channels);
        let block = |conv: Conv2dConfig| {
            ConvBlock2dConfig::new(conv)
                .with_norm(Some(norm.clone()))
                .with_act(Some(act.clone()))
                .match_norm_features()
        };
        let paths = kernel_size
            .iter()
            .map(|&k| {
                let (k, d) = match keep_3x3 {
                    true => (3, dilation * (k - 1) / 2),
                    false => (k, dilation),
                };
                // timm's `get_padding(k, stride, d)`.
                let pad = ((stride - 1) + d * (k - 1)) / 2;
                block(
                    Conv2dConfig::new([path_in, out_channels], [k, k])
                        .with_stride([stride, stride])
                        .with_dilation([d, d])
                        .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
                        .with_groups(groups)
                        .with_bias(false),
                )
            })
            .collect();
        let attn = make_divisible(out_channels as f64 * rd_ratio, rd_divisor, None, 0.9);
        Self {
            paths,
            split_input,
            reduce: block(Conv2dConfig::new([out_channels, attn], [1, 1]).with_bias(false)),
            select: Conv2dConfig::new([attn, out_channels * num_paths], [1, 1]).with_bias(false),
        }
    }

    /// The input channels: the paths' (all of them, under `split_input`).
    pub fn in_channels(&self) -> usize {
        let first = self.paths.first().map_or(0, ConvBlock2dMeta::in_channels);
        match self.split_input {
            true => first * self.paths.len(),
            false => first,
        }
    }

    /// The output channels: one path's.
    pub fn out_channels(&self) -> usize {
        self.paths.first().map_or(0, ConvBlock2dMeta::out_channels)
    }

    /// The stride: the paths'.
    pub fn stride(&self) -> usize {
        self.paths.first().map_or(1, |p| p.stride()[0])
    }

    /// Checks the shape: at least one path, all of one input, output and
    /// stride, and an attention from one path's channels to all of theirs.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when it isn't.
    pub fn try_validate(&self) -> BunsenResult<()> {
        let illegal = |message: alloc::string::String| {
            Err(BunsenError::illegal(format!(
                "SelectiveKernelConfig: {message}"
            )))
        };
        let Some(first) = self.paths.first() else {
            return illegal("no paths".into());
        };
        for (idx, path) in self.paths.iter().enumerate() {
            if path.in_channels() != first.in_channels()
                || path.out_channels() != first.out_channels()
                || path.stride() != first.stride()
            {
                return illegal(format!(
                    "path {idx} is {} to {} at stride {}; path 0 {} to {} at {}",
                    path.in_channels(),
                    path.out_channels(),
                    path.stride()[0],
                    first.in_channels(),
                    first.out_channels(),
                    first.stride()[0]
                ));
            }
        }
        let (c, n) = (self.out_channels(), self.paths.len());
        let attn = self.reduce.out_channels();
        if self.reduce.in_channels() != c || self.select.channels != [attn, c * n] {
            return illegal(format!(
                "the attention is {} to {attn}, then {:?}; it reads {c} channels and weighs {}",
                self.reduce.in_channels(),
                self.select.channels,
                c * n
            ));
        }
        Ok(())
    }
}

impl ModuleInit<SelectiveKernel> for SelectiveKernelConfig {
    /// Builds the paths and the attention.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](SelectiveKernelConfig::try_validate), and as the
    /// conv blocks' `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SelectiveKernel> {
        self.try_validate()?;
        Ok(SelectiveKernel {
            paths: self
                .paths
                .iter()
                .map(|p| p.try_init(device))
                .collect::<BunsenResult<_>>()?,
            reduce: self.reduce.try_init(device)?,
            select: self.select.init(device),
            split_input: self.split_input,
        })
    }
}

/// `SKNet`'s selective kernel; built by [`SelectiveKernelConfig`].
#[derive(Module, Debug)]
pub struct SelectiveKernel {
    /// The paths.
    pub paths: Vec<ConvBlock2d>,

    /// The attention's reduction, its norm and activation.
    pub reduce: ConvBlock2d,

    /// The attention's selection.
    pub select: Conv2d,

    /// Whether the input's channels split across the paths.
    #[module(skip)]
    pub split_input: bool,
}

impl SelectiveKernel {
    /// The input channels.
    pub fn in_channels(&self) -> usize {
        let first = self.paths[0].in_channels();
        match self.split_input {
            true => first * self.paths.len(),
            false => first,
        }
    }

    /// The output channels: one path's.
    pub fn out_channels(&self) -> usize {
        self.paths[0].out_channels()
    }

    /// The stride.
    pub fn stride(&self) -> usize {
        self.paths[0].stride()[0]
    }

    /// The selective kernel, with `f` run on its output (where a regularizer
    /// goes).
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, height, width]`.
    /// - `f`: an endofunction over the output.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn map_forward<F>(
        &self,
        input: Tensor<4>,
        f: F,
    ) -> Tensor<4>
    where
        F: FnOnce(Tensor<4>) -> Tensor<4>,
    {
        let n = self.paths.len();
        let outs: Vec<Tensor<4>> = match self.split_input {
            true => input
                .split(self.paths[0].in_channels(), 1)
                .into_iter()
                .zip(&self.paths)
                .map(|(x, path)| path.forward(x))
                .collect(),
            false => self
                .paths
                .iter()
                .map(|path| path.forward(input.clone()))
                .collect(),
        };
        let x: Tensor<5> = Tensor::stack(outs, 1);
        let [b, _, c, h, w] = x.dims();
        let gap = x
            .clone()
            .sum_dim(1)
            .reshape([b, c, h, w])
            .mean_dim(2)
            .mean_dim(3);
        let attn = self.select.forward(self.reduce.forward(gap));
        let attn = softmax(attn.reshape([b, n, c, 1, 1]), 1);
        let out = f((x * attn).sum_dim(1).reshape([b, c, h, w]));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "out_channels", "height", "width"],
            &out.dims(),
            &[
                ("batch", b),
                ("out_channels", c),
                ("height", h),
                ("width", w)
            ]
        );
        out
    }

    /// The selective kernel.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, in_channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        self.map_forward(input, |x| x)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::BatchNormConfig,
        tensor::{
            Distribution,
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

    fn bn() -> BunsenNormalizationConfig {
        BunsenNormalizationConfig::Batch(BatchNormConfig::new(0))
    }

    /// timm's widths: `skresnet18`'s first mixer, 64 to 64 split across two
    /// paths of 32 inputs, dilated 1 and 2, attends over
    /// `make_divisible(64 * 0.125, 16)`, 16.
    #[test]
    fn test_timm_widths() {
        let config = SelectiveKernelConfig::timm(
            [64, 64],
            [3, 5],
            true,
            1,
            1,
            1,
            0.125,
            16,
            true,
            bn(),
            BunsenActivationConfig::Relu,
        );
        let dilations: Vec<_> = config.paths.iter().map(|p| p.dilation()[0]).collect();
        assert_eq!(dilations, [1, 2]);
        assert_eq!(config.paths[0].in_channels(), 32);
        assert_eq!(config.reduce.out_channels(), 16);
        assert_eq!(config.select.channels, [16, 128]);
        assert_eq!((config.in_channels(), config.out_channels()), (64, 64));
        config.try_validate().unwrap();
    }

    /// Without `keep_3x3`, each path is its kernel undilated: `3x3` and `5x5`,
    /// padded 1 and 2 (timm's `get_padding`), at stride 2 too.
    #[test]
    fn test_kernels_not_kept_3x3() {
        let config = SelectiveKernelConfig::timm(
            [8, 8],
            [3, 5],
            false,
            2,
            1,
            1,
            0.5,
            2,
            false,
            bn(),
            BunsenActivationConfig::Relu,
        );
        let shapes: Vec<_> = config
            .paths
            .iter()
            .map(|p| (p.kernel_size()[0], p.dilation()[0], p.padding()))
            .collect();
        assert_eq!(
            shapes,
            [
                (3, 1, PaddingConfig2d::Explicit(1, 1, 1, 1)),
                (5, 1, PaddingConfig2d::Explicit(2, 2, 2, 2)),
            ]
        );
    }

    /// The forward by hand: each path over its share of the input, a softmax
    /// over the paths of the attention of their sum, their weighted sum.
    #[test]
    fn test_forward_by_hand() {
        let device = cpu_device();
        let config = SelectiveKernelConfig::timm(
            [8, 6],
            [3, 5],
            true,
            2,
            1,
            2,
            0.5,
            2,
            true,
            bn(),
            BunsenActivationConfig::Relu,
        );
        let sk: SelectiveKernel = config.init(&device);
        let x = Tensor::<4>::random([2, 8, 7, 7], Distribution::Default, &device);
        let halves = x.clone().split(4, 1);
        let a = sk.paths[0].forward(halves[0].clone());
        let b = sk.paths[1].forward(halves[1].clone());
        let gap = (a.clone() + b.clone()).mean_dim(2).mean_dim(3);
        let attn = softmax(
            sk.select
                .forward(sk.reduce.forward(gap))
                .reshape([2, 2, 6, 1, 1]),
            1,
        );
        let wa = attn.clone().narrow(1, 0, 1).reshape([2, 6, 1, 1]);
        let wb = attn.narrow(1, 1, 1).reshape([2, 6, 1, 1]);
        let want = a * wa + b * wb;
        let got = sk.forward(x);
        assert_eq!(got.dims(), [2, 6, 4, 4]);
        got.into_data()
            .assert_approx_eq::<f32>(&want.into_data(), Tolerance::rel_abs(1e-6, 1e-6));
    }

    #[test]
    fn test_illegal_shapes() {
        let mut config = SelectiveKernelConfig::timm(
            [8, 8],
            [3, 5],
            true,
            1,
            1,
            1,
            0.5,
            2,
            false,
            bn(),
            BunsenActivationConfig::Relu,
        );
        config.paths.truncate(1);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("then [4, 16]; it reads 8 channels and weighs 8")
            .assert_err(&config.try_validate());
    }
}
