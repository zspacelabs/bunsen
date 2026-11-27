//! `Res2Net`'s scale split.

use alloc::{
    format,
    vec::Vec,
};

use burn::{
    config::Config,
    module::Module,
    nn::pool::{
        AvgPool2d,
        AvgPool2dConfig,
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    blocks::conv::{
        ConvBlock2d,
        ConvBlock2dConfig,
        ConvBlock2dMeta,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
};

/// `Res2Net`'s scale split, the config of a [`Res2Conv`] (timm's
/// `Bottle2neck`, its `convs`, `bns` and `pool`).
///
/// The input's `width * scale` channels split into `scale` chunks of `width`.
/// Each of the first `scale - 1` (one, at a `scale` of 1) goes through its
/// own conv block, after the first adding the previous block's output when
/// [`hierarchical`](Self::hierarchical); the last chunk passes through, or
/// through [`pool`](Self::pool). The outputs concatenate back to
/// `width * scale`.
#[derive(Config, Debug)]
pub struct Res2ConvConfig {
    /// The chunks.
    pub scale: usize,

    /// The branch convs, `width` to `width` each, all of one stride: `scale -
    /// 1` of them, or one at a `scale` of 1.
    pub convs: Vec<ConvBlock2dConfig>,

    /// Whether each branch after the first adds the previous branch's output
    /// to its chunk: `false` in a unit that strides or reshapes (timm's
    /// `is_first`), where the chunks are convolved apart.
    #[config(default = true)]
    pub hierarchical: bool,

    /// The pool over the last chunk, of the convs' stride (timm's
    /// `AvgPool2d(3, stride, padding=1)`, in a unit that strides or
    /// reshapes); none passes it through.
    #[config(default = "None")]
    pub pool: Option<AvgPool2dConfig>,
}

impl Res2ConvConfig {
    /// timm's branch pool at `stride`: a `3x3` average, padded 1, the padding
    /// counted (`nn.AvgPool2d`'s default, which the original weights need).
    pub fn timm_pool(stride: usize) -> AvgPool2dConfig {
        AvgPool2dConfig::new([3, 3])
            .with_strides([stride, stride])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_count_include_pad(true)
    }

    /// The chunk width: the branch convs' input channels.
    pub fn width(&self) -> usize {
        self.convs.first().map_or(0, ConvBlock2dMeta::in_channels)
    }

    /// The input and output channels: `width * scale`.
    pub fn channels(&self) -> usize {
        self.width() * self.scale
    }

    /// The stride: the branch convs'.
    pub fn stride(&self) -> usize {
        self.convs.first().map_or(1, |c| c.stride()[0])
    }

    /// Checks the split: a nonzero scale, `max(1, scale - 1)` branch convs,
    /// each `width` to `width` at one stride, and a pool only over a last
    /// chunk, of that stride.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when it isn't.
    pub fn try_validate(&self) -> BunsenResult<()> {
        let illegal = |field: &'static str, message: alloc::string::String| {
            Err(ConstraintError::custom("Res2ConvConfig", field, message).into())
        };
        if self.scale == 0 {
            return Err(ConstraintError::zero_or_empty("Res2ConvConfig", "scale").into());
        }
        let want = self.scale.saturating_sub(1).max(1);
        if self.convs.len() != want {
            return illegal(
                "convs",
                format!(
                    "{} branch convs at a scale of {}; it takes {want}",
                    self.convs.len(),
                    self.scale
                ),
            );
        }
        let (width, stride) = (self.width(), self.stride());
        for (idx, conv) in self.convs.iter().enumerate() {
            if conv.in_channels() != width || conv.out_channels() != width {
                return illegal(
                    "convs",
                    format!(
                        "branch conv {idx} is {} to {}; every branch is {width} to {width}",
                        conv.in_channels(),
                        conv.out_channels()
                    ),
                );
            }
            if conv.stride()[0] != stride {
                return illegal(
                    "convs",
                    format!(
                        "branch conv {idx} strides {}; the first {stride}",
                        conv.stride()[0]
                    ),
                );
            }
        }
        match &self.pool {
            Some(_) if self.scale == 1 => {
                illegal("pool", "a pool, at a scale of 1: no last chunk".into())
            }
            Some(pool) if pool.strides[0] != stride => illegal(
                "pool",
                format!(
                    "the pool strides {}; the branches {stride}",
                    pool.strides[0]
                ),
            ),
            None if self.scale > 1 && stride != 1 => illegal(
                "pool",
                format!("no pool at stride {stride}: the last chunk would keep its size"),
            ),
            _ => Ok(()),
        }
    }
}

impl ModuleInit<Res2Conv> for Res2ConvConfig {
    /// Builds the branches.
    ///
    /// # Errors
    ///
    /// As [`try_validate`](Res2ConvConfig::try_validate), and as the convs'
    /// `try_init`.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Res2Conv> {
        self.try_validate()?;
        Ok(Res2Conv {
            convs: self
                .convs
                .iter()
                .map(|c| c.try_init(device))
                .collect::<BunsenResult<_>>()?,
            pool: self.pool.as_ref().map(AvgPool2dConfig::init),
            scale: self.scale,
            hierarchical: self.hierarchical,
        })
    }
}

/// `Res2Net`'s scale split; built by [`Res2ConvConfig`].
#[derive(Module, Debug)]
pub struct Res2Conv {
    /// The branch convs.
    pub convs: Vec<ConvBlock2d>,

    /// The pool over the last chunk, if any.
    pub pool: Option<AvgPool2d>,

    /// The chunks.
    #[module(skip)]
    pub scale: usize,

    /// Whether each branch after the first adds the previous one's output.
    #[module(skip)]
    pub hierarchical: bool,
}

impl Res2Conv {
    /// The chunk width.
    pub fn width(&self) -> usize {
        self.convs[0].in_channels()
    }

    /// The input and output channels: `width * scale`.
    pub fn channels(&self) -> usize {
        self.width() * self.scale
    }

    /// The stride.
    pub fn stride(&self) -> usize {
        self.convs[0].stride()[0]
    }

    /// The split, with `f` run on the concatenated output (where a
    /// regularizer goes: timm's `drop_block`).
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, width * scale, height, width]`.
    /// - `f`: an endofunction over the output.
    ///
    /// # Returns
    ///
    /// `[batch, width * scale, ceil(height / stride), ceil(width / stride)]`.
    pub fn map_forward<F>(
        &self,
        input: Tensor<4>,
        f: F,
    ) -> Tensor<4>
    where
        F: FnOnce(Tensor<4>) -> Tensor<4>,
    {
        let [batch] = crate::contracts::unpack_shape_contract!(
            ["batch", "channels", _, _],
            &input.dims(),
            &["batch"],
            &[("channels", self.channels())]
        );
        let chunks = input.split(self.width(), 1);
        let mut outs = Vec::with_capacity(self.scale);
        let mut prev: Option<Tensor<4>> = None;
        for (chunk, conv) in chunks.iter().zip(&self.convs) {
            let x = match prev.take() {
                Some(prev) if self.hierarchical => prev + chunk.clone(),
                _ => chunk.clone(),
            };
            let x = conv.forward(x);
            prev = Some(x.clone());
            outs.push(x);
        }
        if self.scale > 1 {
            let last = chunks[self.scale - 1].clone();
            outs.push(match &self.pool {
                Some(pool) => pool.forward(last),
                None => last,
            });
        }
        let x = f(Tensor::cat(outs, 1));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "channels", _, _],
            &x.dims(),
            &[("batch", batch), ("channels", self.channels())]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;
        x
    }

    /// The split.
    ///
    /// # Arguments
    ///
    /// - `input`: `[batch, width * scale, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, width * scale, ceil(height / stride), ceil(width / stride)]`.
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        self.map_forward(input, |x| x)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use burn::{
        nn::{
            PaddingConfig2d,
            conv::Conv2dConfig,
        },
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        errors::testing::ErrorMatcher,
        support::testing::cpu_device,
    };

    /// A `3x3` branch conv, `width` to `width`, at `stride`, no norm.
    fn branch(
        width: usize,
        stride: usize,
    ) -> ConvBlock2dConfig {
        ConvBlock2dConfig::new(
            Conv2dConfig::new([width, width], [3, 3])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false),
        )
        .with_norm(None)
    }

    /// The split by hand: each branch over its chunk plus the previous
    /// branch's output, the last chunk passed through.
    #[test]
    fn test_hierarchical_split() {
        let device = cpu_device();
        let config = Res2ConvConfig::new(3, vec![branch(2, 1), branch(2, 1)]);
        let res2: Res2Conv = config.init(&device);
        let x = Tensor::<4>::random([1, 6, 5, 5], Distribution::Default, &device);
        let chunks = x.clone().split(2, 1);
        let a = res2.convs[0].forward(chunks[0].clone());
        let b = res2.convs[1].forward(a.clone() + chunks[1].clone());
        let want = Tensor::cat(vec![a, b, chunks[2].clone()], 1);
        res2.forward(x)
            .into_data()
            .assert_approx_eq::<f32>(&want.into_data(), Tolerance::rel_abs(1e-6, 1e-6));
    }

    /// In a striding unit, the chunks are convolved apart and the last one
    /// pooled to the branches' size.
    #[test]
    fn test_first_unit_split() {
        let device = cpu_device();
        let config = Res2ConvConfig::new(3, vec![branch(2, 2), branch(2, 2)])
            .with_hierarchical(false)
            .with_pool(Some(Res2ConvConfig::timm_pool(2)));
        let res2: Res2Conv = config.init(&device);
        let x = Tensor::<4>::random([1, 6, 5, 5], Distribution::Default, &device);
        let chunks = x.clone().split(2, 1);
        let a = res2.convs[0].forward(chunks[0].clone());
        let b = res2.convs[1].forward(chunks[1].clone());
        let c = res2.pool.as_ref().unwrap().forward(chunks[2].clone());
        let want = Tensor::cat(vec![a, b, c], 1);
        let got = res2.forward(x);
        assert_eq!(got.dims(), [1, 6, 3, 3]);
        got.into_data()
            .assert_approx_eq::<f32>(&want.into_data(), Tolerance::rel_abs(1e-6, 1e-6));
    }

    #[test]
    fn test_illegal_splits() {
        let illegal = |config: Res2ConvConfig, message: &str| {
            ErrorMatcher::illegal()
                .message_contains(message)
                .assert_err(&config.try_validate());
        };
        illegal(
            Res2ConvConfig::new(4, vec![branch(2, 1), branch(2, 1)]),
            "2 branch convs at a scale of 4; it takes 3",
        );
        illegal(
            Res2ConvConfig::new(3, vec![branch(2, 2), branch(2, 1)])
                .with_pool(Some(Res2ConvConfig::timm_pool(2))),
            "branch conv 1 strides 1; the first 2",
        );
        illegal(
            Res2ConvConfig::new(3, vec![branch(2, 2), branch(2, 2)]),
            "no pool at stride 2",
        );
        illegal(
            Res2ConvConfig::new(1, vec![branch(2, 1)])
                .with_pool(Some(Res2ConvConfig::timm_pool(1))),
            "a pool, at a scale of 1",
        );
    }
}
