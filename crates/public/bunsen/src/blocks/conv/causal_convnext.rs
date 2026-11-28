//! `CausalConvNext1d`: a causal 1-d `ConvNeXt` block.

use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        Linear,
        LinearConfig,
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    tensor::Device,
};

use crate::{
    blocks::{
        activation::{
            BunsenActivation,
            BunsenActivationConfig,
        },
        norm::{
            BunsenLayerNormConfig,
            BunsenNormalization,
            BunsenNormalizationConfig,
        },
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
    },
    ops::transformers::linear_attention::causal_depthwise_conv1d,
};

/// Common meta for [`CausalConvNext1d`] and [`CausalConvNext1dConfig`].
pub trait CausalConvNext1dMeta {
    /// The channels, read and written.
    fn width(&self) -> usize;

    /// The pointwise expansion's hidden width.
    fn hidden(&self) -> usize;

    /// The depthwise kernel's taps.
    fn kernel(&self) -> usize;

    /// Whether the branch is scaled per channel, by `γ`, before the residual
    /// add (`ConvNeXt`'s layer scale).
    fn layer_scale(&self) -> bool;
}

/// Config for [`CausalConvNext1d`].
#[derive(Config, Debug)]
pub struct CausalConvNext1dConfig {
    /// The channels.
    pub width: usize,

    /// The pointwise expansion's hidden width.
    pub hidden: usize,

    /// The depthwise kernel's taps, at least 2.
    #[config(default = 7)]
    pub kernel: usize,

    /// The normalization after the depthwise convolution; its feature count
    /// is set to [`width`](Self::width).
    #[config(
        default = "BunsenNormalizationConfig::Layer(BunsenLayerNormConfig::new(0).with_epsilon(1e-6))"
    )]
    pub norm: BunsenNormalizationConfig,

    /// The activation after the expansion: `GELU`, `ConvNeXt`'s.
    #[config(default = "BunsenActivationConfig::Gelu")]
    pub act: BunsenActivationConfig,

    /// Whether the branch is scaled per channel, by `γ`, before the residual
    /// add (`ConvNeXt`'s layer scale). Off by default.
    #[config(default = false)]
    pub layer_scale: bool,
}

impl CausalConvNext1dMeta for CausalConvNext1dConfig {
    fn width(&self) -> usize {
        self.width
    }

    fn hidden(&self) -> usize {
        self.hidden
    }

    fn kernel(&self) -> usize {
        self.kernel
    }

    fn layer_scale(&self) -> bool {
        self.layer_scale
    }
}

impl ModuleInit<CausalConvNext1d> for CausalConvNext1dConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a kernel of
    /// fewer than 2 taps.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<CausalConvNext1d> {
        if self.kernel < 2 {
            return Err(ConstraintError::out_of_range(
                "CausalConvNext1dConfig",
                "kernel",
                self.kernel,
                ">= 2",
            )
            .into());
        }
        let c = self.width;
        Ok(CausalConvNext1d {
            dwconv: Conv1dConfig::new(c, c, self.kernel)
                .with_groups(c)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            norm: self.norm.clone().with_num_features(c).try_init(device)?,
            pwconv1: LinearConfig::new(c, self.hidden).init(device),
            act: self.act.try_init(device)?,
            pwconv2: LinearConfig::new(self.hidden, c).init(device),
            gamma: self
                .layer_scale
                .then(|| Param::from_tensor(Tensor::ones([c], device))),
        })
    }
}

/// A causal 1-d `ConvNeXt` block (`DualCodec`'s and ZONOS3's codec's):
///
/// ```text
/// h = dwconv(x)           (depthwise, each channel over its own past: k - 1 left padding)
/// y = x + γ ⊙ pwconv2(act(pwconv1(norm(h))))
/// ```
///
/// with `γ` a per-channel layer scale when
/// [`layer_scale`](CausalConvNext1dMeta::layer_scale) is set (Qwen3-TTS's
/// codec's), and 1 otherwise (ZONOS3's). Channels last, `[B, T, C]`. The
/// depthwise convolution continues from a carried state
/// ([`forward_with_state`](Self::forward_with_state)), so a stream run in
/// chunks is the same as one run whole.
///
/// Built by [`CausalConvNext1dConfig`].
#[derive(Module, Debug)]
pub struct CausalConvNext1d {
    /// The depthwise convolution: weight `[C, 1, K]`, bias `[C]`.
    pub dwconv: Conv1d,

    /// The normalization over the channels.
    pub norm: BunsenNormalization,

    /// The expansion, `[C, H]`.
    pub pwconv1: Linear,

    /// The activation after the expansion.
    pub act: BunsenActivation,

    /// The projection back, `[H, C]`.
    pub pwconv2: Linear,

    /// The layer scale `γ`, `[C]`, when the block has one.
    pub gamma: Option<Param<Tensor<1>>>,
}

impl CausalConvNext1dMeta for CausalConvNext1d {
    fn width(&self) -> usize {
        self.dwconv.weight.dims()[0]
    }

    fn hidden(&self) -> usize {
        self.pwconv1.weight.dims()[1]
    }

    fn kernel(&self) -> usize {
        self.dwconv.weight.dims()[2]
    }

    fn layer_scale(&self) -> bool {
        self.gamma.is_some()
    }
}

impl CausalConvNext1d {
    /// The block over a whole sequence, from zeros before its start.
    ///
    /// # Arguments
    /// - `x`: `[B, T, C]`.
    ///
    /// # Returns
    /// - `[B, T, C]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        self.forward_with_state(x, None).0
    }

    /// The block over the next `T` positions of a stream.
    ///
    /// # Arguments
    /// - `x`: `[B, T, C]`.
    /// - `state`: `[B, K - 1, C]`, the inputs before `x`; `None`, zeros.
    ///
    /// # Returns
    /// - `[B, T, C]`, and the next call's `[B, K - 1, C]` state.
    pub fn forward_with_state(
        &self,
        x: Tensor<3>,
        state: Option<Tensor<3>>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [c, k] = [self.width(), self.kernel()];
        let [b, t] = unpack_shape_contract!(["B", "T", "C"], &x.dims(), &["B", "T"], &[("C", c)]);
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (b, t);
        let (h, state) =
            causal_depthwise_conv1d(x.clone(), self.dwconv.weight.val().reshape([c, k]), state);
        let h = match &self.dwconv.bias {
            Some(bias) => h + bias.val().reshape([1, 1, c]),
            None => h,
        };
        let h = self
            .pwconv2
            .forward(self.act.forward(self.pwconv1.forward(self.norm.forward(h))));
        let h = match &self.gamma {
            Some(gamma) => h * gamma.val().reshape([1, 1, c]),
            None => h,
        };
        let y = x + h;
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "T", "C"],
            &y.dims(),
            &[("B", b), ("T", t), ("C", c)]
        );
        (y, state)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::PaddingConfig1d,
        tensor::{
            Distribution,
            Tolerance,
            activation::gelu,
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

    /// The block written out with burn's own convolution, left-padded by
    /// `k - 1` and channels first, as upstream runs it.
    #[test]
    fn test_is_the_definition() {
        let device = cpu_device();
        let block: CausalConvNext1d = CausalConvNext1dConfig::new(6, 10).init(&device);
        assert_eq!((block.width(), block.hidden(), block.kernel()), (6, 10, 7));
        assert!(!block.layer_scale());
        let x: Tensor<3> = seeded_tensor(1, [2, 9, 6], Distribution::Default, &device);

        let padded = Conv1dConfig::new(6, 6, 7)
            .with_groups(6)
            .with_padding(PaddingConfig1d::Explicit(6, 0));
        let mut conv: Conv1d = padded.init(&device);
        conv.weight = block.dwconv.weight.clone();
        conv.bias = block.dwconv.bias.clone();
        let h = conv.forward(x.clone().swap_dims(1, 2)).swap_dims(1, 2);
        let expected = x.clone()
            + block
                .pwconv2
                .forward(gelu(block.pwconv1.forward(block.norm.forward(h))));

        assert_tensors_close(&block.forward(x), &expected, Tolerance::default());
    }

    /// Run in chunks through its state, the block is the block run whole.
    #[test]
    fn test_chunks_are_the_whole() {
        let device = cpu_device();
        let block: CausalConvNext1d = CausalConvNext1dConfig::new(4, 8)
            .with_kernel(5)
            .init(&device);
        let x: Tensor<3> = seeded_tensor(2, [1, 11, 4], Distribution::Default, &device);
        let whole = block.forward(x.clone());
        let mut state = None;
        let mut parts = Vec::new();
        for (from, to) in [(0, 2), (2, 7), (7, 8), (8, 11)] {
            let (y, next) = block.forward_with_state(x.clone().slice_dim(1, from..to), state);
            parts.push(y);
            state = Some(next);
        }
        assert_tensors_close(&Tensor::cat(parts, 1), &whole, Tolerance::default());
    }

    /// With a layer scale, the branch is scaled per channel: a new block's
    /// `γ` of ones is the block without one.
    #[test]
    fn test_layer_scale() {
        let device = cpu_device();
        let plain: CausalConvNext1d = CausalConvNext1dConfig::new(4, 8).init(&device);
        let mut scaled: CausalConvNext1d = CausalConvNext1dConfig::new(4, 8)
            .with_layer_scale(true)
            .init(&device);
        assert!(scaled.layer_scale());
        scaled.dwconv = plain.dwconv.clone();
        scaled.norm = plain.norm.clone();
        scaled.pwconv1 = plain.pwconv1.clone();
        scaled.pwconv2 = plain.pwconv2.clone();
        let x: Tensor<3> = seeded_tensor(3, [2, 5, 4], Distribution::Default, &device);
        let y = plain.forward(x.clone());
        assert_tensors_close(&scaled.forward(x.clone()), &y, Tolerance::default());

        let gamma = Tensor::<1>::from_floats([0.5, -1.0, 2.0, 0.0], &device);
        scaled.gamma = Some(Param::from_tensor(gamma.clone()));
        let expected = x.clone() + (y - x.clone()) * gamma.reshape([1, 1, 4]);
        assert_tensors_close(&scaled.forward(x), &expected, Tolerance::default());
    }

    /// The Qwen3-TTS toy codec's two upsample stages
    /// (`tools/gen_qwen3_tts_toy_fixtures.py`): each a causal transposed
    /// convolution, then a `ConvNeXt` block with a layer scale, channels
    /// first upstream and here swapped to channels last.
    #[cfg(feature = "store")]
    #[test]
    fn test_qwen3_tts_toy_upsample() {
        use crate::{
            blocks::{
                conv::CausalConvTranspose1dConfig,
                norm::BunsenNormalization,
            },
            support::testing::{
                ToyFixture,
                assert_io_close,
            },
        };

        let device = cpu_device();
        let path = crate::support::testing::testdata!("qwen3_tts/toy.safetensors");
        let fixture = ToyFixture::load_io(path, &device);
        let param1 = |name: &str| Param::from_tensor(fixture.io::<1>(name));
        // torch's `Linear` is `[out, in]`; burn's is `[in, out]`
        let linear_t = |name: &str| Param::from_tensor(fixture.io::<2>(name).transpose());

        let mut x = fixture.io::<3>("io.codec.transformer").swap_dims(1, 2);
        for stage in 0..2 {
            let at = |part: &str| format!("decoder.upsample.{stage}.{part}");
            let weight = fixture.io::<3>(&at("0.conv.weight"));
            let [c, _, k] = weight.dims();
            let mut up = CausalConvTranspose1dConfig::new([c, c], k)
                .with_stride(k)
                .init(&device);
            up.conv.weight = Param::from_tensor(weight);
            up.conv.bias = Some(param1(&at("0.conv.bias")));

            let mut block: CausalConvNext1d = CausalConvNext1dConfig::new(c, 4 * c)
                .with_layer_scale(true)
                .init(&device);
            block.dwconv.weight = Param::from_tensor(fixture.io::<3>(&at("1.dwconv.conv.weight")));
            block.dwconv.bias = Some(param1(&at("1.dwconv.conv.bias")));
            crate::support::testing::expect_variant!(let BunsenNormalization::Layer(norm) = &mut block.norm, "a layer norm");
            norm.gamma = param1(&at("1.norm.weight"));
            norm.beta = Some(param1(&at("1.norm.bias")));
            block.pwconv1.weight = linear_t(&at("1.pwconv1.weight"));
            block.pwconv1.bias = Some(param1(&at("1.pwconv1.bias")));
            block.pwconv2.weight = linear_t(&at("1.pwconv2.weight"));
            block.pwconv2.bias = Some(param1(&at("1.pwconv2.bias")));
            block.gamma = Some(param1(&at("1.gamma")));

            x = block.forward(up.forward(x).swap_dims(1, 2)).swap_dims(1, 2);
            assert_io_close(
                &at(""),
                x.clone(),
                fixture.io(&format!("io.codec.upsample_{stage}")),
            );
        }
    }

    #[test]
    fn test_refuses_one_tap() {
        ErrorMatcher::illegal().assert_err(
            &CausalConvNext1dConfig::new(4, 8)
                .with_kernel(1)
                .try_init(&cpu_device()),
        );
    }
}
