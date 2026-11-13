//! `CausalConvNext1d`: a causal 1-d `ConvNeXt` block.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
        PaddingConfig1d,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    tensor::{
        Device,
        activation::gelu,
    },
};

use crate::{
    blocks::norm::{
        BunsenNormalization,
        BunsenNormalizationConfig,
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenError,
        BunsenResult,
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
        default = "BunsenNormalizationConfig::Layer(burn::nn::LayerNormConfig::new(0).with_epsilon(1e-6))"
    )]
    pub norm: BunsenNormalizationConfig,
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
            return Err(BunsenError::illegal(format!(
                "a causal ConvNeXt kernel of {} taps; at least 2",
                self.kernel
            )));
        }
        let c = self.width;
        Ok(CausalConvNext1d {
            dwconv: Conv1dConfig::new(c, c, self.kernel)
                .with_groups(c)
                .with_padding(PaddingConfig1d::Valid)
                .init(device),
            norm: self.norm.clone().with_num_features(c).init(device),
            pwconv1: LinearConfig::new(c, self.hidden).init(device),
            pwconv2: LinearConfig::new(self.hidden, c).init(device),
        })
    }
}

/// A causal 1-d `ConvNeXt` block (`DualCodec`'s and ZONOS3's codec's):
///
/// ```text
/// h = dwconv(x)           (depthwise, each channel over its own past: k - 1 left padding)
/// y = x + pwconv2(gelu(pwconv1(norm(h))))
/// ```
///
/// with no per-channel scale on the branch. Channels last, `[B, T, C]`. The
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

    /// The projection back, `[H, C]`.
    pub pwconv2: Linear,
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
        let (h, state) =
            causal_depthwise_conv1d(x.clone(), self.dwconv.weight.val().reshape([c, k]), state);
        let h = match &self.dwconv.bias {
            Some(bias) => h + bias.val().reshape([1, 1, c]),
            None => h,
        };
        let h = self
            .pwconv2
            .forward(gelu(self.pwconv1.forward(self.norm.forward(h))));
        let y = x + h;
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
        },
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    /// The block written out with burn's own convolution, left-padded by
    /// `k - 1` and channels first, as upstream runs it.
    #[test]
    fn test_is_the_definition() {
        let device = cpu_device();
        let block: CausalConvNext1d = CausalConvNext1dConfig::new(6, 10).init(&device);
        assert_eq!((block.width(), block.hidden(), block.kernel()), (6, 10, 7));
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

    #[test]
    fn test_refuses_one_tap() {
        assert!(
            CausalConvNext1dConfig::new(4, 8)
                .with_kernel(1)
                .try_init(&cpu_device())
                .is_err()
        );
    }
}
