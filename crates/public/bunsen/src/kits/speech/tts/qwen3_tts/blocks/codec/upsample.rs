//! [`Qwen3TtsCodecUpsample`]: one of the codec decoder's upsampling stages, and its config.

use burn::{
    config::Config,
    prelude::*,
};

use crate::{
    blocks::conv::{
        CausalConvNext1d,
        CausalConvNext1dConfig,
        CausalConvNext1dMeta,
        CausalConvTranspose1d,
        CausalConvTranspose1dConfig,
        CausalConvTranspose1dMeta,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`Qwen3TtsCodecUpsample`] and
/// [`Qwen3TtsCodecUpsampleConfig`].
pub trait Qwen3TtsCodecUpsampleMeta {
    /// The width, `L`, in and out.
    fn width(&self) -> usize;

    /// The factor, `r`: output frames per input frame.
    fn factor(&self) -> usize;

    /// The `ConvNeXt` block's hidden width.
    fn hidden(&self) -> usize;

    /// The `ConvNeXt` block's depthwise taps.
    fn block_kernel(&self) -> usize;
}

/// Config for [`Qwen3TtsCodecUpsample`]: its factor, and its `ConvNeXt`
/// block's config.
#[derive(Config, Debug)]
pub struct Qwen3TtsCodecUpsampleConfig {
    /// The width, `L`.
    pub width: usize,

    /// The factor, `r`.
    pub factor: usize,

    /// The `ConvNeXt` block, over `L`.
    pub block: CausalConvNext1dConfig,
}

impl Qwen3TtsCodecUpsampleMeta for Qwen3TtsCodecUpsampleConfig {
    fn width(&self) -> usize {
        self.width
    }

    fn factor(&self) -> usize {
        self.factor
    }

    fn hidden(&self) -> usize {
        self.block.hidden
    }

    fn block_kernel(&self) -> usize {
        self.block.kernel
    }
}

impl ModuleInit<Qwen3TtsCodecUpsample> for Qwen3TtsCodecUpsampleConfig {
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a block not
    /// over `L`; as the convolutions' builds: a factor of 0 is refused.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Qwen3TtsCodecUpsample> {
        let [w, r] = [self.width, self.factor];
        if self.block.width != w {
            return Err(crate::errors::ConstraintError::new(
                "Qwen3TtsCodecUpsampleConfig",
                "block.width",
                crate::errors::Rule::Relation {
                    lhs: ("block.width".into(), self.block.width.to_string()),
                    op: "==",
                    rhs: ("width".into(), w.to_string()),
                },
            )
            .into());
        }
        Ok(Qwen3TtsCodecUpsample {
            up: CausalConvTranspose1dConfig::new([w, w], r)
                .with_stride(r)
                .try_init(device)?,
            block: self.block.try_init(device)?,
        })
    }
}

/// One upsampler: a causal transposed convolution (kernel and stride its
/// factor), then a causal `ConvNeXt` block (the release's with a layer
/// scale).
///
/// Built by [`Qwen3TtsCodecUpsampleConfig`].
#[derive(Module, Debug)]
pub struct Qwen3TtsCodecUpsample {
    /// The transposed convolution.
    pub up: CausalConvTranspose1d,

    /// The `ConvNeXt` block, channels last.
    pub block: CausalConvNext1d,
}

impl Qwen3TtsCodecUpsampleMeta for Qwen3TtsCodecUpsample {
    fn width(&self) -> usize {
        self.up.channels()[0]
    }

    fn factor(&self) -> usize {
        self.up.stride()
    }

    fn hidden(&self) -> usize {
        self.block.hidden()
    }

    fn block_kernel(&self) -> usize {
        self.block.kernel()
    }
}

impl Qwen3TtsCodecUpsample {
    /// `[B, L, T]` to `[B, L, r T]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        self.block
            .forward(self.up.forward(x).swap_dims(1, 2))
            .swap_dims(1, 2)
    }
}
