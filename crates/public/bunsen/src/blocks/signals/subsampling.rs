//! `Subsampling`: an encoder front end, chosen by config.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    blocks::signals::{
        DwStridingSubsampling,
        DwStridingSubsamplingConfig,
        DwStridingSubsamplingMeta,
        FeatureStacking,
        FeatureStackingConfig,
        FeatureStackingMeta,
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// Common meta for [`Subsampling`] and [`SubsamplingConfig`].
pub trait SubsamplingMeta {
    /// The features per input frame: the mel bins.
    fn n_input(&self) -> usize;

    /// The width it writes.
    fn d_model(&self) -> usize;

    /// The input frames per output frame.
    fn factor(&self) -> usize;

    /// The output length for `len` input frames.
    fn output_len(
        &self,
        len: usize,
    ) -> usize;
}

/// Config for [`Subsampling`]: which front end, and its config.
///
/// Implements [`SubsamplingMeta`].
#[derive(Config, Debug)]
pub enum SubsamplingConfig {
    /// [`FeatureStacking`]: frames stacked and projected.
    Stacking(FeatureStackingConfig),

    /// [`DwStridingSubsampling`]: the `FastConformer`'s strided convolutions.
    DwStriding(DwStridingSubsamplingConfig),
}

impl SubsamplingMeta for SubsamplingConfig {
    fn n_input(&self) -> usize {
        match self {
            Self::Stacking(c) => c.d_input(),
            Self::DwStriding(c) => c.n_mels(),
        }
    }

    fn d_model(&self) -> usize {
        match self {
            Self::Stacking(c) => c.d_output(),
            Self::DwStriding(c) => DwStridingSubsamplingMeta::d_model(c),
        }
    }

    fn factor(&self) -> usize {
        match self {
            Self::Stacking(c) => FeatureStackingMeta::factor(c),
            Self::DwStriding(c) => DwStridingSubsamplingMeta::factor(c),
        }
    }

    fn output_len(
        &self,
        len: usize,
    ) -> usize {
        match self {
            Self::Stacking(c) => FeatureStackingMeta::output_len(c, len),
            Self::DwStriding(c) => DwStridingSubsamplingMeta::output_len(c, len),
        }
    }
}

impl ModuleInit<Subsampling> for SubsamplingConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Subsampling> {
        Ok(match self {
            Self::Stacking(c) => Subsampling::Stacking(c.try_init(device)?),
            Self::DwStriding(c) => Subsampling::DwStriding(c.try_init(device)?),
        })
    }
}

/// An encoder's front end, whichever its config chose: input frames to
/// fewer, wider frames, each item's length with them.
///
/// Every variant shares one contract, `forward(x, lengths) -> (y, lengths)`
/// with `y` `[batch, output_len(time), d_model]`, but each reads the input
/// layout its upstream did: [`FeatureStacking`] `[batch, time, n_input]`,
/// [`DwStridingSubsampling`] `[batch, n_input, time]`. A
/// [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder)'s
/// [`AxisOrder`](crate::blocks::layout::AxisOrder) bridges a caller's layout
/// to it.
///
/// Built by [`SubsamplingConfig`].
/// Implements [`SubsamplingMeta`].
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Subsampling {
    /// [`FeatureStacking`].
    Stacking(FeatureStacking),

    /// [`DwStridingSubsampling`].
    DwStriding(DwStridingSubsampling),
}

impl SubsamplingMeta for Subsampling {
    fn n_input(&self) -> usize {
        match self {
            Self::Stacking(m) => m.d_input(),
            Self::DwStriding(m) => m.n_mels(),
        }
    }

    fn d_model(&self) -> usize {
        match self {
            Self::Stacking(m) => m.d_output(),
            Self::DwStriding(m) => DwStridingSubsamplingMeta::d_model(m),
        }
    }

    fn factor(&self) -> usize {
        match self {
            Self::Stacking(m) => FeatureStackingMeta::factor(m),
            Self::DwStriding(m) => DwStridingSubsamplingMeta::factor(m),
        }
    }

    fn output_len(
        &self,
        len: usize,
    ) -> usize {
        match self {
            Self::Stacking(m) => FeatureStackingMeta::output_len(m, len),
            Self::DwStriding(m) => DwStridingSubsamplingMeta::output_len(m, len),
        }
    }
}

impl Subsampling {
    /// Subsamples `x`, in this variant's own input layout.
    ///
    /// # Arguments
    /// * `x` - `[batch, time, n_input]` (stacking) or `[batch, n_input, time]`
    ///   (depthwise striding).
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    ///
    /// # Returns
    /// `[batch, output_len(time), d_model]`, and each item's output length.
    pub fn forward(
        &self,
        x: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        match self {
            Self::Stacking(m) => m.forward(x, lengths),
            Self::DwStriding(m) => m.forward(x, lengths),
        }
    }
}
