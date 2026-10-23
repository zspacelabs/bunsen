//! `SubsampledEncoder`: a frame-rate-changing front end, then an encoder stack.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    blocks::{
        frame_rate::{
            FeatureStacking,
            FeatureStackingConfig,
            FeatureStackingMeta,
        },
        layout::AxisOrder,
        transformers::encoders::{
            EncoderTransformer,
            EncoderTransformerConfig,
            EncoderTransformerMeta,
            conformer::{
                ConformerEncoderConfig,
                DwStridingSubsampling,
                DwStridingSubsamplingConfig,
                DwStridingSubsamplingMeta,
            },
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
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
/// [`SubsampledEncoder`]'s [`AxisOrder`] bridges a caller's layout to it.
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

/// Common meta for [`SubsampledEncoder`] and [`SubsampledEncoderConfig`].
pub trait SubsampledEncoderMeta {
    /// The features per input frame: the mel bins.
    fn n_mels(&self) -> usize;

    /// The width of the embeddings and of the encoder.
    fn d_model(&self) -> usize;

    /// The input frames per encoder frame.
    fn subsampling_factor(&self) -> usize;

    /// The encoder's layers.
    fn n_layers(&self) -> usize;
}

/// Config for [`SubsampledEncoder`]: the layout it is fed in, its front end,
/// and its stack.
///
/// Implements [`SubsampledEncoderMeta`].
#[derive(Config, Debug)]
pub struct SubsampledEncoderConfig {
    /// The front end.
    pub subsampling: SubsamplingConfig,

    /// The stack after it.
    pub transformer: EncoderTransformerConfig,

    /// The permutes from the caller's layout to the front end's, and of the
    /// stack's output back; none by default.
    #[config(default = "AxisOrder::IDENTITY")]
    pub layout: AxisOrder,
}

impl SubsampledEncoderMeta for SubsampledEncoderConfig {
    fn n_mels(&self) -> usize {
        self.subsampling.n_input()
    }

    fn d_model(&self) -> usize {
        self.transformer.d_model()
    }

    fn subsampling_factor(&self) -> usize {
        self.subsampling.factor()
    }

    fn n_layers(&self) -> usize {
        self.transformer.n_layers()
    }
}

impl From<ConformerEncoderConfig> for SubsampledEncoderConfig {
    /// The `FastConformer` as a subsampled encoder: its depthwise striding,
    /// then its [`ConformerTransformer`](super::conformer::ConformerTransformer),
    /// fed the striding's own `[batch, n_mels, time]`.
    fn from(config: ConformerEncoderConfig) -> Self {
        Self::new(
            SubsamplingConfig::DwStriding(config.subsampling.clone()),
            config.transformer().into(),
        )
    }
}

impl ModuleInit<SubsampledEncoder> for SubsampledEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SubsampledEncoder> {
        if self.subsampling.d_model() != self.transformer.d_model() {
            return Err(ConstraintError::new(
                "SubsampledEncoderConfig",
                "",
                Rule::Relation {
                    lhs: (
                        "subsampling.d_model()".into(),
                        self.subsampling.d_model().to_string(),
                    ),
                    op: "==",
                    rhs: (
                        "transformer.d_model()".into(),
                        self.transformer.d_model().to_string(),
                    ),
                },
            )
            .into());
        }
        Ok(SubsampledEncoder {
            subsampling: self.subsampling.try_init(device)?,
            transformer: self.transformer.try_init(device)?,
            layout: self.layout,
        })
    }
}

/// A front end that changes the frame rate ([`Subsampling`]), then an
/// encoder stack that keeps it ([`EncoderTransformer`]): the shape of a
/// speech encoder, whichever front end and stack its config chose.
///
/// Split as a streaming model runs it: [`pre_encode`](Self::pre_encode), the
/// front end alone, and [`forward_pre_encoded`](Self::forward_pre_encoded),
/// the stack over embeddings a caller may have edited in between (a speaker
/// cache splicing in earlier frames).
///
/// Its fields are `subsampling` and `transformer`, so its tensors' paths are
/// the front end's under `subsampling.` and the stack's under
/// `transformer.` (the enums add no segment). Its [`AxisOrder`] is config,
/// not a tensor.
///
/// Built by [`SubsampledEncoderConfig`].
/// Implements [`SubsampledEncoderMeta`].
#[derive(Module, Debug)]
pub struct SubsampledEncoder {
    /// The front end.
    pub subsampling: Subsampling,

    /// The stack after it.
    pub transformer: EncoderTransformer,

    /// The permutes at its boundary.
    #[module(skip)]
    pub layout: AxisOrder,
}

impl SubsampledEncoderMeta for SubsampledEncoder {
    fn n_mels(&self) -> usize {
        self.subsampling.n_input()
    }

    fn d_model(&self) -> usize {
        self.transformer.d_model()
    }

    fn subsampling_factor(&self) -> usize {
        self.subsampling.factor()
    }

    fn n_layers(&self) -> usize {
        self.transformer.n_layers()
    }
}

impl SubsampledEncoder {
    /// The encoder's output length for `len` input frames.
    pub fn output_len(
        &self,
        len: usize,
    ) -> usize {
        self.subsampling.output_len(len)
    }

    /// Encodes features: [`pre_encode`](Self::pre_encode), then
    /// [`forward_pre_encoded`](Self::forward_pre_encoded).
    ///
    /// # Arguments
    /// * `features` - in the caller's layout, which the config's [`AxisOrder`]
    ///   takes to the front end's.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills its time axis.
    ///
    /// # Returns
    /// `[batch, output_len(time), d_model]` (permuted by the layout's `post`,
    /// if any), and each item's output length.
    pub fn forward(
        &self,
        features: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        let padded = lengths.is_some();
        let (x, lengths) = self.pre_encode(features, lengths);
        let x = self.forward_pre_encoded(x, padded.then_some(lengths.as_slice()));
        (x, lengths)
    }

    /// The front end alone: the layout's `pre` permute, then the subsampling.
    ///
    /// # Returns
    /// `[batch, output_len(time), d_model]`, and each item's output length.
    pub fn pre_encode(
        &self,
        features: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        self.subsampling
            .forward(self.layout.pre_apply(features), lengths)
    }

    /// The stack over pre-encode embeddings, then the layout's `post` permute.
    ///
    /// # Arguments
    /// * `embs` - `[batch, frames, d_model]`, as
    ///   [`pre_encode`](Self::pre_encode) gives them.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `frames`.
    pub fn forward_pre_encoded(
        &self,
        embs: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        self.layout
            .post_apply(self.transformer.forward(embs, lengths))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::{
        blocks::transformers::encoders::{
            conformer::{
                ConformerBlockConfig,
                ConformerTransformerConfig,
            },
            rope::{
                RopeEncoderBlockConfig,
                RopeTransformerEncoderConfig,
            },
        },
        support::testing::cpu_device,
    };

    fn conformer(layout: AxisOrder) -> SubsampledEncoderConfig {
        SubsampledEncoderConfig::new(
            SubsamplingConfig::DwStriding(
                DwStridingSubsamplingConfig::new(16, 4, 8).with_n_stages(2),
            ),
            ConformerTransformerConfig::new(ConformerBlockConfig::new(8, 2, 16, 3), 2).into(),
        )
        .with_layout(layout)
    }

    /// The front end and the stack, in order; and a caller feeding the
    /// other layout gets the same answer through the layout's permute.
    #[test]
    fn test_layout_bridges_the_callers_order() {
        let device = cpu_device();
        device.seed(5);
        let native: SubsampledEncoder = conformer(AxisOrder::IDENTITY).init(&device);
        assert_eq!(native.n_mels(), 16);
        assert_eq!(native.subsampling_factor(), 4);
        assert_eq!(native.n_layers(), 2);

        // `[batch, n_mels, time]`, the depthwise striding's own layout.
        let mels = Tensor::<3>::random([2, 16, 20], Distribution::Default, &device);
        let (want, want_lengths) = native.forward(mels.clone(), Some(&[20, 13]));
        assert_eq!(want.dims(), [2, native.output_len(20), 8]);

        let mut swapped = native.clone();
        swapped.layout = AxisOrder::SWAP_IN;
        let (got, got_lengths) = swapped.forward(mels.swap_dims(1, 2), Some(&[20, 13]));
        assert_eq!(got_lengths, want_lengths);
        got.to_data()
            .assert_approx_eq::<f32>(&want.to_data(), Tolerance::absolute(1e-6));
    }

    /// Its paths, variants skipped as bunsen's loaders skip them, are the
    /// front end's under `subsampling.` and the stack's under
    /// `transformer.`; the layout is not among them.
    #[test]
    #[cfg(feature = "store")]
    fn test_paths() {
        use burn::store::ModuleSnapshot;

        let device = cpu_device();
        let stacking = SubsampledEncoderConfig::new(
            SubsamplingConfig::Stacking(FeatureStackingConfig::new(6, 8).with_factor(2)),
            RopeTransformerEncoderConfig::new(RopeEncoderBlockConfig::new(8, 2), 1).into(),
        );
        for config in [conformer(AxisOrder::SWAP_IN), stacking] {
            let encoder: SubsampledEncoder = config.init(&device);
            let paths: Vec<String> = encoder
                .collect(None, None, true)
                .into_iter()
                .map(|t| t.name)
                .collect();
            assert!(!paths.is_empty());
            assert!(
                paths
                    .iter()
                    .all(|p| p.starts_with("subsampling.") || p.starts_with("transformer.")),
                "{paths:?}"
            );
            assert!(paths.iter().any(|p| p.starts_with("transformer.layers.0.")));
        }
    }

    /// A front end and a stack of different widths are refused.
    #[test]
    fn test_widths_must_agree() {
        let config = SubsampledEncoderConfig::new(
            SubsamplingConfig::Stacking(FeatureStackingConfig::new(6, 16).with_factor(2)),
            RopeTransformerEncoderConfig::new(RopeEncoderBlockConfig::new(8, 2), 1).into(),
        );
        assert!(config.try_init(&cpu_device()).is_err());
    }
}
