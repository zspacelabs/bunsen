//! # `ConvSeq2d` - sequence of [`ConvBlock2d`].

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        activation::ActivationConfig,
        norm::NormalizationConfig,
    },
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
        BunsenError,
        BunsenResult,
        ConstraintError,
        ResultContext,
        Rule,
    },
};

/// [`ConvSeq2d`] Meta.
///
/// Sequence-level metadata, derived from the chain of per-block
/// [`ConvBlock2dMeta`]. Implemented by:
/// * [`ConvSeq2dConfig`]
/// * [`ConvSeq2d`]
pub trait ConvSeq2dMeta {
    /// The per-block [`ConvBlock2dMeta`], in sequence order.
    fn block_metas(&self) -> Vec<&dyn ConvBlock2dMeta>;

    /// The number of blocks in the sequence.
    fn len(&self) -> usize {
        self.block_metas().len()
    }

    /// Whether the sequence has no blocks.
    fn is_empty(&self) -> bool {
        self.block_metas().is_empty()
    }

    /// The number of input channels of the sequence.
    ///
    /// This is the [`in_channels`](ConvBlock2dMeta::in_channels) of the first
    /// block.
    ///
    /// # Panics
    ///
    /// If the sequence is empty.
    fn in_channels(&self) -> usize {
        self.block_metas()
            .first()
            .expect("ConvSeq2d must have at least one block")
            .in_channels()
    }

    /// The number of output channels of the sequence.
    ///
    /// This is the [`out_channels`](ConvBlock2dMeta::out_channels) of the last
    /// block.
    ///
    /// # Panics
    ///
    /// If the sequence is empty.
    fn out_channels(&self) -> usize {
        self.block_metas()
            .last()
            .expect("ConvSeq2d must have at least one block")
            .out_channels()
    }

    /// The total stride of the sequence; `[height, width]`.
    ///
    /// This is the per-dimension product of each block's
    /// [`stride`](ConvBlock2dMeta::stride).
    fn stride(&self) -> [usize; 2] {
        self.block_metas().iter().fold([1, 1], |acc, meta| {
            let stride = meta.stride();
            [acc[0] * stride[0], acc[1] * stride[1]]
        })
    }

    /// Validates the sequence.
    ///
    /// A legal sequence:
    /// * is non-empty,
    /// * has channel-compatible adjacent blocks; i.e. each block's
    ///   [`out_channels`](ConvBlock2dMeta::out_channels) equals the following
    ///   block's [`in_channels`](ConvBlock2dMeta::in_channels).
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, if the sequence is empty or has a channel
    /// mismatch between adjacent blocks.
    fn validate(&self) -> BunsenResult<()> {
        let metas = self.block_metas();
        if metas.is_empty() {
            return Err(ConstraintError::zero_or_empty("ConvSeq2d", "blocks").into());
        }

        for (idx, pair) in metas.windows(2).enumerate() {
            let prev = pair[0];
            let next = pair[1];
            if prev.out_channels() != next.in_channels() {
                return Err(ConstraintError::new(
                    "ConvSeq2d",
                    "blocks",
                    Rule::Chain {
                        index: idx + 1,
                        out: format!("{} channels", prev.out_channels()),
                        r#in: format!("{} channels", next.in_channels()),
                    },
                )
                .into());
            }
        }

        Ok(())
    }

    /// Predicts the output resolution for a given input resolution.
    ///
    /// Folds the input resolution through each block's
    /// [`try_output_resolution`](ConvBlock2dMeta::try_output_resolution), which
    /// models the true 2D convolution arithmetic (kernel size, padding,
    /// dilation, and stride).
    ///
    /// # Arguments
    ///
    /// * `input_resolution` - The input resolution `[in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// The predicted output resolution `[out_height, out_width]`.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if some block has
    /// no legal output resolution for its input resolution (the kernel does
    /// not fit the padded input).
    fn try_output_resolution(
        &self,
        input_resolution: [usize; 2],
    ) -> BunsenResult<[usize; 2]> {
        let mut resolution = input_resolution;
        for (idx, meta) in self.block_metas().iter().enumerate() {
            resolution = meta
                .try_output_resolution(resolution)
                .with_context(|| format!("ConvSeq2d block {idx}"))?;
        }
        Ok(resolution)
    }

    /// Predicts the output shape for a given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The input shape `[batch_size, in_channels, in_height,
    ///   in_width]`.
    ///
    /// # Returns
    ///
    /// The predicted output shape `[batch_size, out_channels, out_height,
    /// out_width]`.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the input
    /// channels do not match the sequence's [`in_channels`](Self::in_channels),
    /// or if the input resolution has no legal output through the sequence
    /// (see [`Self::try_output_resolution`]).
    fn try_output_shape(
        &self,
        input_shape: [usize; 4],
    ) -> BunsenResult<[usize; 4]> {
        let [batch_size, in_channels, in_height, in_width] = input_shape;
        if in_channels != self.in_channels() {
            return Err(BunsenError::illegal(format!(
                "ConvSeq2d expected in_channels ({}), got ({in_channels})",
                self.in_channels(),
            )));
        }
        let [out_height, out_width] = self.try_output_resolution([in_height, in_width])?;
        Ok([batch_size, self.out_channels(), out_height, out_width])
    }
}

/// [`ConvSeq2d`] Config.
///
/// Implements [`ConvSeq2dMeta`].
///
/// Built into a [`ConvSeq2d`] via [`ModuleInit::init`] /
/// [`ModuleInit::try_init`].
///
/// # Example
///
/// Building a small down-sampling stem: a "same"-padded 3x3 conv followed by
/// a stride-2 3x3 conv that halves the spatial resolution. Each block's
/// `out_channels` feeds the next block's `in_channels`.
///
/// ```rust,no_run
/// use bunsen::{
///     blocks::conv::{
///         ConvBlock2dConfig,
///         ConvSeq2d,
///         ConvSeq2dConfig,
///         ConvSeq2dMeta,
///     },
///     burner::module::ModuleInit,
///     support::testing::cpu_device,
/// };
/// use burn::nn::{
///     PaddingConfig2d,
///     activation::ActivationConfig,
///     conv::Conv2dConfig,
/// };
///
/// let device = cpu_device();
///
/// let config = ConvSeq2dConfig::new(vec![
///     ConvBlock2dConfig::new(
///         Conv2dConfig::new([3, 64], [3, 3])
///             .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
///     ),
///     ConvBlock2dConfig::new(
///         Conv2dConfig::new([64, 128], [3, 3])
///             .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
///             .with_stride([2, 2]),
///     ),
/// ])
/// // Apply a shared activation to every block:
/// .with_act(ActivationConfig::Relu);
///
/// // Predict the output shape before building:
/// // [batch, 3, 64, 64] -> [batch, 128, 32, 32].
/// assert_eq!(
///     config.try_output_shape([1, 3, 64, 64]).unwrap(),
///     [1, 128, 32, 32]
/// );
///
/// // `try_init` builds and validates the sequence.
/// let seq: ConvSeq2d = config.try_init(&device).unwrap();
/// ```
#[derive(Config, Debug)]
pub struct ConvSeq2dConfig {
    /// The [`ConvBlock2dConfig`] modules, in sequence order.
    pub blocks: Vec<ConvBlock2dConfig>,
}

impl ConvSeq2dMeta for ConvSeq2dConfig {
    fn block_metas(&self) -> Vec<&dyn ConvBlock2dMeta> {
        self.blocks
            .iter()
            .map(|config| config as &dyn ConvBlock2dMeta)
            .collect()
    }
}

impl ConvSeq2dConfig {
    /// Set the [`Option<ActivationConfig>`] for all blocks.
    pub fn with_act<A: Into<Option<ActivationConfig>>>(
        self,
        act: A,
    ) -> Self {
        let act = act.into();
        Self {
            blocks: self
                .blocks
                .into_iter()
                .map(|block| block.with_act(act.clone()))
                .collect(),
        }
    }

    /// Set the [`Option<NormalizationConfig>`] for all blocks.
    pub fn with_norm<N: Into<Option<NormalizationConfig>>>(
        self,
        norm: N,
    ) -> Self {
        let norm = norm.into();
        Self {
            blocks: self
                .blocks
                .into_iter()
                .map(|block| block.with_norm(norm.clone()))
                .collect(),
        }
    }
}

impl ModuleInit<ConvSeq2d> for ConvSeq2dConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvSeq2d> {
        let blocks = self
            .blocks
            .iter()
            .map(|config| config.try_init(device))
            .collect::<BunsenResult<Vec<_>>>()?;
        ConvSeq2d::try_new(blocks)
    }
}

/// Sequence of [`ConvBlock2d`].
///
/// A non-empty chain of [`ConvBlock2d`] modules, where each block's
/// output channels feed the next block's input channels.
///
/// Implements [`ConvSeq2dMeta`].
///
/// Built (and validated) via [`Self::try_new`], or from a [`ConvSeq2dConfig`].
///
/// # Example
///
/// Building a two-block, stride-2 down-sampling sequence via
/// [`try_init`](ModuleInit::try_init), then running a forward pass:
///
/// ```rust,no_run
/// use bunsen::support::testing::cpu_device;
/// use bunsen::{
///     blocks::conv::{
///         ConvBlock2dConfig,
///         ConvSeq2d,
///         ConvSeq2dConfig,
///         ConvSeq2dMeta,
///     },
///     burner::module::ModuleInit,
/// };
/// use burn::{
///     Tensor,
///     nn::{
///         PaddingConfig2d,
///         conv::Conv2dConfig,
///     },
///     tensor::Distribution,
/// };
///
/// let device = cpu_device();
///
/// // Two stride-2 down-sampling blocks (3x3 kernel, "same" padding), each
/// // halving the resolution. `try_init` builds and validates the sequence.
/// let seq: ConvSeq2d = ConvSeq2dConfig::new(vec![
///     ConvBlock2dConfig::new(
///         Conv2dConfig::new([3, 64], [3, 3])
///             .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
///             .with_stride([2, 2]),
///     ),
///     ConvBlock2dConfig::new(
///         Conv2dConfig::new([64, 128], [3, 3])
///             .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
///             .with_stride([2, 2]),
///     ),
/// ])
/// .try_init(&device)
/// .unwrap();
///
/// // [batch, 3, 64, 64] -> [batch, 128, 16, 16]; predicted by `try_output_shape`.
/// assert_eq!(seq.try_output_shape([2, 3, 64, 64]).unwrap(), [2, 128, 16, 16]);
///
/// let input = Tensor::<4>::random([2, 3, 64, 64], Distribution::Default, &device);
/// let output = seq.forward(input);
/// assert_eq!(output.dims(), [2, 128, 16, 16]);
/// ```
#[derive(Module, Debug)]
pub struct ConvSeq2d {
    /// The internal [`ConvBlock2d`] modules.
    pub blocks: Vec<ConvBlock2d>,
}

impl ConvSeq2dMeta for ConvSeq2d {
    fn block_metas(&self) -> Vec<&dyn ConvBlock2dMeta> {
        self.blocks
            .iter()
            .map(|block| block as &dyn ConvBlock2dMeta)
            .collect()
    }
}

impl ConvSeq2d {
    /// Creates a new [`ConvSeq2d`] module.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if the blocks do
    /// not form a legal sequence; see
    /// [`ConvSeq2dMeta::validate`].
    pub fn try_new(blocks: Vec<ConvBlock2d>) -> BunsenResult<Self> {
        let seq = Self { blocks };
        seq.validate()?;
        Ok(seq)
    }

    /// Performs a forward pass through the sequence of [`ConvBlock2d`] modules.
    ///
    /// # Arguments
    /// * `input` - The input tensor of shape `[batch_size, in_channels,
    ///   in_height, in_width]`.
    ///
    /// # Returns
    ///
    /// The output tensor of shape `[batch_size, out_channels, out_height,
    /// out_width]`; the shape is predicted by
    /// [`ConvSeq2dMeta::try_output_shape`].
    pub fn forward(
        &self,
        input: Tensor<4>,
    ) -> Tensor<4> {
        let mut output = input;
        for block in &self.blocks {
            output = block.forward(output);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::{
            PaddingConfig2d,
            activation::ActivationConfig,
            conv::Conv2dConfig,
        },
        tensor::Distribution,
    };

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::{
                ErrorMatcher,
                predicate,
            },
        },
        support::testing::cpu_device,
    };

    /// Builds a "same"-padded `ConvBlock2dConfig` (`out = in / stride` per
    /// dim).
    fn block_config(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> ConvBlock2dConfig {
        ConvBlock2dConfig::new(
            Conv2dConfig::new([in_channels, out_channels], [3, 3])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false),
        )
        .with_norm(None)
        .with_act(Some(ActivationConfig::Relu))
    }

    /// Builds a "same"-padded `ConvBlock2d` (`out = in / stride` per dim).
    fn block(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> ConvBlock2d {
        block_config(in_channels, out_channels, stride).init(&cpu_device())
    }

    #[test]
    fn test_validate_empty() {
        let empty = ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .cause(predicate("empty blocks", |e: &ConstraintError| {
                e.rule == Rule::ZeroOrEmpty
            }));
        empty.assert_err(&ConvSeq2d::try_new(vec![]));

        // The config-level meta validates the same way.
        empty.assert_err(&ConvSeq2dConfig::new(vec![]).validate());
    }

    #[test]
    fn test_validate_channel_mismatch() {
        // block 0 out_channels = 4, block 1 in_channels = 8 -> mismatch.
        let blocks = vec![block(2, 4, 1), block(8, 16, 1)];
        let chain = ErrorMatcher::kind(BunsenErrorKind::Illegal).cause(predicate(
            "a chain break at block 1",
            |e: &ConstraintError| matches!(e.rule, Rule::Chain { index: 1, .. }),
        ));
        chain.assert_err(&ConvSeq2d::try_new(blocks));

        // The config rejects it at init.
        let result: BunsenResult<ConvSeq2d> =
            ConvSeq2dConfig::new(vec![block_config(2, 4, 1), block_config(8, 16, 1)])
                .try_init(&cpu_device());
        chain.assert_err(&result);
    }

    #[test]
    fn test_meta() {
        let seq = ConvSeq2d::try_new(vec![block(2, 4, 2), block(4, 8, 2), block(8, 8, 1)]).unwrap();
        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());
        assert_eq!(seq.in_channels(), 2);
        assert_eq!(seq.out_channels(), 8);
        assert_eq!(seq.stride(), [4, 4]);
    }

    #[test]
    fn test_config_meta_matches_module() {
        // The config and the module it builds expose the same meta.
        let config = ConvSeq2dConfig::new(vec![block_config(2, 4, 2), block_config(4, 8, 2)]);
        assert_eq!(config.len(), 2);
        assert_eq!(config.in_channels(), 2);
        assert_eq!(config.out_channels(), 8);
        assert_eq!(config.stride(), [4, 4]);
        assert_eq!(
            config.try_output_shape([1, 2, 16, 16]).unwrap(),
            [1, 8, 4, 4]
        );

        let seq: ConvSeq2d = config.init(&cpu_device());
        assert_eq!(seq.in_channels(), config.in_channels());
        assert_eq!(seq.out_channels(), config.out_channels());
        assert_eq!(seq.stride(), config.stride());
        assert_eq!(
            seq.try_output_shape([1, 2, 16, 16]).unwrap(),
            config.try_output_shape([1, 2, 16, 16]).unwrap()
        );
    }

    #[test]
    fn test_output_resolution() {
        // "same"-padded, stride-2 blocks: out = floor((in + 1) / 2) per dim.
        let seq = ConvSeq2d::try_new(vec![block(2, 4, 2), block(4, 8, 2)]).unwrap();
        // [16, 16] -> [8, 8] -> [4, 4]
        assert_eq!(seq.try_output_resolution([16, 16]).unwrap(), [4, 4]);

        // [12, 12] -> [6, 6] -> [3, 3]
        assert_eq!(seq.try_output_resolution([12, 12]).unwrap(), [3, 3]);

        // [6, 6] -> [3, 3] -> [2, 2] (true conv arithmetic; no divisibility)
        assert_eq!(seq.try_output_resolution([6, 6]).unwrap(), [2, 2]);
    }

    #[test]
    fn test_output_resolution_dilated() {
        let device = cpu_device().autodiff();
        // Valid-padded, dilated block: out = in - dilation * (kernel - 1).
        let dilated = ConvBlock2dConfig::new(
            Conv2dConfig::new([2, 4], [3, 3])
                .with_stride([1, 1])
                .with_dilation([2, 2])
                .with_padding(PaddingConfig2d::Valid)
                .with_bias(false),
        )
        .with_norm(None)
        .with_act(None)
        .init(&device);
        let seq = ConvSeq2d::try_new(vec![dilated]).unwrap();

        // kernel_width = 1 + 2 * (3 - 1) = 5; in - 4 per dim.
        assert_eq!(seq.try_output_resolution([10, 12]).unwrap(), [6, 8]);
        assert_eq!(seq.try_output_shape([1, 2, 10, 12]).unwrap(), [1, 4, 6, 8]);

        let input = Tensor::<4>::random([1, 2, 10, 12], Distribution::Default, &device);
        assert_eq!(seq.forward(input).dims(), [1, 4, 6, 8]);
    }

    #[test]
    fn test_output_shape_matches_forward() {
        let device = cpu_device().autodiff();
        let seq = ConvSeq2d::try_new(vec![block(2, 4, 2), block(4, 8, 2)]).unwrap();

        let batch_size = 3;
        let input = Tensor::<4>::random([batch_size, 2, 16, 16], Distribution::Default, &device);

        let predicted = seq.try_output_shape([batch_size, 2, 16, 16]).unwrap();
        let actual = seq.forward(input).dims();

        assert_eq!(predicted, [batch_size, 8, 4, 4]);
        assert_eq!(predicted, actual);
    }

    #[test]
    fn test_output_shape_channel_mismatch() {
        let seq = ConvSeq2d::try_new(vec![block(2, 4, 1)]).unwrap();
        assert!(seq.try_output_shape([1, 3, 8, 8]).is_err());
    }

    #[test]
    fn test_forward_matches_sequential() {
        let device = cpu_device().autodiff();
        let blocks = vec![block(2, 4, 2), block(4, 8, 1)];
        let seq = ConvSeq2d::try_new(blocks).unwrap();

        let input = Tensor::<4>::random([2, 2, 8, 8], Distribution::Default, &device);

        let output = seq.forward(input.clone());
        let expected = {
            let mut x = input;
            for b in &seq.blocks {
                x = b.forward(x);
            }
            x
        };
        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
