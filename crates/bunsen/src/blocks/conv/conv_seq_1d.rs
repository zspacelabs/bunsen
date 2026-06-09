//! # `ConvSeq1d` - sequence of [`ConvBlock1d`].

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        activation::ActivationConfig,
        norm::NormalizationConfig,
    },
    prelude::Backend,
};

use crate::{
    blocks::conv::{
        ConvBlock1d,
        ConvBlock1dConfig,
        ConvBlock1dMeta,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// [`ConvSeq1d`] Meta.
///
/// Sequence-level metadata, derived from the chain of per-block
/// [`ConvBlock1dMeta`]. Implemented by:
/// * [`ConvSeq1dConfig`]
/// * [`ConvSeq1d`]
pub trait ConvSeq1dMeta {
    /// The per-block [`ConvBlock1dMeta`], in sequence order.
    fn block_metas(&self) -> Vec<&dyn ConvBlock1dMeta>;

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
    /// This is the [`in_channels`](ConvBlock1dMeta::in_channels) of the first
    /// block.
    ///
    /// # Panics
    ///
    /// If the sequence is empty.
    fn in_channels(&self) -> usize {
        self.block_metas()
            .first()
            .expect("ConvSeq1d must have at least one block")
            .in_channels()
    }

    /// The number of output channels of the sequence.
    ///
    /// This is the [`out_channels`](ConvBlock1dMeta::out_channels) of the last
    /// block.
    ///
    /// # Panics
    ///
    /// If the sequence is empty.
    fn out_channels(&self) -> usize {
        self.block_metas()
            .last()
            .expect("ConvSeq1d must have at least one block")
            .out_channels()
    }

    /// The total length stride of the sequence.
    ///
    /// This is the product of each block's
    /// [`stride`](ConvBlock1dMeta::stride).
    fn stride(&self) -> usize {
        self.block_metas()
            .iter()
            .map(|meta| meta.stride()[0])
            .product()
    }

    /// Validates the sequence.
    ///
    /// A legal sequence:
    /// * is non-empty,
    /// * has channel-compatible adjacent blocks; i.e. each block's
    ///   [`out_channels`](ConvBlock1dMeta::out_channels) equals the following
    ///   block's [`in_channels`](ConvBlock1dMeta::in_channels).
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the sequence is empty or has a channel
    /// mismatch between adjacent blocks.
    fn validate(&self) -> BunsenResult<()> {
        let metas = self.block_metas();
        if metas.is_empty() {
            return Err(BunsenError::Invalid(
                "ConvSeq1d must have at least one block".to_string(),
            ));
        }

        for (idx, pair) in metas.windows(2).enumerate() {
            let prev = pair[0];
            let next = pair[1];
            if prev.out_channels() != next.in_channels() {
                return Err(BunsenError::Invalid(format!(
                    "ConvSeq1d block {idx} out_channels ({}) != block {} in_channels ({})",
                    prev.out_channels(),
                    idx + 1,
                    next.in_channels(),
                )));
            }
        }

        Ok(())
    }

    /// Predicts the output length for a given input length.
    ///
    /// Folds the input length through each block's
    /// [`try_output_length`](ConvBlock1dMeta::try_output_length), which models
    /// the true 1D convolution arithmetic (kernel size, padding, dilation, and
    /// stride).
    ///
    /// # Arguments
    ///
    /// * `in_length` - The input length.
    ///
    /// # Returns
    ///
    /// The predicted output length.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if some block has no legal output length for
    /// its input length (the kernel does not fit the padded input).
    fn try_output_length(
        &self,
        in_length: usize,
    ) -> BunsenResult<usize> {
        let mut length = in_length;
        for (idx, meta) in self.block_metas().iter().enumerate() {
            length = meta
                .try_output_length(length)
                .map_err(|err| BunsenError::Invalid(format!("ConvSeq1d block {idx}: {err}")))?;
        }
        Ok(length)
    }

    /// Predicts the output shape for a given input shape.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - The input shape `[batch_size, in_channels,
    ///   in_length]`.
    ///
    /// # Returns
    ///
    /// The predicted output shape `[batch_size, out_channels, out_length]`.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the input channels do not match the
    /// sequence's [`in_channels`](Self::in_channels), or if the input length
    /// has no legal output through the sequence (see
    /// [`Self::try_output_length`]).
    fn try_output_shape(
        &self,
        input_shape: [usize; 3],
    ) -> BunsenResult<[usize; 3]> {
        let [batch_size, in_channels, in_length] = input_shape;
        if in_channels != self.in_channels() {
            return Err(BunsenError::Invalid(format!(
                "ConvSeq1d expected in_channels ({}), got ({in_channels})",
                self.in_channels(),
            )));
        }
        let out_length = self.try_output_length(in_length)?;
        Ok([batch_size, self.out_channels(), out_length])
    }
}

/// [`ConvSeq1d`] Config.
///
/// Implements [`ConvSeq1dMeta`].
///
/// Built into a [`ConvSeq1d`] via [`ModuleInit::init`] /
/// [`ModuleInit::try_init`].
#[derive(Config, Debug)]
pub struct ConvSeq1dConfig {
    /// The [`ConvBlock1dConfig`] modules, in sequence order.
    pub blocks: Vec<ConvBlock1dConfig>,
}

impl ConvSeq1dMeta for ConvSeq1dConfig {
    fn block_metas(&self) -> Vec<&dyn ConvBlock1dMeta> {
        self.blocks
            .iter()
            .map(|config| config as &dyn ConvBlock1dMeta)
            .collect()
    }
}

impl ConvSeq1dConfig {
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

impl<B: Backend> ModuleInit<B, ConvSeq1d<B>> for ConvSeq1dConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> BunsenResult<ConvSeq1d<B>> {
        let blocks = self
            .blocks
            .iter()
            .map(|config| config.try_init(device))
            .collect::<BunsenResult<Vec<_>>>()?;
        ConvSeq1d::try_new(blocks)
    }
}

/// Sequence of [`ConvBlock1d`].
///
/// A non-empty chain of [`ConvBlock1d`] modules, where each block's
/// output channels feed the next block's input channels.
///
/// Implements [`ConvSeq1dMeta`].
///
/// Built (and validated) via [`Self::try_new`], or from a [`ConvSeq1dConfig`].
#[derive(Module, Debug)]
pub struct ConvSeq1d<B: Backend> {
    /// The internal [`ConvBlock1d`] modules.
    pub blocks: Vec<ConvBlock1d<B>>,
}

impl<B: Backend> ConvSeq1dMeta for ConvSeq1d<B> {
    fn block_metas(&self) -> Vec<&dyn ConvBlock1dMeta> {
        self.blocks
            .iter()
            .map(|block| block as &dyn ConvBlock1dMeta)
            .collect()
    }
}

impl<B: Backend> ConvSeq1d<B> {
    /// Creates a new [`ConvSeq1d`] module.
    ///
    /// # Errors
    ///
    /// [`BunsenError::Invalid`] if the blocks do not form a legal sequence; see
    /// [`ConvSeq1dMeta::validate`].
    pub fn try_new(blocks: Vec<ConvBlock1d<B>>) -> BunsenResult<Self> {
        let seq = Self { blocks };
        seq.validate()?;
        Ok(seq)
    }

    /// Performs a forward pass through the sequence of [`ConvBlock1d`] modules.
    ///
    /// # Arguments
    /// * `input` - The input tensor of shape `[batch_size, in_channels,
    ///   in_length]`.
    ///
    /// # Returns
    ///
    /// The output tensor of shape `[batch_size, out_channels, out_length]`; the
    /// shape is predicted by [`ConvSeq1dMeta::try_output_shape`].
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
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
        backend::Autodiff,
        nn::{
            PaddingConfig1d,
            activation::ActivationConfig,
            conv::Conv1dConfig,
        },
        tensor::Distribution,
    };

    use super::*;
    use crate::support::testing::CpuBackend;

    type I = CpuBackend;
    type B = Autodiff<I>;

    /// Builds a "same"-padded `ConvBlock1dConfig` (`out_length = in_length /
    /// stride`).
    fn block_config(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> ConvBlock1dConfig {
        ConvBlock1dConfig::new(
            Conv1dConfig::new(in_channels, out_channels, 3)
                .with_stride(stride)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .with_bias(false),
        )
        .with_norm(None)
        .with_act(Some(ActivationConfig::Relu))
    }

    /// Builds a "same"-padded `ConvBlock1d` (`out_length = in_length /
    /// stride`).
    fn block(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> ConvBlock1d<B> {
        block_config(in_channels, out_channels, stride).init(&Default::default())
    }

    #[test]
    fn test_validate_empty() {
        let err = ConvSeq1d::<B>::try_new(vec![]).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)));

        // The config-level meta validates the same way.
        let err = ConvSeq1dConfig::new(vec![]).validate().unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)));
    }

    #[test]
    fn test_validate_channel_mismatch() {
        // block 0 out_channels = 4, block 1 in_channels = 8 -> mismatch.
        let blocks = vec![block(2, 4, 1), block(8, 16, 1)];
        let err = ConvSeq1d::try_new(blocks).unwrap_err();
        assert!(matches!(err, BunsenError::Invalid(_)));

        // The config rejects it at init.
        let result: BunsenResult<ConvSeq1d<B>> =
            ConvSeq1dConfig::new(vec![block_config(2, 4, 1), block_config(8, 16, 1)])
                .try_init(&Default::default());
        assert!(matches!(result.unwrap_err(), BunsenError::Invalid(_)));
    }

    #[test]
    fn test_meta() {
        let seq = ConvSeq1d::try_new(vec![block(2, 4, 2), block(4, 8, 2), block(8, 8, 1)]).unwrap();
        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());
        assert_eq!(seq.in_channels(), 2);
        assert_eq!(seq.out_channels(), 8);
        assert_eq!(seq.stride(), 4);
    }

    #[test]
    fn test_config_meta_matches_module() {
        // The config and the module it builds expose the same meta.
        let config = ConvSeq1dConfig::new(vec![block_config(2, 4, 2), block_config(4, 8, 2)]);
        assert_eq!(config.len(), 2);
        assert_eq!(config.in_channels(), 2);
        assert_eq!(config.out_channels(), 8);
        assert_eq!(config.stride(), 4);
        assert_eq!(config.try_output_shape([1, 2, 16]).unwrap(), [1, 8, 4]);

        let seq: ConvSeq1d<B> = config.init(&Default::default());
        assert_eq!(seq.in_channels(), config.in_channels());
        assert_eq!(seq.out_channels(), config.out_channels());
        assert_eq!(seq.stride(), config.stride());
        assert_eq!(
            seq.try_output_shape([1, 2, 16]).unwrap(),
            config.try_output_shape([1, 2, 16]).unwrap()
        );
    }

    #[test]
    fn test_output_length() {
        // "same"-padded, stride-2 blocks: out = floor((in + 1) / 2) per block.
        let seq = ConvSeq1d::try_new(vec![block(2, 4, 2), block(4, 8, 2)]).unwrap();
        // 16 -> 8 -> 4
        assert_eq!(seq.try_output_length(16).unwrap(), 4);

        // 12 -> 6 -> 3
        assert_eq!(seq.try_output_length(12).unwrap(), 3);

        // 6 -> 3 -> 2 (true conv arithmetic; no stride-divisibility requirement)
        assert_eq!(seq.try_output_length(6).unwrap(), 2);
    }

    #[test]
    fn test_output_length_dilated() {
        let device = Default::default();
        // Valid-padded, dilated block: out = in - dilation * (kernel - 1).
        let dilated = ConvBlock1dConfig::new(
            Conv1dConfig::new(2, 4, 3)
                .with_stride(1)
                .with_dilation(2)
                .with_padding(PaddingConfig1d::Valid)
                .with_bias(false),
        )
        .with_norm(None)
        .with_act(None)
        .init(&device);
        let seq = ConvSeq1d::try_new(vec![dilated]).unwrap();

        // kernel_width = 1 + 2 * (3 - 1) = 5; 10 - 4 = 6.
        assert_eq!(seq.try_output_length(10).unwrap(), 6);
        assert_eq!(seq.try_output_shape([1, 2, 10]).unwrap(), [1, 4, 6]);

        let input = Tensor::<B, 3>::random([1, 2, 10], Distribution::Default, &device);
        assert_eq!(seq.forward(input).dims(), [1, 4, 6]);
    }

    #[test]
    fn test_output_shape_matches_forward() {
        let device = Default::default();
        let seq = ConvSeq1d::try_new(vec![block(2, 4, 2), block(4, 8, 2)]).unwrap();

        let batch_size = 3;
        let in_length = 16;
        let input =
            Tensor::<B, 3>::random([batch_size, 2, in_length], Distribution::Default, &device);

        let predicted = seq.try_output_shape([batch_size, 2, in_length]).unwrap();
        let actual = seq.forward(input).dims();

        assert_eq!(predicted, [batch_size, 8, 4]);
        assert_eq!(predicted, actual);
    }

    #[test]
    fn test_output_shape_channel_mismatch() {
        let seq = ConvSeq1d::try_new(vec![block(2, 4, 1)]).unwrap();
        assert!(seq.try_output_shape([1, 3, 8]).is_err());
    }

    #[test]
    fn test_forward_matches_sequential() {
        let device = Default::default();
        let blocks = vec![block(2, 4, 2), block(4, 8, 1)];
        let seq = ConvSeq1d::try_new(blocks).unwrap();

        let input = Tensor::<B, 3>::random([2, 2, 8], Distribution::Default, &device);

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
