//! The classifier head: a global pool and a linear layer.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
        pool::{
            AdaptiveAvgPool2d,
            AdaptiveAvgPool2dConfig,
        },
    },
    prelude::Tensor,
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
};

/// The global pool over a feature map: the config of a [`GlobalPool2d`].
///
/// A seam: each kind is a module of its own. Built today: the average
/// (timm's `avg`). timm's `max`, `avgmax` and `catavgmax` are further kinds.
#[derive(Config, Debug)]
pub enum GlobalPool2dConfig {
    /// The mean over each channel's map.
    Avg,
}

impl ModuleInit<GlobalPool2d> for GlobalPool2dConfig {
    fn try_init(
        &self,
        _device: &Device,
    ) -> BunsenResult<GlobalPool2d> {
        Ok(match self {
            Self::Avg => GlobalPool2d::Avg(AdaptiveAvgPool2dConfig::new([1, 1]).init()),
        })
    }
}

/// A global pool over a feature map; built by [`GlobalPool2dConfig`].
#[derive(Module, Debug)]
pub enum GlobalPool2d {
    /// The mean over each channel's map.
    Avg(AdaptiveAvgPool2d),
}

impl GlobalPool2d {
    /// Pools each channel's map to one value.
    ///
    /// # Arguments
    ///
    /// - `features`: `[batch, channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, channels]`.
    pub fn forward(
        &self,
        features: Tensor<4>,
    ) -> Tensor<2> {
        match self {
            // [B, C, 1, 1] -> [B, C]
            Self::Avg(pool) => pool.forward(features).flatten(1, 3),
        }
    }
}

/// [`ConvHead`]'s config: the pool, the features it reads, and the classes.
#[derive(Config, Debug)]
pub struct ConvHeadConfig {
    /// The channels of the feature map: the last stage's output.
    pub d_features: usize,

    /// The classes.
    pub num_classes: usize,

    /// The global pool.
    #[config(default = "GlobalPool2dConfig::Avg")]
    pub pool: GlobalPool2dConfig,
}

impl ModuleInit<ConvHead> for ConvHeadConfig {
    /// Builds the head.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for 0 features or 0 classes.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConvHead> {
        if self.d_features == 0 {
            return Err(ConstraintError::zero_or_empty("ConvHeadConfig", "d_features").into());
        }
        if self.num_classes == 0 {
            return Err(ConstraintError::zero_or_empty("ConvHeadConfig", "num_classes").into());
        }
        Ok(ConvHead {
            pool: self.pool.try_init(device)?,
            fc: LinearConfig::new(self.d_features, self.num_classes).init(device),
        })
    }
}

/// A conv net's classifier head: a global pool, then a linear layer. Built
/// by [`ConvHeadConfig`].
#[derive(Module, Debug)]
pub struct ConvHead {
    /// The global pool.
    pub pool: GlobalPool2d,

    /// The classifier.
    pub fc: Linear,
}

impl ConvHead {
    /// The features the head reads.
    pub fn d_features(&self) -> usize {
        self.fc.weight.dims()[0]
    }

    /// The classes.
    pub fn num_classes(&self) -> usize {
        self.fc.weight.dims()[1]
    }

    /// The pooled features.
    ///
    /// # Arguments
    ///
    /// - `features`: `[batch, d_features, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, d_features]`.
    pub fn pool(
        &self,
        features: Tensor<4>,
    ) -> Tensor<2> {
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_features", "height", "width"],
            &features.dims(),
            &[("d_features", self.d_features())],
        );
        self.pool.forward(features)
    }

    /// The logits of pooled features.
    ///
    /// # Arguments
    ///
    /// - `pooled`: `[batch, d_features]`.
    ///
    /// # Returns
    ///
    /// `[batch, num_classes]`.
    pub fn classify(
        &self,
        pooled: Tensor<2>,
    ) -> Tensor<2> {
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "d_features"],
            &pooled.dims(),
            &[("d_features", self.d_features())],
        );
        self.fc.forward(pooled)
    }

    /// The head's logits of a feature map: [`pool`](Self::pool), then
    /// [`classify`](Self::classify).
    pub fn forward(
        &self,
        features: Tensor<4>,
    ) -> Tensor<2> {
        self.classify(self.pool(features))
    }

    /// The head re-initialized for `num_classes` classes, on the same
    /// features.
    pub fn with_classes(
        self,
        num_classes: usize,
    ) -> Self {
        let device = self.fc.weight.device();
        Self {
            fc: LinearConfig::new(self.d_features(), num_classes).init(&device),
            ..self
        }
    }
}
