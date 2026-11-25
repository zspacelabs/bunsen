//! The classifier head: a global pool and a linear layer.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout,
        DropoutConfig,
        Linear,
        LinearConfig,
    },
    prelude::Tensor,
    tensor::{
        Device,
        module::adaptive_avg_pool2d,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
    support::validators::try_probability,
};

/// The global pool over a feature map, timm's `SelectAdaptivePool2d`
/// kinds: the config of a [`GlobalPool2d`].
#[derive(Config, Debug, Copy, PartialEq, Eq)]
pub enum GlobalPool2dConfig {
    /// The mean over each channel's map (timm's `avg`).
    Avg,

    /// The max over each channel's map (timm's `max`).
    Max,

    /// The mean of the two (timm's `avgmax`).
    AvgMax,

    /// The two, concatenated: twice the channels, the means first (timm's
    /// `catavgmax`).
    CatAvgMax,
}

impl GlobalPool2dConfig {
    /// The kind timm names `name` (`avg`, `max`, `avgmax`, `catavgmax`).
    pub fn from_timm(name: &str) -> Option<Self> {
        Some(match name {
            "avg" => Self::Avg,
            "max" => Self::Max,
            "avgmax" => Self::AvgMax,
            "catavgmax" => Self::CatAvgMax,
            _ => return None,
        })
    }

    /// The pooled width of `channels` channels: twice them for
    /// [`CatAvgMax`](Self::CatAvgMax).
    pub fn pooled_features(
        self,
        channels: usize,
    ) -> usize {
        match self {
            Self::CatAvgMax => 2 * channels,
            _ => channels,
        }
    }
}

impl ModuleInit<GlobalPool2d> for GlobalPool2dConfig {
    fn try_init(
        &self,
        _device: &Device,
    ) -> BunsenResult<GlobalPool2d> {
        Ok(GlobalPool2d { kind: *self })
    }
}

/// A global pool over a feature map; built by [`GlobalPool2dConfig`]. No
/// kind has parameters, so the module is its kind.
#[derive(Module, Debug)]
pub struct GlobalPool2d {
    /// The kind.
    #[module(skip)]
    pub kind: GlobalPool2dConfig,
}

impl GlobalPool2d {
    /// The pooled width of `channels` channels.
    pub fn pooled_features(
        &self,
        channels: usize,
    ) -> usize {
        self.kind.pooled_features(channels)
    }

    /// Pools each channel's map.
    ///
    /// # Arguments
    ///
    /// - `features`: `[batch, channels, height, width]`.
    ///
    /// # Returns
    ///
    /// `[batch, pooled]`: `channels` wide, or twice it for
    /// [`CatAvgMax`](GlobalPool2dConfig::CatAvgMax).
    pub fn forward(
        &self,
        features: Tensor<4>,
    ) -> Tensor<2> {
        let avg = |x: Tensor<4>| adaptive_avg_pool2d(x, [1, 1]);
        let max = |x: Tensor<4>| x.max_dim(2).max_dim(3);
        // [B, C', 1, 1] -> [B, C']
        let pooled = match self.kind {
            GlobalPool2dConfig::Avg => avg(features),
            GlobalPool2dConfig::Max => max(features),
            GlobalPool2dConfig::AvgMax => (avg(features.clone()) + max(features)) * 0.5,
            GlobalPool2dConfig::CatAvgMax => {
                Tensor::cat(alloc::vec![avg(features.clone()), max(features)], 1)
            }
        };
        pooled.flatten(1, 3)
    }
}

/// [`ConvHead`]'s config: the pool, the features it reads, the dropout, and
/// the classes.
#[derive(Config, Debug)]
pub struct ConvHeadConfig {
    /// The channels of the feature map: the last stage's output.
    pub d_features: usize,

    /// The classes.
    pub num_classes: usize,

    /// The global pool.
    #[config(default = "GlobalPool2dConfig::Avg")]
    pub pool: GlobalPool2dConfig,

    /// The dropout before the classifier (timm's `drop_rate`; training
    /// only).
    #[config(default = "0.0")]
    pub drop_rate: f64,
}

impl ModuleInit<ConvHead> for ConvHeadConfig {
    /// Builds the head.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for 0 features or 0 classes; for a dropout
    /// that is not a probability.
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
        try_probability(self.drop_rate)?;
        Ok(ConvHead {
            pool: self.pool.try_init(device)?,
            drop: (self.drop_rate != 0.0).then(|| DropoutConfig::new(self.drop_rate).init()),
            fc: LinearConfig::new(self.pool.pooled_features(self.d_features), self.num_classes)
                .init(device),
        })
    }
}

/// A conv net's classifier head: a global pool, a dropout, then a linear
/// layer. Built by [`ConvHeadConfig`].
#[derive(Module, Debug)]
pub struct ConvHead {
    /// The global pool.
    pub pool: GlobalPool2d,

    /// The dropout before the classifier, when its rate is not 0.
    pub drop: Option<Dropout>,

    /// The classifier.
    pub fc: Linear,
}

impl ConvHead {
    /// The channels of the feature map the head reads.
    pub fn d_features(&self) -> usize {
        match self.pool.kind {
            GlobalPool2dConfig::CatAvgMax => self.pooled_features() / 2,
            _ => self.pooled_features(),
        }
    }

    /// The width of the pooled features, the classifier's input.
    pub fn pooled_features(&self) -> usize {
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
    /// `[batch, pooled_features]`.
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

    /// The logits of pooled features, through the dropout.
    ///
    /// # Arguments
    ///
    /// - `pooled`: `[batch, pooled_features]`.
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
            ["batch", "pooled_features"],
            &pooled.dims(),
            &[("pooled_features", self.pooled_features())],
        );
        let pooled = match &self.drop {
            Some(drop) => drop.forward(pooled),
            None => pooled,
        };
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
            fc: LinearConfig::new(self.pooled_features(), num_classes).init(&device),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::support::testing::cpu_device;

    /// Each pool over one map, `[1, 2, 2, 2]`: channel 0 is 1, 2, 3, 6
    /// (mean 3, max 6), channel 1 is all 4.
    #[test]
    fn test_pools() {
        let device = cpu_device();
        let map = Tensor::<4>::from_data(
            TensorData::new(alloc::vec![1f32, 2., 3., 6., 4., 4., 4., 4.], [1, 2, 2, 2]),
            &device,
        );
        for (kind, want) in [
            (GlobalPool2dConfig::Avg, alloc::vec![3f32, 4.]),
            (GlobalPool2dConfig::Max, alloc::vec![6., 4.]),
            (GlobalPool2dConfig::AvgMax, alloc::vec![4.5, 4.]),
            (GlobalPool2dConfig::CatAvgMax, alloc::vec![3., 4., 6., 4.]),
        ] {
            let pool: GlobalPool2d = kind.init(&device);
            assert_eq!(pool.pooled_features(2), want.len(), "{kind:?}");
            let got = pool.forward(map.clone()).into_data();
            assert_eq!(got.try_to_vec::<f32>().unwrap(), want, "{kind:?}");
        }
        assert_eq!(
            GlobalPool2dConfig::from_timm("catavgmax"),
            Some(GlobalPool2dConfig::CatAvgMax)
        );
        assert_eq!(GlobalPool2dConfig::from_timm(""), None);
    }
}
