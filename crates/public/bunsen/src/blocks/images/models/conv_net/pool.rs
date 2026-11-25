//! The global pool over a conv net's feature map.

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::{
        Device,
        module::adaptive_avg_pool2d,
    },
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
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
