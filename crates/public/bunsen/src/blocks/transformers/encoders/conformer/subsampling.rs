//! The `FastConformer`'s depthwise-striding subsampling.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
        PaddingConfig2d,
        conv::Conv2dConfig,
    },
    tensor::Device,
};

use crate::{
    blocks::conv::{
        ConvBlock2d,
        ConvBlock2dConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::attention::length_mask,
};

/// Common meta for [`DwStridingSubsampling`] and
/// [`DwStridingSubsamplingConfig`].
pub trait DwStridingSubsamplingMeta {
    /// The mel channels it reads.
    fn n_mels(&self) -> usize;

    /// The convolution channel count.
    fn channels(&self) -> usize;

    /// The output width.
    fn d_model(&self) -> usize;

    /// The number of strided stages.
    fn n_stages(&self) -> usize;

    /// The time (and frequency) reduction: `2^n_stages`.
    fn factor(&self) -> usize {
        1 << self.n_stages()
    }

    /// The output length for an input of `len` frames.
    ///
    /// Each stage is a kernel-3, stride-2 convolution padded by one on each
    /// side, so it maps `len` to `(len - 1) / 2 + 1`.
    fn output_len(
        &self,
        len: usize,
    ) -> usize {
        (0..self.n_stages()).fold(len, |len, _| if len == 0 { 0 } else { (len - 1) / 2 + 1 })
    }

    /// The frequency bins left after subsampling.
    fn n_freq_out(&self) -> usize;
}

/// Config for [`DwStridingSubsampling`].
///
/// Implements [`DwStridingSubsamplingMeta`].
#[derive(Config, Debug)]
pub struct DwStridingSubsamplingConfig {
    /// The feature (mel) count of the input.
    pub n_mels: usize,

    /// The convolution channel count.
    pub channels: usize,

    /// The output width.
    pub d_model: usize,

    /// The number of strided stages: 3 for 8x subsampling.
    #[config(default = "3")]
    pub n_stages: usize,
}

impl DwStridingSubsamplingMeta for DwStridingSubsamplingConfig {
    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn channels(&self) -> usize {
        self.channels
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_stages(&self) -> usize {
        self.n_stages
    }

    fn n_freq_out(&self) -> usize {
        self.output_len(self.n_mels)
    }
}

impl ModuleInit<DwStridingSubsampling> for DwStridingSubsamplingConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<DwStridingSubsampling> {
        if self.n_stages == 0 {
            return Err(ConstraintError::new(
                "DwStridingSubsamplingConfig",
                "n_stages",
                Rule::ZeroOrEmpty,
            )
            .into());
        }
        let c = self.channels;
        let strided = |c_in: usize, groups: usize| {
            Conv2dConfig::new([c_in, c], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_groups(groups)
        };
        let stem = ConvBlock2dConfig::new(strided(1, 1)).try_init(device)?;
        let mut stages = Vec::with_capacity(self.n_stages - 1);
        for _ in 1..self.n_stages {
            stages.push(DwStridingStage {
                depthwise: ConvBlock2dConfig::new(strided(c, c))
                    .with_act(None)
                    .try_init(device)?,
                pointwise: ConvBlock2dConfig::new(Conv2dConfig::new([c, c], [1, 1]))
                    .try_init(device)?,
            });
        }
        Ok(DwStridingSubsampling {
            stem,
            stages,
            linear: LinearConfig::new(c * self.n_freq_out(), self.d_model).init(device),
            n_mels: self.n_mels,
        })
    }
}

/// One later stage of [`DwStridingSubsampling`]: a strided depthwise
/// convolution, then a pointwise convolution and `ReLU`.
#[derive(Module, Debug)]
pub struct DwStridingStage {
    /// Kernel 3, stride 2, one group per channel; no activation.
    pub depthwise: ConvBlock2d,

    /// Kernel 1, then `ReLU`.
    pub pointwise: ConvBlock2d,
}

/// The `FastConformer`'s depthwise-striding subsampling (`dw_striding` in
/// `NeMo`): `n_stages` stride-2 convolutions over time and frequency, then a
/// projection of the flattened channels and remaining frequencies to
/// `d_model`.
///
/// The first stage is a full 3x3 convolution from one channel and `ReLU`; each
/// later one is a [`DwStridingStage`]. Rybakov et al. (2023), "Fast
/// Conformer".
///
/// Padding frames are zeroed after every convolution, so a batch of
/// different-length inputs computes what each would alone.
///
/// Built by [`DwStridingSubsamplingConfig`].
/// Implements [`DwStridingSubsamplingMeta`].
#[derive(Module, Debug)]
pub struct DwStridingSubsampling {
    /// The first stage: kernel 3, stride 2, `1 -> channels`, then `ReLU`.
    pub stem: ConvBlock2d,

    /// The later stages.
    pub stages: Vec<DwStridingStage>,

    /// `channels * n_freq_out -> d_model`.
    pub linear: Linear,

    /// The mel channels it reads: what `n_freq_out` rounds up from.
    #[module(skip)]
    pub n_mels: usize,
}

impl DwStridingSubsamplingMeta for DwStridingSubsampling {
    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn channels(&self) -> usize {
        self.stem.conv.weight.dims()[0]
    }

    fn d_model(&self) -> usize {
        self.linear.weight.dims()[1]
    }

    fn n_stages(&self) -> usize {
        self.stages.len() + 1
    }

    fn n_freq_out(&self) -> usize {
        self.linear.weight.dims()[0] / self.channels()
    }
}

impl DwStridingSubsampling {
    /// Subsamples features.
    ///
    /// # Arguments
    /// * `features` - `[batch, n_mels, time]`.
    /// * `lengths` - each item's valid frame count, `batch` of them; `None`
    ///   means every item fills `time`.
    ///
    /// # Returns
    /// `[batch, output_len(time), d_model]`, and each item's output length.
    pub fn forward(
        &self,
        features: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        let [batch, _, time] = features.dims();
        let device = features.device();
        let mut lengths: Vec<usize> = match lengths {
            Some(lengths) => {
                assert_eq!(lengths.len(), batch, "one length per batch item");
                lengths.to_vec()
            }
            None => vec![time; batch],
        };
        let step = |lengths: &mut Vec<usize>| {
            for len in lengths.iter_mut() {
                *len = if *len == 0 { 0 } else { (*len - 1) / 2 + 1 };
            }
        };
        let masked = |x: Tensor<4>, lengths: &[usize]| {
            let t = x.dims()[2];
            x * length_mask(lengths, t, &device)
                .float()
                .reshape([batch as i32, 1, t as i32, 1])
        };

        // [batch, 1, time, n_mels]: time and frequency as image axes.
        let mut x = features.swap_dims(1, 2).unsqueeze_dim::<4>(1);
        x = self.stem.forward(x);
        step(&mut lengths);
        x = masked(x, &lengths);
        for stage in &self.stages {
            x = stage.depthwise.forward(x);
            step(&mut lengths);
            x = masked(x, &lengths);
            x = masked(stage.pointwise.forward(x), &lengths);
        }

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "channels", "time", "n_freq"],
            &x.dims(),
            &[
                ("batch", batch),
                ("channels", self.channels()),
                ("time", self.output_len(time)),
                ("n_freq", self.n_freq_out()),
            ],
        );
        let [_, c, t, f] = x.dims();
        let x = x.swap_dims(1, 2).reshape([batch, t, c * f]);
        let y = self.linear.forward(x);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "d_model"],
            &y.dims(),
            &[
                ("batch", batch),
                ("time", self.output_len(time)),
                ("d_model", self.d_model()),
            ],
        );

        (y, lengths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = DwStridingSubsamplingConfig::new(24, 4, 16);
        assert_eq!(config.factor(), 8);
        assert_eq!(config.n_freq_out(), 3);
        assert_eq!(config.output_len(53), 7);
        assert_eq!(config.output_len(37), 5);
        assert_eq!(config.output_len(8), 1);
        assert_eq!(config.output_len(0), 0);

        let module: DwStridingSubsampling = config.init(&cpu_device());
        assert_eq!(module.n_mels(), 24);
        assert_eq!(module.n_mels(), config.n_mels());
        assert_eq!(module.channels(), config.channels());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.n_stages(), config.n_stages());
        assert_eq!(module.n_freq_out(), config.n_freq_out());

        let (y, lengths) =
            module.forward(Tensor::zeros([2, 24, 53], &cpu_device()), Some(&[53, 37]));
        assert_eq!(y.dims(), [2, 7, 16]);
        assert_eq!(lengths, [7, 5]);
    }
}
