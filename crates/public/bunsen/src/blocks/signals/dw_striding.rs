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
    burner::{
        module::{
            ModuleInit,
            Rekey,
        },
        tensor::{
            permute_data,
            reshape_data,
        },
    },
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::attention::length_mask,
};

/// How [`DwStridingSubsampling`] flattens its last convolution's output,
/// `[channels, n_freq]` per frame, into the vector its projection reads.
///
/// A layout choice with a loading consequence: the projection's weight
/// columns follow the flatten, so a checkpoint flattened the other way loads
/// through its key map's transform ([`DwStridingSubsamplingMeta::nemo_rekey`],
/// [`transformers_rekey`](DwStridingSubsamplingMeta::transformers_rekey)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FlattenOrder {
    /// Channels outer, frequencies inner: `NeMo`'s and `transformers'`.
    #[default]
    ChannelsFreqs,

    /// Frequencies outer, channels inner.
    FreqsChannels,
}

/// Common meta for [`DwStridingSubsampling`] and
/// [`DwStridingSubsamplingConfig`], and the key maps that follow from it.
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

    /// How the last convolution's output is flattened for the projection.
    fn flatten(&self) -> FlattenOrder;

    /// Its tensors as `transformers`' `ParakeetEncoderSubsampling` names them,
    /// relative to it: the flat `layers.{k}` onto the stem and the stages,
    /// everything else as it is; and the projection's weight permuted when
    /// this flattens otherwise than `transformers` does.
    fn transformers_rekey(&self) -> Rekey {
        reflatten(self, dw_striding_list("layers.", self.n_stages()))
    }

    /// Its tensors as `NeMo`'s `ConvSubsampling` (`pre_encode`) names them,
    /// relative to it: the flat `conv.{k}` onto the stem and the stages, and
    /// `out` the projection, its weight permuted when this flattens otherwise
    /// than `NeMo` does. Anything else is dropped.
    fn nemo_rekey(&self) -> Rekey {
        let conv = dw_striding_list("conv.", self.n_stages());
        reflatten(
            self,
            Rekey::drop_rest()
                .custom(move |name| name.starts_with("conv.").then(|| conv.apply(name)))
                .rename("out.", "linear."),
        )
    }
}

/// `rekey`, and, when `meta` flattens otherwise than the checkpoints bunsen
/// reads (channels outer), the projection's weight permuted to match: in
/// `PyTorch`'s `[out, in]`, `[d_model, channels * n_freq]` to
/// `[d_model, n_freq * channels]`.
fn reflatten<M: DwStridingSubsamplingMeta + ?Sized>(
    meta: &M,
    rekey: Rekey,
) -> Rekey {
    match meta.flatten() {
        FlattenOrder::ChannelsFreqs => rekey,
        FlattenOrder::FreqsChannels => {
            let (c, f) = (meta.channels(), meta.n_freq_out());
            rekey.transform("linear.weight", move |data| {
                let d = data.shape().num_elements() / (c * f);
                let data = reshape_data(data, vec![d, c, f]);
                reshape_data(permute_data(data, &[0, 2, 1]), vec![d, f * c])
            })
        }
    }
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

    /// How the last convolution's output is flattened for the projection.
    #[config(default = "FlattenOrder::ChannelsFreqs")]
    pub flatten: FlattenOrder,
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

    fn flatten(&self) -> FlattenOrder {
        self.flatten
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
            flatten: self.flatten,
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

    /// How the last convolution's output is flattened for the projection.
    #[module(skip)]
    pub flatten: FlattenOrder,
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

    fn flatten(&self) -> FlattenOrder {
        self.flatten
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
        let x = match self.flatten {
            FlattenOrder::ChannelsFreqs => x.swap_dims(1, 2).reshape([batch, t, c * f]),
            FlattenOrder::FreqsChannels => x.permute([0, 2, 3, 1]).reshape([batch, t, f * c]),
        };
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

/// The stem or stage module a flat `ModuleList` index names, under
/// [`DwStridingSubsampling`]'s layout: `0` is the stem's convolution; from
/// `2`, each stage is a depthwise convolution, a pointwise convolution and an
/// activation, in threes. `None` for an activation, or past the stages.
fn dw_striding_module(
    k: usize,
    n_stages: usize,
) -> Option<String> {
    match k {
        0 => Some("stem.conv".to_string()),
        k if k >= 2 && (k - 2) / 3 + 1 < n_stages => match (k - 2) % 3 {
            0 => Some(format!("stages.{}.depthwise.conv", (k - 2) / 3)),
            1 => Some(format!("stages.{}.pointwise.conv", (k - 2) / 3)),
            _ => None,
        },
        _ => None,
    }
}

/// `{prefix}{k}.{param}`, a flat `ModuleList`'s entry, to its module's path.
fn dw_striding_list(
    prefix: &'static str,
    n_stages: usize,
) -> Rekey {
    Rekey::keep_rest().custom(move |name| {
        let (k, param) = name.strip_prefix(prefix)?.split_once('.')?;
        let module = k.parse().ok().and_then(|k| dw_striding_module(k, n_stages));
        Some(module.map(|module| format!("{module}.{param}")))
    })
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
