//! The Parakeet TDT model.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    tensor::Device,
};

use super::{
    TdtJoint,
    TdtJointConfig,
    TdtPredictionNetwork,
    TdtPredictionNetworkConfig,
};
use crate::{
    blocks::conformer::{
        ConformerBlockConfig,
        ConformerEncoder,
        ConformerEncoderConfig,
        ConformerEncoderMeta,
        DwStridingSubsamplingConfig,
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// Common meta for [`ParakeetTdt`] and [`ParakeetTdtConfig`].
pub trait ParakeetTdtMeta {
    /// The encoder's width.
    fn d_model(&self) -> usize;

    /// The prediction network's and joint's width.
    fn d_pred(&self) -> usize;

    /// The token count, the blank included.
    fn n_vocab(&self) -> usize;

    /// The durations, in encoder frames, the joint's duration logits score.
    fn durations(&self) -> &[usize];

    /// The encoder's time reduction.
    fn subsampling_factor(&self) -> usize;

    /// The blank token: the last.
    fn blank_id(&self) -> usize {
        self.n_vocab() - 1
    }
}

/// Config for [`ParakeetTdt`].
///
/// Implements [`ParakeetTdtMeta`].
#[derive(Config, Debug)]
pub struct ParakeetTdtConfig {
    /// The `FastConformer` encoder.
    pub encoder: ConformerEncoderConfig,

    /// The token count, the blank included.
    pub n_vocab: usize,

    /// The prediction network's and joint's width.
    pub d_pred: usize,

    /// The prediction network's LSTM layer count.
    #[config(default = "2")]
    pub n_pred_layers: usize,

    /// The durations, in encoder frames, the joint scores.
    #[config(default = "vec![0, 1, 2, 3, 4]")]
    pub durations: Vec<usize>,
}

impl ParakeetTdtConfig {
    /// NVIDIA's `parakeet-tdt-0.6b-v3`, and its derivatives: 128 mels, a
    /// 24-layer 1024-wide `FastConformer` with 8 heads and no projection
    /// biases, 8193 tokens, a 2-layer 640-wide prediction network, and
    /// durations 0 to 4.
    pub fn tdt_0_6b_v3() -> Self {
        Self::new(
            ConformerEncoderConfig::new(
                DwStridingSubsamplingConfig::new(128, 256, 1024),
                ConformerBlockConfig::new(1024, 8, 4096, 9)
                    .with_bias(false)
                    .with_conv_bias(false),
                24,
            ),
            8193,
            640,
        )
    }
}

impl ParakeetTdtMeta for ParakeetTdtConfig {
    fn d_model(&self) -> usize {
        self.encoder.d_model()
    }

    fn d_pred(&self) -> usize {
        self.d_pred
    }

    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn durations(&self) -> &[usize] {
        &self.durations
    }

    fn subsampling_factor(&self) -> usize {
        self.encoder.subsampling_factor()
    }
}

impl ModuleInit<ParakeetTdt> for ParakeetTdtConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ParakeetTdt> {
        if self.durations.is_empty() {
            return Err(
                ConstraintError::new("ParakeetTdtConfig", "durations", Rule::ZeroOrEmpty).into(),
            );
        }
        Ok(ParakeetTdt {
            encoder: self.encoder.try_init(device)?,
            encoder_projector: LinearConfig::new(self.d_model(), self.d_pred).init(device),
            decoder: TdtPredictionNetworkConfig::new(self.n_vocab, self.d_pred)
                .with_n_layers(self.n_pred_layers)
                .try_init(device)?,
            joint: TdtJointConfig::new(self.d_pred, self.n_vocab, self.durations.len())
                .try_init(device)?,
            durations: self.durations.clone(),
        })
    }
}

/// NVIDIA's Parakeet TDT: a `FastConformer` encoder and a token-and-duration
/// transducer (Xu et al. 2023) over it.
///
/// The encoder runs once over the features; then a decode alternates the
/// [`TdtPredictionNetwork`] over the tokens emitted so far and the
/// [`TdtJoint`] over one encoder frame, which scores the next token (or the
/// blank) and how many frames to advance:
/// [`TdtGreedyDecoder`](crate::kits::speech::parakeet::TdtGreedyDecoder).
///
/// The fields carry the names `transformers`' `ParakeetForTDT` gives them.
///
/// Built by [`ParakeetTdtConfig`].
/// Implements [`ParakeetTdtMeta`].
#[derive(Module, Debug)]
pub struct ParakeetTdt {
    /// The `FastConformer` encoder.
    pub encoder: ConformerEncoder,

    /// `d_model -> d_pred`: encoder frames into the joint's space.
    pub encoder_projector: Linear,

    /// The prediction network.
    pub decoder: TdtPredictionNetwork,

    /// The joint network.
    pub joint: TdtJoint,

    /// The durations, in encoder frames, the joint scores.
    #[module(skip)]
    pub durations: Vec<usize>,
}

impl ParakeetTdtMeta for ParakeetTdt {
    fn d_model(&self) -> usize {
        self.encoder.d_model()
    }

    fn d_pred(&self) -> usize {
        self.encoder_projector.weight.dims()[1]
    }

    fn n_vocab(&self) -> usize {
        use super::TdtJointMeta;
        self.joint.n_vocab()
    }

    fn durations(&self) -> &[usize] {
        &self.durations
    }

    fn subsampling_factor(&self) -> usize {
        self.encoder.subsampling_factor()
    }
}

impl ParakeetTdt {
    /// Encodes features, and projects the frames into the joint's space.
    ///
    /// # Arguments
    /// * `features` - `[batch, n_mels, time]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    ///
    /// # Returns
    /// `[batch, time / subsampling_factor, d_pred]` (rounded up), and each
    /// item's output length.
    pub fn encode(
        &self,
        features: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        #[cfg(any(debug_assertions, test))]
        let [batch, time] = crate::contracts::unpack_shape_contract!(
            ["batch", "n_mels", "time"],
            &features.dims(),
            &["batch", "time"],
        );

        let (x, lengths) = self.encoder.forward(features, lengths);
        let y = self.encoder_projector.forward(x);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "d_pred"],
            &y.dims(),
            &[
                ("batch", batch),
                ("time", self.encoder.output_len(time)),
                ("d_pred", self.d_pred()),
            ],
        );

        (y, lengths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    /// A toy config, non-default in every field the meta reads.
    fn toy_config() -> ParakeetTdtConfig {
        ParakeetTdtConfig::new(
            ConformerEncoderConfig::new(
                DwStridingSubsamplingConfig::new(24, 4, 16),
                ConformerBlockConfig::new(16, 2, 32, 5)
                    .with_bias(false)
                    .with_conv_bias(false),
                2,
            ),
            11,
            12,
        )
        .with_durations(vec![0, 1, 2, 3, 4])
    }

    #[test]
    fn test_meta() {
        let config = toy_config().with_durations(vec![0, 2, 4]);
        let model: ParakeetTdt = config.init(&cpu_device());
        assert_eq!(model.d_model(), config.d_model());
        assert_eq!(model.d_pred(), config.d_pred());
        assert_eq!(model.n_vocab(), config.n_vocab());
        assert_eq!(model.durations(), config.durations());
        assert_eq!(model.subsampling_factor(), config.subsampling_factor());
        assert_eq!(model.blank_id(), 10);
    }

    #[test]
    fn test_v3_geometry() {
        let config = ParakeetTdtConfig::tdt_0_6b_v3();
        assert_eq!(config.d_model(), 1024);
        assert_eq!(config.blank_id(), 8192);
        assert_eq!(config.subsampling_factor(), 8);
    }
}
