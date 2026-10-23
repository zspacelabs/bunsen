//! The Sortformer model.
use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        Linear,
        LinearConfig,
        activation::ActivationConfig,
    },
    tensor::Device,
};

use super::{
    SortformerHead,
    SortformerHeadConfig,
    SortformerHeadMeta,
};
use crate::{
    blocks::{
        frame_rate::{
            FeatureStackingConfig,
            SubpixelUpsample1d,
            SubpixelUpsample1dConfig,
            SubpixelUpsample1dMeta,
        },
        layout::AxisOrder,
        transformers::encoders::{
            SubsampledEncoder,
            SubsampledEncoderConfig,
            SubsampledEncoderMeta,
            SubsamplingConfig,
            conformer::{
                ConformerBlockConfig,
                ConformerEncoderConfig,
                DwStridingSubsamplingConfig,
            },
            residual::{
                NormPlacement,
                ResidualEncoderAttentionBlockConfig,
                ResidualEncoderTransformer,
                ResidualEncoderTransformerConfig,
                ResidualEncoderTransformerMeta,
            },
            rope::{
                RopeEncoderBlockConfig,
                RopeTransformerEncoderConfig,
            },
        },
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    contracts::{
        assert_shape_contract,
        unpack_shape_contract,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::attention::length_mask,
};

/// Common meta for [`Sortformer`], [`SortformerStructureConfig`] and the
/// policies that lower to it.
pub trait SortformerMeta {
    /// The mel channels it reads.
    fn n_mels(&self) -> usize;

    /// The encoder's width: what a speaker cache holds per frame.
    fn d_model(&self) -> usize;

    /// The head's width.
    fn d_hidden(&self) -> usize;

    /// The speakers it tells apart.
    fn n_speakers(&self) -> usize;

    /// The mel frames per encoder frame: the cache's frame.
    fn subsampling_factor(&self) -> usize;

    /// The output frames per encoder frame: 1, or the upsampling factor.
    fn output_upsampling(&self) -> usize;
}

/// Config for [`Sortformer`]: the unrolled tree.
///
/// Implements [`SortformerMeta`].
#[derive(Config, Debug)]
pub struct SortformerStructureConfig {
    /// The encoder: pre-encode, then the encoder layers.
    pub encoder: SubsampledEncoderConfig,

    /// The head; the encoder's output is projected to its width.
    pub head: SortformerHeadConfig,

    /// A transformer over the projected frames, at the head's width, when
    /// set.
    #[config(default = "None")]
    pub transformer: Option<ResidualEncoderTransformerConfig>,

    /// Upsamples the projected frames before the head, when set; its width
    /// is the head's.
    #[config(default = "None")]
    pub upsample: Option<SubpixelUpsample1dConfig>,

    /// Whether the model carries a learned silence embedding, the speaker
    /// cache's silence fill.
    #[config(default = "false")]
    pub learned_silence: bool,
}

impl SortformerMeta for SortformerStructureConfig {
    fn n_mels(&self) -> usize {
        self.encoder.n_mels()
    }

    fn d_model(&self) -> usize {
        self.encoder.d_model()
    }

    fn d_hidden(&self) -> usize {
        self.head.d_hidden
    }

    fn n_speakers(&self) -> usize {
        self.head.n_speakers
    }

    fn subsampling_factor(&self) -> usize {
        self.encoder.subsampling_factor()
    }

    fn output_upsampling(&self) -> usize {
        self.upsample.as_ref().map_or(1, |u| u.factor)
    }
}

impl ModuleInit<Sortformer> for SortformerStructureConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Sortformer> {
        let (d_model, d_hidden) = (self.d_model(), self.d_hidden());
        if let Some(transformer) = &self.transformer
            && transformer.d_model() != d_hidden
        {
            return Err(ConstraintError::new(
                "SortformerStructureConfig",
                "",
                Rule::Relation {
                    lhs: (
                        "transformer.d_model()".into(),
                        transformer.d_model().to_string(),
                    ),
                    op: "==",
                    rhs: ("head.d_hidden".into(), d_hidden.to_string()),
                },
            )
            .into());
        }
        if let Some(upsample) = &self.upsample
            && upsample.d_model != d_hidden
        {
            return Err(ConstraintError::new(
                "SortformerStructureConfig",
                "",
                Rule::Relation {
                    lhs: ("upsample.d_model".into(), upsample.d_model.to_string()),
                    op: "==",
                    rhs: ("head.d_hidden".into(), d_hidden.to_string()),
                },
            )
            .into());
        }
        Ok(Sortformer {
            encoder: self.encoder.try_init(device)?,
            projection: LinearConfig::new(d_model, d_hidden).init(device),
            transformer: self
                .transformer
                .as_ref()
                .map(|t| t.try_init(device))
                .transpose()?,
            upsample: self
                .upsample
                .as_ref()
                .map(|u| u.try_init(device))
                .transpose()?,
            head: self.head.try_init(device)?,
            silence: self
                .learned_silence
                .then(|| Param::from_tensor(Tensor::zeros([d_model], device))),
        })
    }
}

/// `NVIDIA`'s Nemotron-3-Diarization family: feature stacking and a
/// rotary-attention transformer, a projection, a sub-pixel upsample back to
/// the mel frame rate, and the speaker head; a learned silence embedding.
/// The defaults are `Nemotron-3-Diarization`'s.
///
/// Implements [`SortformerMeta`]; lowers to [`SortformerStructureConfig`].
#[derive(Config, Debug)]
pub struct NemotronDiarizationConfig {
    /// The mel channels.
    #[config(default = "128")]
    pub n_mels: usize,

    /// The encoder's width.
    #[config(default = "512")]
    pub d_model: usize,

    /// The encoder layers.
    #[config(default = "31")]
    pub n_layers: usize,

    /// The attention heads.
    #[config(default = "8")]
    pub n_heads: usize,

    /// The feed-forward's hidden width over `d_model`.
    #[config(default = "4")]
    pub ff_expansion: usize,

    /// The mel frames stacked into one encoder frame, and the upsampling
    /// back.
    #[config(default = "8")]
    pub subsampling_factor: usize,

    /// The rotary embedding's base.
    #[config(default = "10000")]
    pub rope_base: usize,

    /// The head's width.
    #[config(default = "192")]
    pub d_hidden: usize,

    /// The speakers.
    #[config(default = "8")]
    pub n_speakers: usize,
}

impl NemotronDiarizationConfig {
    /// `Nemotron-3-Diarization`: the defaults.
    pub fn nemotron_3_diarization() -> Self {
        Self::new()
    }
}

impl SortformerMeta for NemotronDiarizationConfig {
    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    fn n_speakers(&self) -> usize {
        self.n_speakers
    }

    fn subsampling_factor(&self) -> usize {
        self.subsampling_factor
    }

    fn output_upsampling(&self) -> usize {
        self.subsampling_factor
    }
}

impl ToStructureConfig for NemotronDiarizationConfig {
    type Structure = SortformerStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<SortformerStructureConfig> {
        // Fed `[batch, time, n_mels]`, the stacking's own layout.
        let encoder = SubsampledEncoderConfig::new(
            SubsamplingConfig::Stacking(
                FeatureStackingConfig::new(self.n_mels, self.d_model)
                    .with_factor(self.subsampling_factor),
            ),
            RopeTransformerEncoderConfig::new(
                RopeEncoderBlockConfig::new(self.d_model, self.n_heads)
                    .with_ff_expansion(self.ff_expansion),
                self.n_layers,
            )
            .with_rope_base(self.rope_base)
            .into(),
        );
        Ok(SortformerStructureConfig::new(
            encoder,
            SortformerHeadConfig::new(self.d_hidden, self.n_speakers),
        )
        .with_upsample(Some(
            SubpixelUpsample1dConfig::new(self.d_hidden).with_factor(self.subsampling_factor),
        ))
        .with_learned_silence(true))
    }
}

/// `NVIDIA`'s streaming Sortformer v2 family: a `FastConformer` (depthwise
/// striding, Conformer blocks, input scaling), a projection, a post-norm
/// `ReLU` transformer at the head's width, and the speaker head, at the
/// encoder's frame rate; no learned silence embedding. The defaults are
/// `diar_streaming_sortformer_4spk-v2`'s, which v2.1 shares.
///
/// Implements [`SortformerMeta`]; lowers to [`SortformerStructureConfig`].
#[derive(Config, Debug)]
pub struct SortformerV2Config {
    /// The mel channels.
    #[config(default = "128")]
    pub n_mels: usize,

    /// The encoder's width.
    #[config(default = "512")]
    pub d_model: usize,

    /// The Conformer blocks.
    #[config(default = "17")]
    pub n_layers: usize,

    /// The Conformer's attention heads.
    #[config(default = "8")]
    pub n_heads: usize,

    /// The Conformer's feed-forward width over `d_model`.
    #[config(default = "4")]
    pub ff_expansion: usize,

    /// The mel frames per encoder frame: a power of two.
    #[config(default = "8")]
    pub subsampling_factor: usize,

    /// The subsampling's convolution channels.
    #[config(default = "256")]
    pub subsampling_channels: usize,

    /// The Conformer's depthwise convolution kernel.
    #[config(default = "9")]
    pub conv_kernel_size: usize,

    /// The head's width, and the transformer's.
    #[config(default = "192")]
    pub d_hidden: usize,

    /// The transformer's layers.
    #[config(default = "18")]
    pub transformer_layers: usize,

    /// The transformer's attention heads.
    #[config(default = "8")]
    pub transformer_heads: usize,

    /// The transformer's feed-forward width over `d_hidden`.
    #[config(default = "4")]
    pub transformer_ff_expansion: usize,

    /// The speakers.
    #[config(default = "4")]
    pub n_speakers: usize,
}

impl SortformerV2Config {
    /// `diar_streaming_sortformer_4spk-v2` and `-v2.1`: the defaults.
    pub fn diar_streaming_sortformer_4spk_v2() -> Self {
        Self::new()
    }
}

impl SortformerMeta for SortformerV2Config {
    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    fn n_speakers(&self) -> usize {
        self.n_speakers
    }

    fn subsampling_factor(&self) -> usize {
        self.subsampling_factor
    }

    fn output_upsampling(&self) -> usize {
        1
    }
}

impl ToStructureConfig for SortformerV2Config {
    type Structure = SortformerStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<SortformerStructureConfig> {
        let factor = self.subsampling_factor;
        if factor < 2 || !factor.is_power_of_two() {
            return Err(ConstraintError::new(
                "SortformerV2Config",
                "subsampling_factor",
                Rule::OutOfRange {
                    value: factor.to_string(),
                    range: "a power of two, >= 2".into(),
                },
            )
            .into());
        }
        let encoder = ConformerEncoderConfig::new(
            DwStridingSubsamplingConfig::new(self.n_mels, self.subsampling_channels, self.d_model)
                .with_n_stages(factor.trailing_zeros() as usize),
            ConformerBlockConfig::new(
                self.d_model,
                self.n_heads,
                self.d_model * self.ff_expansion,
                self.conv_kernel_size,
            ),
            self.n_layers,
        )
        .with_scale_input(true);
        let block = ResidualEncoderAttentionBlockConfig::new(self.d_hidden)
            .with_d_head(self.d_hidden / self.transformer_heads.max(1))
            .with_expansion_factor(self.transformer_ff_expansion)
            .with_key_bias(true)
            .with_norm_placement(NormPlacement::Post)
            .with_activation(ActivationConfig::Relu);
        Ok(SortformerStructureConfig::new(
            // Fed `[batch, time, n_mels]`; the striding reads `[batch, n_mels,
            // time]`.
            encoder.try_to_structure()?.with_layout(AxisOrder::SWAP_IN),
            SortformerHeadConfig::new(self.d_hidden, self.n_speakers),
        )
        .with_transformer(Some(ResidualEncoderTransformerConfig::new(
            block,
            self.transformer_layers,
        ))))
    }
}

/// A Sortformer's speaker activity: sigmoid probabilities per speaker, at the
/// output's frame rate and at the encoder's.
#[derive(Clone, Debug)]
pub struct SpeakerProbs {
    /// `[batch, frames * output_upsampling, n_speakers]`, the output.
    pub output: Tensor<3>,

    /// `[batch, frames, n_speakers]`, per encoder frame: the output when it
    /// is not upsampled, else the mean of each frame's `output_upsampling`
    /// outputs. What a speaker cache scores.
    pub encoder_rate: Tensor<3>,
}

/// NVIDIA's Sortformer speaker diarizer: an encoder over mels, a projection to
/// the head's width, optionally a transformer at that width and a sub-pixel
/// upsample back to the mel frame rate, and a head scoring each speaker per
/// frame, sigmoid.
///
/// Speakers are numbered in the order the model first hears them. The model
/// holds no stream state: a speaker cache over [`pre_encode`]'s embeddings is
/// the caller's, fed back through [`forward_pre_encoded`].
///
/// [`pre_encode`]: Self::pre_encode
/// [`forward_pre_encoded`]: Self::forward_pre_encoded
///
/// Built by [`SortformerStructureConfig`], or a policy that lowers to it
/// ([`NemotronDiarizationConfig`], [`SortformerV2Config`]).
/// Implements [`SortformerMeta`].
#[derive(Module, Debug)]
pub struct Sortformer {
    /// The encoder.
    pub encoder: SubsampledEncoder,

    /// `[d_model] -> [d_hidden]`.
    pub projection: Linear,

    /// The transformer over the projected frames, if any.
    pub transformer: Option<ResidualEncoderTransformer>,

    /// The upsample back to the mel frame rate, if any.
    pub upsample: Option<SubpixelUpsample1d>,

    /// The speaker head.
    pub head: SortformerHead,

    /// `[d_model]`: the learned silence embedding a speaker cache fills its
    /// silence slots with, if the model has one.
    pub silence: Option<Param<Tensor<1>>>,
}

impl SortformerMeta for Sortformer {
    fn n_mels(&self) -> usize {
        self.encoder.n_mels()
    }

    fn d_model(&self) -> usize {
        self.encoder.d_model()
    }

    fn d_hidden(&self) -> usize {
        self.head.d_hidden()
    }

    fn n_speakers(&self) -> usize {
        self.head.n_speakers()
    }

    fn subsampling_factor(&self) -> usize {
        self.encoder.subsampling_factor()
    }

    fn output_upsampling(&self) -> usize {
        self.upsample.as_ref().map_or(1, |u| u.factor())
    }
}

impl Sortformer {
    /// The pre-encode embeddings of `mels`: what a speaker cache holds.
    ///
    /// # Arguments
    /// * `mels` - `[batch, time, n_mels]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    ///
    /// # Returns
    /// `[batch, ceil(time / subsampling_factor), d_model]`, and each item's
    /// length.
    pub fn pre_encode(
        &self,
        mels: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        assert_shape_contract!(
            ["batch", "time", "n_mels"],
            &mels.dims(),
            &[("n_mels", self.n_mels())],
        );
        self.encoder.pre_encode(mels, lengths)
    }

    /// The speaker activity of pre-encode embeddings: the encoder layers, the
    /// projection, the transformer, the upsample, the head, sigmoid; zero past
    /// each item's length.
    ///
    /// # Arguments
    /// * `embs` - `[batch, frames, d_model]`, a speaker cache's frames and a
    ///   chunk's, as one sequence.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `frames`.
    pub fn forward_pre_encoded(
        &self,
        embs: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> SpeakerProbs {
        let [batch, frames] = unpack_shape_contract!(
            ["batch", "frames", "d_model"],
            &embs.dims(),
            &["batch", "frames"],
            &[("d_model", self.d_model())],
        );
        let device = embs.device();
        let up = self.output_upsampling();

        let x = self.encoder.forward_pre_encoded(embs, lengths);
        let x = self.projection.forward(x);
        let x = match &self.transformer {
            Some(transformer) => transformer.forward(x, lengths),
            None => x,
        };
        let x = match &self.upsample {
            Some(upsample) => upsample.forward(x),
            None => x,
        };
        let mut output = burn::tensor::activation::sigmoid(self.head.forward(x));
        if let Some(lengths) = lengths {
            let out_lengths: Vec<usize> = lengths.iter().map(|&n| n * up).collect();
            let valid = length_mask(&out_lengths, frames * up, &device)
                .float()
                .unsqueeze_dim(2);
            output = output * valid;
        }
        let encoder_rate = if up == 1 {
            output.clone()
        } else {
            output
                .clone()
                .reshape([batch, frames, up, self.n_speakers()])
                .mean_dim(2)
                .reshape([batch, frames, self.n_speakers()])
        };

        #[cfg(any(debug_assertions, test))]
        {
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "frames" * "up", "speakers"],
                &output.dims(),
                &[
                    ("batch", batch),
                    ("frames", frames),
                    ("up", up),
                    ("speakers", self.n_speakers())
                ],
            );
            crate::contracts::assert_shape_contract_periodically!(
                ["batch", "frames", "speakers"],
                &encoder_rate.dims(),
                &[
                    ("batch", batch),
                    ("frames", frames),
                    ("speakers", self.n_speakers())
                ],
            );
        }
        SpeakerProbs {
            output,
            encoder_rate,
        }
    }

    /// The speaker activity of `mels`, whole: [`pre_encode`](Self::pre_encode)
    /// then [`forward_pre_encoded`](Self::forward_pre_encoded), with no
    /// speaker cache.
    ///
    /// # Arguments
    /// * `mels` - `[batch, time, n_mels]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    pub fn forward(
        &self,
        mels: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> SpeakerProbs {
        let padded = lengths.is_some();
        let (embs, lengths) = self.pre_encode(mels, lengths);
        self.forward_pre_encoded(embs, padded.then_some(lengths.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::support::testing::{
        cpu_device,
        seeded_tensor,
    };

    /// A toy policy, non-default in every field the meta reads.
    pub(crate) fn toy() -> NemotronDiarizationConfig {
        NemotronDiarizationConfig::new()
            .with_n_mels(24)
            .with_d_model(16)
            .with_n_layers(2)
            .with_n_heads(2)
            .with_ff_expansion(2)
            .with_d_hidden(8)
            .with_n_speakers(3)
            .with_subsampling_factor(4)
            .with_rope_base(500)
    }

    fn assert_meta_agree(
        a: &impl SortformerMeta,
        b: &impl SortformerMeta,
    ) {
        assert_eq!(a.n_mels(), b.n_mels());
        assert_eq!(a.d_model(), b.d_model());
        assert_eq!(a.d_hidden(), b.d_hidden());
        assert_eq!(a.n_speakers(), b.n_speakers());
        assert_eq!(a.subsampling_factor(), b.subsampling_factor());
        assert_eq!(a.output_upsampling(), b.output_upsampling());
    }

    /// The policy, its structure and the module answer the meta alike, and
    /// both pathways build the same module.
    #[test]
    fn test_meta_and_pathways() {
        let policy = toy();
        let structure = policy.to_structure();
        let device = cpu_device();
        let direct: Sortformer = policy.init(&device);
        let lowered: Sortformer = structure.init(&device);
        assert_meta_agree(&policy, &structure);
        assert_meta_agree(&policy, &direct);
        assert_meta_agree(&direct, &lowered);
        assert_eq!(direct.num_params(), lowered.num_params());
        assert!(direct.silence.is_some());

        let bad = structure.with_upsample(Some(SubpixelUpsample1dConfig::new(5)));
        assert!(ModuleInit::<Sortformer>::try_init(&bad, &device).is_err());
    }

    #[test]
    fn test_nemotron_geometry() {
        let config = NemotronDiarizationConfig::nemotron_3_diarization();
        assert_eq!(config.n_mels(), 128);
        assert_eq!(config.d_model(), 512);
        assert_eq!(config.n_speakers(), 8);
        assert_eq!(config.output_upsampling(), 8);
    }

    /// The whole forward is the two halves; the encoder rate is the output's
    /// mean over each encoder frame; padding is zero.
    #[test]
    fn test_halves_and_rates() {
        let device = cpu_device();
        let model: Sortformer = toy().init(&device);
        let mels = seeded_tensor::<3>(4, [1, 22, 24], Distribution::Default, &device);

        let whole = model.forward(mels.clone(), None);
        let (embs, lengths) = model.pre_encode(mels, None);
        assert_eq!(embs.dims(), [1, 6, 16]);
        assert_eq!(lengths, [6]);
        let halves = model.forward_pre_encoded(embs, None);
        assert_eq!(whole.output.dims(), [1, 24, 3]);
        whole
            .output
            .clone()
            .into_data()
            .assert_eq(&halves.output.into_data(), true);

        let mean = whole
            .output
            .clone()
            .reshape([1, 6, 4, 3])
            .mean_dim(2)
            .reshape([1, 6, 3]);
        whole
            .encoder_rate
            .into_data()
            .assert_approx_eq::<f32>(&mean.into_data(), burn::tensor::Tolerance::absolute(1e-6));

        let embs = seeded_tensor::<3>(5, [1, 6, 16], Distribution::Default, &device);
        let padded = model.forward_pre_encoded(embs, Some(&[4]));
        let tail: f32 = padded.output.slice_dim(1, 16..24).abs().sum().into_scalar();
        assert_eq!(tail, 0.0);
    }
}
