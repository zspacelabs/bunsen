//! The `FastConformer` encoder.
use burn::config::Config;

use super::{
    ConformerBlock,
    ConformerBlockConfig,
    ConformerTransformerConfig,
};
use crate::{
    blocks::{
        signals::{
            DwStridingSubsampling,
            DwStridingSubsamplingConfig,
            DwStridingSubsamplingMeta,
            SubsamplingConfig,
        },
        transformers::encoders::SubsampledEncoderConfig,
    },
    burner::module::{
        Rekey,
        ToStructureConfig,
    },
    errors::BunsenResult,
};

/// Common meta for [`ConformerEncoderConfig`].
pub trait ConformerEncoderMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The number of Conformer blocks.
    fn n_layers(&self) -> usize;

    /// The time reduction of the subsampling.
    fn subsampling_factor(&self) -> usize;
}

/// The `FastConformer` encoder, as a policy: [`DwStridingSubsampling`] (8x
/// over time by default), then a
/// [`ConformerTransformer`](super::ConformerTransformer), the stack of
/// [`ConformerBlock`]s sharing one relative-position table. It lowers to a
/// [`SubsampledEncoderConfig`], and builds a
/// [`SubsampledEncoder`](crate::blocks::transformers::encoders::SubsampledEncoder) fed the
/// striding's own `[batch, n_mels, time]`.
///
/// This is the encoder of NVIDIA's Parakeet and Canary models, and of
/// Sortformer v2's.
///
/// Implements [`ConformerEncoderMeta`].
#[derive(Config, Debug)]
pub struct ConformerEncoderConfig {
    /// The subsampling front of the encoder.
    pub subsampling: DwStridingSubsamplingConfig,

    /// The block every layer is built from.
    pub block: ConformerBlockConfig,

    /// The number of Conformer blocks.
    pub n_layers: usize,

    /// Whether the subsampled input is scaled by `sqrt(d_model)` (`NeMo`'s
    /// `xscaling`).
    #[config(default = "false")]
    pub scale_input: bool,
}

impl ConformerEncoderMeta for ConformerEncoderConfig {
    fn d_model(&self) -> usize {
        self.block.d_model
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn subsampling_factor(&self) -> usize {
        self.subsampling.factor()
    }
}

impl ConformerEncoderConfig {
    /// The stack's config: `n_layers` of `block`, scaled as `scale_input`
    /// says.
    pub fn transformer(&self) -> ConformerTransformerConfig {
        ConformerTransformerConfig::new(self.block.clone(), self.n_layers)
            .with_scale_input(self.scale_input)
    }

    /// A `FastConformer`'s tensors as `transformers`' `ParakeetEncoder` names
    /// them, onto the `SubsampledEncoder` this builds: the subsampling's and
    /// each block's maps ([`DwStridingSubsampling::transformers_rekey`],
    /// [`ConformerBlock::transformers_rekey`]) mounted where they sit, and
    /// the batch-norm counters dropped.
    pub fn transformers_rekey(n_stages: usize) -> Rekey {
        Rekey::keep_rest()
            .drop_suffix(".num_batches_tracked")
            .mount(
                "subsampling.",
                "subsampling.",
                DwStridingSubsampling::transformers_rekey(n_stages),
            )
            .mount_indexed(
                "layers.",
                "transformer.layers.",
                ConformerBlock::transformers_rekey(),
            )
    }

    /// A `FastConformer`'s tensors as `NeMo`'s `ConformerEncoder` names them,
    /// onto the `SubsampledEncoder` this builds:
    /// the subsampling's and each block's maps
    /// ([`DwStridingSubsampling::nemo_rekey`], [`ConformerBlock::nemo_rekey`])
    /// mounted where they sit. The batch-norm counters, and anything else, are
    /// dropped.
    pub fn nemo_rekey(n_stages: usize) -> Rekey {
        Rekey::drop_rest()
            .drop_suffix(".num_batches_tracked")
            .mount(
                "pre_encode.",
                "subsampling.",
                DwStridingSubsampling::nemo_rekey(n_stages),
            )
            .mount_indexed(
                "layers.",
                "transformer.layers.",
                ConformerBlock::nemo_rekey(),
            )
    }
}

impl ToStructureConfig for ConformerEncoderConfig {
    type Structure = SubsampledEncoderConfig;

    fn try_to_structure(&self) -> BunsenResult<SubsampledEncoderConfig> {
        Ok(SubsampledEncoderConfig::new(
            SubsamplingConfig::DwStriding(self.subsampling.clone()),
            self.transformer().into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use burn::Tensor;

    use super::*;
    use crate::{
        blocks::transformers::encoders::{
            SubsampledEncoder,
            SubsampledEncoderMeta,
        },
        burner::module::ModuleInit,
        support::testing::cpu_device,
    };

    #[test]
    fn test_meta() {
        let config = ConformerEncoderConfig::new(
            DwStridingSubsamplingConfig::new(24, 4, 16),
            ConformerBlockConfig::new(16, 2, 32, 5),
            2,
        );
        let module: SubsampledEncoder = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.n_layers(), config.n_layers());
        assert_eq!(module.subsampling_factor(), config.subsampling_factor());

        let (y, lengths) = module.forward(Tensor::zeros([1, 24, 20], &cpu_device()), None);
        assert_eq!(y.dims(), [1, 3, 16]);
        assert_eq!(lengths, [3]);
    }

    /// A small encoder that scales its input, as Sortformer v2's does.
    fn scaling_encoder() -> SubsampledEncoder {
        ConformerEncoderConfig::new(
            DwStridingSubsamplingConfig::new(24, 4, 16),
            ConformerBlockConfig::new(16, 2, 32, 5),
            2,
        )
        .with_scale_input(true)
        .init(&cpu_device())
    }

    /// `forward` is exactly `pre_encode` then `forward_pre_encoded`, with and
    /// without lengths; the pre-encode is the subsampling alone, the scaling
    /// left to the second half.
    #[test]
    fn test_split_composes() {
        use burn::tensor::Distribution;
        let device = cpu_device();
        let encoder = scaling_encoder();
        let features = Tensor::<3>::random([2, 24, 40], Distribution::Default, &device);

        let (whole, whole_lengths) = encoder.forward(features.clone(), None);
        let (embs, lengths) = encoder.pre_encode(features.clone(), None);
        assert_eq!(lengths, whole_lengths);
        let (subsampled, _) = encoder.subsampling.forward(features.clone(), None);
        embs.to_data().assert_eq(&subsampled.to_data(), true);
        encoder
            .forward_pre_encoded(embs, None)
            .to_data()
            .assert_eq(&whole.to_data(), true);

        let given = [40, 26];
        let (whole, whole_lengths) = encoder.forward(features.clone(), Some(&given));
        let (embs, lengths) = encoder.pre_encode(features, Some(&given));
        assert_eq!(lengths, whole_lengths);
        assert_eq!(lengths, [encoder.output_len(40), encoder.output_len(26)]);
        encoder
            .forward_pre_encoded(embs, Some(&lengths))
            .to_data()
            .assert_eq(&whole.to_data(), true);
    }

    /// The length mask survives the split: a short item's valid frames,
    /// batched behind a longer one and masked, are what it gives alone.
    #[test]
    fn test_split_masks_padding() {
        use burn::tensor::{
            Distribution,
            Tolerance,
        };
        let device = cpu_device();
        let encoder = scaling_encoder();
        let embs = Tensor::<3>::random([2, 10, 16], Distribution::Default, &device);
        let lengths = [10, 6];

        let batched = encoder.forward_pre_encoded(embs.clone(), Some(&lengths));
        let alone = encoder.forward_pre_encoded(embs.slice([1..2, 0..6]), None);
        batched
            .slice([1..2, 0..6])
            .to_data()
            .assert_approx_eq::<f32>(&alone.to_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_widths_must_agree() {
        let config = ConformerEncoderConfig::new(
            DwStridingSubsamplingConfig::new(24, 4, 12),
            ConformerBlockConfig::new(16, 2, 32, 5),
            2,
        );
        assert!(ModuleInit::<SubsampledEncoder>::try_init(&config, &cpu_device()).is_err());
    }

    /// `NeMo`'s `ConformerEncoder` names, the subsampling's flat `conv`
    /// list and every block's renames among them.
    #[test]
    fn test_nemo_rekey() {
        for (nemo, bunsen) in [
            (
                "pre_encode.conv.0.weight",
                Some("subsampling.stem.conv.weight"),
            ),
            (
                "pre_encode.conv.2.bias",
                Some("subsampling.stages.0.depthwise.conv.bias"),
            ),
            (
                "pre_encode.conv.3.weight",
                Some("subsampling.stages.0.pointwise.conv.weight"),
            ),
            (
                "pre_encode.conv.5.weight",
                Some("subsampling.stages.1.depthwise.conv.weight"),
            ),
            (
                "pre_encode.conv.6.bias",
                Some("subsampling.stages.1.pointwise.conv.bias"),
            ),
            ("pre_encode.conv.4.weight", None),
            ("pre_encode.conv.8.weight", None),
            ("pre_encode.out.weight", Some("subsampling.linear.weight")),
            (
                "layers.3.self_attn.linear_pos.weight",
                Some("transformer.layers.3.self_attn.pos.weight"),
            ),
            (
                "layers.0.self_attn.pos_bias_u",
                Some("transformer.layers.0.self_attn.pos_bias_u"),
            ),
            (
                "layers.1.conv.batch_norm.running_var",
                Some("transformer.layers.1.conv.depthwise.norm.running_var"),
            ),
            ("layers.1.conv.batch_norm.num_batches_tracked", None),
            (
                "layers.0.norm_feed_forward1.weight",
                Some("transformer.layers.0.norm_feed_forward1.weight"),
            ),
            ("pos_enc.pe", None),
        ] {
            assert_eq!(
                ConformerEncoderConfig::nemo_rekey(3).apply(nemo).as_deref(),
                bunsen,
                "{nemo}"
            );
        }
    }
}
