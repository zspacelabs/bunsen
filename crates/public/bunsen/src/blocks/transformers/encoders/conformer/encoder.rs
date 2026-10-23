//! The `FastConformer` encoder.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use super::{
    ConformerBlock,
    ConformerBlockConfig,
    DwStridingSubsampling,
    DwStridingSubsamplingConfig,
    DwStridingSubsamplingMeta,
};
use crate::{
    burner::module::{
        ModuleInit,
        Rekey,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
    ops::transformers::attention::{
        length_mask,
        rel_position_table,
    },
};

/// Common meta for [`ConformerEncoder`] and [`ConformerEncoderConfig`].
pub trait ConformerEncoderMeta {
    /// The model width.
    fn d_model(&self) -> usize;

    /// The number of Conformer blocks.
    fn n_layers(&self) -> usize;

    /// The time reduction of the subsampling.
    fn subsampling_factor(&self) -> usize;
}

/// Config for [`ConformerEncoder`].
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

impl ModuleInit<ConformerEncoder> for ConformerEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ConformerEncoder> {
        if self.subsampling.d_model != self.block.d_model {
            return Err(ConstraintError::new(
                "ConformerEncoderConfig",
                "",
                Rule::Relation {
                    lhs: (
                        "subsampling.d_model".into(),
                        self.subsampling.d_model.to_string(),
                    ),
                    op: "==",
                    rhs: ("block.d_model".into(), self.block.d_model.to_string()),
                },
            )
            .into());
        }
        Ok(ConformerEncoder {
            subsampling: self.subsampling.try_init(device)?,
            layers: (0..self.n_layers)
                .map(|_| self.block.try_init(device))
                .collect::<BunsenResult<_>>()?,
            scale_input: self.scale_input,
        })
    }
}

/// The `FastConformer` encoder: [`DwStridingSubsampling`], then a stack of
/// [`ConformerBlock`]s sharing one relative-position table.
///
/// This is the encoder of NVIDIA's Parakeet and Canary models.
///
/// Built by [`ConformerEncoderConfig`].
/// Implements [`ConformerEncoderMeta`].
#[derive(Module, Debug)]
pub struct ConformerEncoder {
    /// Subsamples the features and projects them to `d_model`.
    pub subsampling: DwStridingSubsampling,

    /// The Conformer blocks.
    pub layers: Vec<ConformerBlock>,

    /// Whether the subsampled input is scaled by `sqrt(d_model)`.
    pub scale_input: bool,
}

impl ConformerEncoderMeta for ConformerEncoder {
    fn d_model(&self) -> usize {
        self.subsampling.d_model()
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn subsampling_factor(&self) -> usize {
        self.subsampling.factor()
    }
}

impl ConformerEncoder {
    /// The encoder's output length for `len` input frames.
    pub fn output_len(
        &self,
        len: usize,
    ) -> usize {
        self.subsampling.output_len(len)
    }

    /// Encodes features: [`pre_encode`](Self::pre_encode), then
    /// [`forward_pre_encoded`](Self::forward_pre_encoded).
    ///
    /// # Arguments
    /// * `features` - `[batch, n_mels, time]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    ///
    /// # Returns
    /// `[batch, output_len(time), d_model]`, and each item's output length.
    /// Frames past an item's length hold values that mean nothing.
    pub fn forward(
        &self,
        features: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        let padded = lengths.is_some();
        let (x, lengths) = self.pre_encode(features, lengths);
        let x = self.forward_pre_encoded(x, padded.then_some(lengths.as_slice()));
        (x, lengths)
    }

    /// The subsampling front alone: the pre-encode embeddings, before the
    /// input scaling and the position table. A caller that edits the frame
    /// sequence between the two halves (a streaming cache splicing in earlier
    /// frames) runs this and then
    /// [`forward_pre_encoded`](Self::forward_pre_encoded).
    ///
    /// # Arguments
    /// * `features` - `[batch, n_mels, time]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `time`.
    ///
    /// # Returns
    /// `[batch, output_len(time), d_model]`, and each item's output length.
    pub fn pre_encode(
        &self,
        features: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> (Tensor<3>, Vec<usize>) {
        #[cfg(any(debug_assertions, test))]
        let [batch, time_in] = crate::contracts::unpack_shape_contract!(
            ["batch", "n_mels", "time"],
            &features.dims(),
            &["batch", "time"],
        );

        let (x, lengths) = self.subsampling.forward(features, lengths);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "time", "d_model"],
            &x.dims(),
            &[
                ("batch", batch),
                ("time", self.output_len(time_in)),
                ("d_model", self.d_model()),
            ],
        );

        (x, lengths)
    }

    /// The encoder past its subsampling: the input scaling (when the encoder
    /// scales), the relative-position table over the frames given, and the
    /// Conformer blocks.
    ///
    /// # Arguments
    /// * `x` - `[batch, frames, d_model]` pre-encode embeddings, as
    ///   [`pre_encode`](Self::pre_encode) gives them.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `frames`.
    ///
    /// # Returns
    /// `[batch, frames, d_model]`. Frames past an item's length hold values
    /// that mean nothing.
    pub fn forward_pre_encoded(
        &self,
        x: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        let [batch, frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "frames", "d_model"],
            &x.dims(),
            &["batch", "frames"],
            &[("d_model", self.d_model())],
        );

        let mut x = x;
        if self.scale_input {
            x = x.mul_scalar((self.d_model() as f64).sqrt());
        }
        let device = x.device();
        let positions = rel_position_table(frames, self.d_model(), &device);
        let valid = lengths.map(|lengths| length_mask(lengths, frames, &device));
        for layer in &self.layers {
            x = layer.forward(x, positions.clone(), valid.clone());
        }

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "frames", "d_model"],
            &x.dims(),
            &[
                ("batch", batch),
                ("frames", frames),
                ("d_model", self.d_model()),
            ],
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;

        x
    }
}

impl ConformerEncoder {
    /// Its tensors as `transformers`' `ParakeetEncoder` names them, relative
    /// to it: the subsampling's and each block's maps
    /// ([`DwStridingSubsampling::transformers_rekey`],
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
            .mount_indexed("layers.", "layers.", ConformerBlock::transformers_rekey())
    }

    /// Its tensors as `NeMo`'s `ConformerEncoder` names them, relative to it:
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
            .mount_indexed("layers.", "layers.", ConformerBlock::nemo_rekey())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_meta() {
        let config = ConformerEncoderConfig::new(
            DwStridingSubsamplingConfig::new(24, 4, 16),
            ConformerBlockConfig::new(16, 2, 32, 5),
            2,
        );
        let module: ConformerEncoder = config.init(&cpu_device());
        assert_eq!(module.d_model(), config.d_model());
        assert_eq!(module.n_layers(), config.n_layers());
        assert_eq!(module.subsampling_factor(), config.subsampling_factor());

        let (y, lengths) = module.forward(Tensor::zeros([1, 24, 20], &cpu_device()), None);
        assert_eq!(y.dims(), [1, 3, 16]);
        assert_eq!(lengths, [3]);
    }

    /// A small encoder that scales its input, as Sortformer v2's does.
    fn scaling_encoder() -> ConformerEncoder {
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
        assert!(ModuleInit::<ConformerEncoder>::try_init(&config, &cpu_device()).is_err());
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
                Some("layers.3.self_attn.pos.weight"),
            ),
            (
                "layers.0.self_attn.pos_bias_u",
                Some("layers.0.self_attn.pos_bias_u"),
            ),
            (
                "layers.1.conv.batch_norm.running_var",
                Some("layers.1.conv.depthwise.norm.running_var"),
            ),
            ("layers.1.conv.batch_norm.num_batches_tracked", None),
            (
                "layers.0.norm_feed_forward1.weight",
                Some("layers.0.norm_feed_forward1.weight"),
            ),
            ("pos_enc.pe", None),
        ] {
            assert_eq!(
                ConformerEncoder::nemo_rekey(3).apply(nemo).as_deref(),
                bunsen,
                "{nemo}"
            );
        }
    }
}
