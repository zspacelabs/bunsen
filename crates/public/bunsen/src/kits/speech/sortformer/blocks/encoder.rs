//! A Sortformer's encoder: pre-encode, then the encoder layers.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::{
    blocks::{
        frame_rate::{
            FeatureStacking,
            FeatureStackingConfig,
            FeatureStackingMeta,
        },
        transformers::rope_encoder::{
            RopeTransformerEncoder,
            RopeTransformerEncoderConfig,
            RopeTransformerEncoderMeta,
        },
    },
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
    },
};

/// Common meta for [`SortformerEncoder`] and [`SortformerEncoderConfig`].
pub trait SortformerEncoderMeta {
    /// The mel channels it reads.
    fn n_mels(&self) -> usize;

    /// Its width: the pre-encode embeddings' and the encoder's.
    fn d_model(&self) -> usize;

    /// The mel frames per encoder frame.
    fn subsampling_factor(&self) -> usize;
}

/// Config for [`SortformerEncoder`]: which encoder, and its config.
///
/// Implements [`SortformerEncoderMeta`].
#[derive(Config, Debug)]
pub enum SortformerEncoderConfig {
    /// Feature stacking, then a pre-norm transformer with rotary attention:
    /// `Nemotron-3-Diarization`'s.
    Rope(RopeSortformerEncoderConfig),
}

impl SortformerEncoderMeta for SortformerEncoderConfig {
    fn n_mels(&self) -> usize {
        match self {
            Self::Rope(c) => c.stacking.d_input,
        }
    }

    fn d_model(&self) -> usize {
        match self {
            Self::Rope(c) => c.encoder.d_model(),
        }
    }

    fn subsampling_factor(&self) -> usize {
        match self {
            Self::Rope(c) => c.stacking.factor,
        }
    }
}

impl ModuleInit<SortformerEncoder> for SortformerEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SortformerEncoder> {
        Ok(match self {
            Self::Rope(c) => SortformerEncoder::Rope(c.try_init(device)?),
        })
    }
}

/// A Sortformer's encoder, split as `NeMo` streams it: [`pre_encode`]
/// (mels to embeddings, the subsampling) and [`forward_pre_encoded`] (the
/// encoder layers), so a speaker cache can hold pre-encode embeddings and
/// feed them back in.
///
/// [`pre_encode`]: Self::pre_encode
/// [`forward_pre_encoded`]: Self::forward_pre_encoded
///
/// Built by [`SortformerEncoderConfig`].
/// Implements [`SortformerEncoderMeta`].
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SortformerEncoder {
    /// Feature stacking, then a pre-norm transformer with rotary attention.
    Rope(RopeSortformerEncoder),
}

impl SortformerEncoderMeta for SortformerEncoder {
    fn n_mels(&self) -> usize {
        match self {
            Self::Rope(e) => e.stacking.d_input(),
        }
    }

    fn d_model(&self) -> usize {
        match self {
            Self::Rope(e) => e.encoder.d_model(),
        }
    }

    fn subsampling_factor(&self) -> usize {
        match self {
            Self::Rope(e) => e.stacking.factor(),
        }
    }
}

impl SortformerEncoder {
    /// The pre-encode embeddings of `mels`.
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
        match self {
            Self::Rope(e) => e.stacking.forward(mels, lengths),
        }
    }

    /// The encoder layers over pre-encode embeddings.
    ///
    /// # Arguments
    /// * `embs` - `[batch, frames, d_model]`.
    /// * `lengths` - each item's valid frame count; `None` means every item
    ///   fills `frames`.
    ///
    /// # Returns
    /// `[batch, frames, d_model]`.
    pub fn forward_pre_encoded(
        &self,
        embs: Tensor<3>,
        lengths: Option<&[usize]>,
    ) -> Tensor<3> {
        match self {
            Self::Rope(e) => e.encoder.forward(embs, lengths),
        }
    }
}

/// Config for [`RopeSortformerEncoder`].
#[derive(Config, Debug)]
pub struct RopeSortformerEncoderConfig {
    /// The pre-encode: mel frames stacked.
    pub stacking: FeatureStackingConfig,

    /// The encoder layers.
    pub encoder: RopeTransformerEncoderConfig,
}

impl ModuleInit<RopeSortformerEncoder> for RopeSortformerEncoderConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<RopeSortformerEncoder> {
        if self.stacking.d_output != self.encoder.d_model() {
            return Err(ConstraintError::new(
                "RopeSortformerEncoderConfig",
                "",
                Rule::Relation {
                    lhs: (
                        "stacking.d_output".into(),
                        self.stacking.d_output.to_string(),
                    ),
                    op: "==",
                    rhs: (
                        "encoder.d_model()".into(),
                        self.encoder.d_model().to_string(),
                    ),
                },
            )
            .into());
        }
        Ok(RopeSortformerEncoder {
            stacking: self.stacking.try_init(device)?,
            encoder: self.encoder.try_init(device)?,
        })
    }
}

/// Feature stacking, then a pre-norm transformer with rotary attention:
/// `Nemotron-3-Diarization`'s encoder (`NeMo`'s `TransformerEncoder` with its
/// `FeatureStacking` pre-encode).
///
/// Built by [`RopeSortformerEncoderConfig`].
#[derive(Module, Debug)]
pub struct RopeSortformerEncoder {
    /// The pre-encode.
    pub stacking: FeatureStacking,

    /// The encoder layers.
    pub encoder: RopeTransformerEncoder,
}
