//! Loading a `Nemotron3DiarizationForAudioFrameClassification` checkpoint.
use burn::store::burn_pack::Tensor as PackTensor;

use super::nemotron_diarization_rekey;
use crate::{
    burner::module::CanonicalKeys,
    data::formats::NamedTensorsApplied,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::diarization::sortformer::Sortformer,
};

/// Loads tensors named as `transformers`'
/// `Nemotron3DiarizationForAudioFrameClassification` state dict into
/// `model`: [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`nemotron_diarization_rekey`], at `model`'s front end.
///
/// # Errors
/// As [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to);
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// `model`'s encoder does not subsample by feature stacking.
pub fn load_nemotron_diarization_tensors(
    model: &mut Sortformer,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let stacking = model.encoder.subsampling.stacking().ok_or_else(|| {
        BunsenError::invalid_resource("a Nemotron encoder subsamples by feature stacking")
    })?;
    let rekey = nemotron_diarization_rekey(stacking);
    let rekey = rekey.then(model.canonical_rekey());
    rekey.apply_to(model, tensors)
}

/// The Nemotron model of `loaded`'s `transformers` checkpoint (its
/// `checkpoint`), built from `config`, in the files' precision (f32) or
/// wider as `policy` asks.
///
/// # Errors
/// When the file can't be read, as [`load_nemotron_diarization_tensors`],
/// and as [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_nemotron_diarization(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::speech::diarization::sortformer::NemotronDiarizationConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<Sortformer> {
    use crate::{
        burner::module::ModuleInit,
        kits::speech::diarization::sortformer::pretrained::CHECKPOINT,
    };
    let tensors =
        crate::data::formats::SafetensorsCheckpoint::from_loaded(loaded, CHECKPOINT)?.tensors()?;
    let mut model: Sortformer = config.try_init(device)?;
    load_nemotron_diarization_tensors(&mut model, tensors)?;
    policy.place(model, device, |_| true)
}
