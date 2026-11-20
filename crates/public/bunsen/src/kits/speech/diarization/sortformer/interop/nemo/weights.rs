//! Loading a Sortformer `.nemo`.
use burn::store::burn_pack::Tensor as PackTensor;

use super::sortformer_v2_rekey;
use crate::{
    burner::module::CanonicalKeys,
    data::formats::NamedTensorsApplied,
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::diarization::sortformer::Sortformer,
};

/// Loads tensors named as a `NeMo` Sortformer v2 state dict (a `.nemo`'s
/// `model_weights.ckpt`) into `model`:
/// [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to) under
/// [`sortformer_v2_rekey`], at `model`'s subsampling.
///
/// # Errors
/// As [`Rekey::apply_to`](crate::burner::module::Rekey::apply_to);
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// `model`'s encoder does not subsample by depthwise striding.
pub fn load_sortformer_v2_tensors(
    model: &mut Sortformer,
    tensors: impl IntoIterator<Item = PackTensor>,
) -> BunsenResult<NamedTensorsApplied> {
    let striding = model.encoder.subsampling.dw_striding().ok_or_else(|| {
        BunsenError::invalid_resource("a Sortformer v2 encoder subsamples by depthwise striding")
    })?;
    let rekey = sortformer_v2_rekey(striding);
    let rekey = rekey.then(model.canonical_rekey());
    rekey.apply_to(model, tensors)
}

/// The Sortformer v2 of `loaded`'s `.nemo` (its `checkpoint`), and the
/// front end it was trained behind: the archive's own config
/// ([`NemoSortformerMirror`](super::NemoSortformerMirror)) gives both, and
/// its state dict loads into the model, in the file's precision (f32) or
/// wider as `policy` asks.
///
/// # Errors
/// When the archive can't be read, as the mirror's conversions (naming the
/// file), as [`load_sortformer_v2_tensors`], and as
/// [`LoadPolicy::place`](crate::data::formats::LoadPolicy::place).
#[cfg(feature = "cache")]
pub fn load_sortformer_v2(
    loaded: &crate::data::resources::LoadedResourceSet,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<(Sortformer, crate::kits::speech::nemo::NemoMelFrontEndConfig)> {
    use crate::{
        burner::module::ModuleInit,
        data::formats::nemo::NemoArchive,
        kits::speech::diarization::sortformer::pretrained::CHECKPOINT,
    };
    let path = loaded.expect(CHECKPOINT)?;
    let in_path = |e: BunsenError| e.context(path.display());
    let archive = NemoArchive::open(path)?;
    let config = super::NemoSortformerMirror::from_yaml(archive.config_yaml()).map_err(in_path)?;
    let structure = config.sortformer_structure_config().map_err(in_path)?;
    let front_end = config.mel_front_end_config().map_err(in_path)?;
    let mut model: Sortformer = structure.try_init(device)?;
    archive.with_weights(|tensors| load_sortformer_v2_tensors(&mut model, tensors))?;
    Ok((policy.place(model, device, |_| true)?, front_end))
}
