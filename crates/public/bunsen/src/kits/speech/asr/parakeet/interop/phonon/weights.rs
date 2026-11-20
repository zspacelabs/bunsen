//! Reading a `FermionResearch` Phonon-2 archive.
use std::path::Path;

use burn::store::burn_pack::Tensor as PackTensor;

use crate::{
    data::formats::fermion::{
        PhononArchive,
        read_five_value_bytes,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::asr::parakeet::ParakeetVocabulary,
};

/// The container inside Phonon-2's archive.
pub const PHONON_2_CONTAINER: &str = "model.fermion";

/// The `NeMo` model config inside Phonon-2's archive: the vocabulary's
/// `labels`.
pub const PHONON_2_CONFIG: &str = "config.json";

/// Reads Phonon-2's archive (`phonon-2.bps.tar.zst`), a [`PhononArchive`] of
/// a five-value container, `model.fermion`, and the `NeMo` model config,
/// `config.json`, whose `labels` are the vocabulary.
///
/// # Errors
/// As [`PhononArchive::read`] and
/// [`read_five_value_bytes`](crate::data::formats::fermion::read_five_value_bytes);
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for an
/// archive without either member or a config that does not parse; all under
/// a frame naming the path.
pub fn read_phonon2_archive(path: &Path) -> BunsenResult<(Vec<PackTensor>, ParakeetVocabulary)> {
    let read = || -> BunsenResult<(Vec<PackTensor>, ParakeetVocabulary)> {
        let mut archive = PhononArchive::read(path)?;
        let container = archive.take(PHONON_2_CONTAINER).ok_or_else(|| {
            BunsenError::invalid_resource(format!(
                "a Phonon-2 archive without `{PHONON_2_CONTAINER}`"
            ))
        })?;
        let config = core::str::from_utf8(archive.expect(PHONON_2_CONFIG)?).map_err(|e| {
            BunsenError::invalid_resource("a config that is not UTF-8").with_cause(e)
        })?;
        Ok((
            read_five_value_bytes(&container)?,
            ParakeetVocabulary::from_nemo_config(config)?,
        ))
    };
    read().map_err(|e| e.context(path.display()))
}

/// The model of `loaded`'s Phonon-2 archive (its `checkpoint`), built from
/// `config`, and the vocabulary the archive carries: the weights in the
/// files' precision (f32), or wider as `policy` asks.
///
/// # Errors
/// As [`read_phonon2_archive`], and as the `transformers` loader.
#[cfg(feature = "cache")]
pub fn load_parakeet_tdt(
    loaded: &crate::data::resources::LoadedResourceSet,
    config: &crate::kits::speech::asr::parakeet::ParakeetTdtConfig,
    device: &burn::tensor::Device,
    policy: crate::data::formats::LoadPolicy,
) -> BunsenResult<(
    crate::kits::speech::asr::parakeet::ParakeetTdt,
    ParakeetVocabulary,
)> {
    use crate::kits::speech::asr::parakeet::{
        interop::transformers::build_parakeet_tdt,
        pretrained::CHECKPOINT,
    };
    let (tensors, vocab) = read_phonon2_archive(loaded.expect(CHECKPOINT)?)?;
    Ok((build_parakeet_tdt(tensors, config, device, policy)?, vocab))
}
