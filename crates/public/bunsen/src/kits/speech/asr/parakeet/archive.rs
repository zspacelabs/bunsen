//! Phonon-2's archive.
use std::path::Path;

use burn::store::burn_pack::Tensor as PackTensor;

use super::ParakeetVocabulary;
use crate::{
    data::pretrained::fermion::{
        PhononArchive,
        read_five_value_bytes,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
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
/// [`read_five_value_bytes`](crate::data::pretrained::fermion::read_five_value_bytes);
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
        let config = std::str::from_utf8(archive.expect(PHONON_2_CONFIG)?).map_err(|e| {
            BunsenError::invalid_resource("a config that is not UTF-8").with_cause(e)
        })?;
        Ok((
            read_five_value_bytes(&container)?,
            ParakeetVocabulary::from_nemo_config(config)?,
        ))
    };
    read().map_err(|e| e.context(path.display()))
}
