//! `FermionResearch`'s Phonon archives.
use std::{
    io::Read,
    path::Path,
};

use burn::store::burn_pack::Tensor as PackTensor;

use super::{
    ParakeetVocabulary,
    read_fermion_bytes,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};

/// The container inside a Phonon archive.
pub const PHONON_CONTAINER: &str = "model.fermion";

/// The `NeMo` model config inside a Phonon archive: the vocabulary's
/// `labels`.
pub const PHONON_CONFIG: &str = "config.json";

/// Reads a Phonon archive (`phonon-2.bps.tar.zst`): a zstd-compressed tar
/// of a [`read_fermion_bytes`] container, `model.fermion`, and the `NeMo`
/// model config, `config.json`, whose `labels` are the vocabulary.
///
/// # Errors
/// An `io::Error` sorted by [`sys_at`] for a file that cannot be read;
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) for
/// one that is not zstd, not a tar, lacks either member, or whose members do
/// not parse; all under a frame naming the path.
pub fn read_phonon_archive(path: &Path) -> BunsenResult<(Vec<PackTensor>, ParakeetVocabulary)> {
    let read = || -> BunsenResult<(Vec<PackTensor>, ParakeetVocabulary)> {
        let compressed = std::fs::read(path).map_err(sys_at("read", path))?;
        let mut tar_bytes = Vec::new();
        ruzstd::decoding::StreamingDecoder::new(compressed.as_slice())
            .map_err(|e| BunsenError::invalid_resource("not a zstd stream").with_cause(e))?
            .read_to_end(&mut tar_bytes)
            .map_err(|e| BunsenError::invalid_resource("a corrupt zstd stream").with_cause(e))?;
        drop(compressed);

        let (mut container, mut config) = (None, None);
        let mut archive = tar::Archive::new(tar_bytes.as_slice());
        let entries = archive
            .entries()
            .map_err(|e| BunsenError::invalid_resource("not a tar archive").with_cause(e))?;
        for entry in entries {
            let mut entry = entry
                .map_err(|e| BunsenError::invalid_resource("a corrupt tar entry").with_cause(e))?;
            let name = entry
                .path()
                .map_err(|e| BunsenError::invalid_resource("a tar entry's path").with_cause(e))?
                .to_string_lossy()
                .into_owned();
            let slot = match name.as_str() {
                PHONON_CONTAINER => &mut container,
                PHONON_CONFIG => &mut config,
                _ => continue,
            };
            let mut bytes = Vec::new();
            entry
                .read_to_end(&mut bytes)
                .map_err(|e| BunsenError::invalid_resource("a corrupt tar entry").with_cause(e))?;
            *slot = Some(bytes);
        }

        let missing = |member: &str| {
            BunsenError::invalid_resource(format!("a Phonon archive without `{member}`"))
        };
        let container = container.ok_or_else(|| missing(PHONON_CONTAINER))?;
        let config = config.ok_or_else(|| missing(PHONON_CONFIG))?;
        let config = std::str::from_utf8(&config).map_err(|e| {
            BunsenError::invalid_resource("a config that is not UTF-8").with_cause(e)
        })?;
        Ok((
            read_fermion_bytes(&container)?,
            ParakeetVocabulary::from_nemo_config(config)?,
        ))
    };
    read().map_err(|e| e.context(path.display()))
}
