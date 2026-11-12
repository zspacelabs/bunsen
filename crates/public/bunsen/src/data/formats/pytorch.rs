//! `PyTorch` checkpoints, read through the burn-store's `PyTorch` reader.

use std::path::Path;

use burn::store::{
    ModuleStore,
    PytorchStore,
    PytorchStoreError,
    burn_pack::Tensor as PackTensor,
    pytorch::PytorchError,
};

use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    LookupError,
    LookupProblem,
    io_error_kind,
};

/// Sorts a [`PytorchStoreError`] from reading the checkpoint at `path` by
/// what it means: a missing or forbidden file is a
/// [`Lookup`](BunsenErrorKind::Lookup), another failure to read it is by
/// its `io::ErrorKind`, and the rest is a file that is not the checkpoint
/// it should be, [`InvalidResource`](BunsenErrorKind::InvalidResource).
pub fn pytorch_store_error(path: &Path) -> impl FnOnce(PytorchStoreError) -> BunsenError + '_ {
    move |error| {
        let kind = match &error {
            PytorchStoreError::Io(io) | PytorchStoreError::Reader(PytorchError::Io(io)) => {
                io_error_kind(io)
            }
            _ => BunsenErrorKind::InvalidResource,
        };
        if kind == BunsenErrorKind::Lookup {
            let problem = match &error {
                PytorchStoreError::Io(io) | PytorchStoreError::Reader(PytorchError::Io(io))
                    if io.kind() == std::io::ErrorKind::PermissionDenied =>
                {
                    LookupProblem::Denied
                }
                _ => LookupProblem::Missing,
            };
            BunsenError::lookup(LookupError::path(path, problem).with_source(error))
        } else {
            BunsenError::from_cause(kind, error).context(path.display())
        }
    }
}

/// Every tensor of the `PyTorch` state dict at `path`, under its own name:
/// a stack's indices as the checkpoint numbers them, never renumbered. Each
/// tensor keeps the dtype the file stores it in.
///
/// # Errors
/// As [`pytorch_store_error`] sorts them: a missing or forbidden file is a
/// [`Lookup`](BunsenErrorKind::Lookup); a file that is not a `PyTorch`
/// checkpoint, an [`InvalidResource`](BunsenErrorKind::InvalidResource) with
/// the [`PytorchStoreError`] as its cause.
pub fn read_pytorch_tensors(path: &Path) -> BunsenResult<Vec<PackTensor>> {
    Ok(PytorchStore::from_file(path.to_path_buf())
        .map_indices_contiguous(false)
        .get_all_tensors()
        .map_err(pytorch_store_error(path))?
        .values()
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::testing::ErrorMatcher;

    #[test]
    fn test_missing_checkpoint_is_a_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_pytorch_tensors(&dir.path().join("absent.pt")).unwrap_err();
        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert(&err);
    }

    #[test]
    fn test_not_a_checkpoint_is_an_invalid_resource() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.pt");
        std::fs::write(&path, b"not a zip, not a pickle").unwrap();
        let err = read_pytorch_tensors(&path).unwrap_err();
        ErrorMatcher::kind(BunsenErrorKind::InvalidResource).assert(&err);
    }
}
