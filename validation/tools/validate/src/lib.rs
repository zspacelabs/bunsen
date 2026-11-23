//! `cargo validate`'s reading of the validation workspace, shared with the
//! tools beside it (`cargo timing`): the schedule (`validation/models.yml`,
//! [`schedule`]) and the profiles that choose from it (`validation/profiles/`,
//! [`profile`]).

use std::path::PathBuf;

pub mod profile;
pub mod schedule;

/// The validation workspace: `validation/`.
pub fn validation_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
