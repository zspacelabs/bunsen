//! # The Hugging Face Hub, as resources
//!
//! What a client of the Hub needs to name its files: a [`HubRepo`] at a
//! revision, the URLs it serves a file and its tree listing at, and the
//! listing's entries ([`HubEntry`]), whose LFS digests pin the files. Plain
//! data and URLs: fetching is the caller's (the `hf:` provider's, through the
//! resource cache; `cargo catalog`'s, through its own client).

use alloc::{
    format,
    string::String,
};

use serde::{
    Deserialize,
    Serialize,
};

use crate::data::resources::is_sha256_hex;

/// Where the Hub is.
pub const HUB_ORIGIN: &str = "https://huggingface.co";

/// The revision a repo is read at unless another is named.
pub const HUB_MAIN: &str = "main";

/// A Hub repo at a revision: a model's, or a dataset's.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HubRepo {
    /// `org/name`.
    pub repo: String,

    /// A branch, tag or commit.
    pub revision: String,

    /// Whether the repo is a dataset's: the Hub serves those under
    /// `datasets/`.
    pub dataset: bool,
}

impl HubRepo {
    /// `repo` (`org/name`) at `revision`.
    pub fn new(
        repo: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        Self {
            repo: repo.into(),
            revision: revision.into(),
            dataset: false,
        }
    }

    /// The same repo, as a dataset's.
    pub fn dataset(mut self) -> Self {
        self.dataset = true;
        self
    }

    /// `datasets/` for a dataset's repo, nothing for a model's: where the
    /// Hub serves its files.
    fn kind_prefix(&self) -> &'static str {
        if self.dataset { "datasets/" } else { "" }
    }

    /// The URL the Hub serves `file` of the repo at.
    pub fn file_url(
        &self,
        file: &str,
    ) -> String {
        self.file_url_at(HUB_ORIGIN, file)
    }

    /// [`file_url`](Self::file_url), at a Hub served from `origin`.
    pub fn file_url_at(
        &self,
        origin: &str,
        file: &str,
    ) -> String {
        format!(
            "{origin}/{}{}/resolve/{}/{file}",
            self.kind_prefix(),
            self.repo,
            self.revision
        )
    }

    /// The URL of the repo's tree listing, recursive: every file, in every
    /// directory.
    pub fn listing_url(&self) -> String {
        format!("{}?recursive=true", self.tree_url_at(HUB_ORIGIN))
    }

    /// The URL of the repo's top-level tree listing, at a Hub served from
    /// `origin`.
    pub fn tree_url_at(
        &self,
        origin: &str,
    ) -> String {
        let api = if self.dataset { "datasets" } else { "models" };
        format!("{origin}/api/{api}/{}/tree/{}", self.repo, self.revision)
    }
}

/// One entry of a repo's tree listing, as the Hub's tree API spells it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HubEntry {
    /// `file` or `directory`.
    #[serde(rename = "type")]
    pub kind: String,

    /// The path within the repo.
    pub path: String,

    /// The size in bytes of what git stores: an LFS pointer's, for a file
    /// in LFS.
    #[serde(default)]
    pub size: u64,

    /// The LFS object, for a file stored through LFS: its `oid` is the
    /// file's SHA-256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lfs: Option<HubLfs>,
}

/// The LFS part of a listing entry.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HubLfs {
    /// The file's SHA-256 in hex, as the listing spells it (the Hub's is
    /// lowercase); [`HubEntry::sha256`] is the pin made from it.
    pub oid: String,

    /// The file's size in bytes.
    #[serde(default)]
    pub size: u64,
}

impl HubEntry {
    /// Whether the entry is a file, not a directory.
    pub fn is_file(&self) -> bool {
        self.kind == "file"
    }

    /// The digest that pins this file, when LFS holds it: its oid, in
    /// lowercase hex, the one spelling of a digest the cache's pins, paths
    /// and checks use. An oid that is not 64 hex digits pins nothing.
    pub fn sha256(&self) -> Option<String> {
        let oid = self.lfs.as_ref()?.oid.to_ascii_lowercase();
        is_sha256_hex(&oid).then_some(oid)
    }

    /// The file's size: its LFS object's, for a file in LFS, else what git
    /// stores.
    pub fn file_size(&self) -> u64 {
        self.lfs.as_ref().map_or(self.size, |l| l.size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The URLs carry the repo and the revision.
    #[test]
    fn test_urls() {
        let repo = HubRepo::new("org/name", "abc");
        assert_eq!(
            repo.file_url("a/b.json"),
            "https://huggingface.co/org/name/resolve/abc/a/b.json"
        );
        assert_eq!(
            repo.listing_url(),
            "https://huggingface.co/api/models/org/name/tree/abc?recursive=true"
        );
        assert_eq!(
            repo.tree_url_at("http://h"),
            "http://h/api/models/org/name/tree/abc"
        );
        let data = HubRepo::new("org/set", "abc").dataset();
        assert_eq!(
            data.file_url("shard_00000.parquet"),
            "https://huggingface.co/datasets/org/set/resolve/abc/shard_00000.parquet"
        );
        assert_eq!(
            data.listing_url(),
            "https://huggingface.co/api/datasets/org/set/tree/abc?recursive=true"
        );
    }

    /// An entry's pin is its LFS oid in lowercase, and its size the LFS
    /// object's.
    #[test]
    fn test_an_entry_pins_by_its_lfs_oid() {
        let entry: HubEntry = serde_json::from_str(&format!(
            r#"{{"type": "file", "path": "m.safetensors", "size": 134,
                "lfs": {{"oid": "{}", "size": 7}}}}"#,
            "AB".repeat(32)
        ))
        .unwrap();
        assert!(entry.is_file());
        assert_eq!(entry.sha256(), Some("ab".repeat(32)));
        assert_eq!(entry.file_size(), 7);
    }
}
