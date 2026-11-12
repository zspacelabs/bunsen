//! What the builder reads from the network: Hub listings, small files, sizes
//! and digests.

use std::{
    collections::BTreeMap,
    io::Read,
};

use serde::Deserialize;
use sha2::{
    Digest,
    Sha256,
};

/// What the builder reads from the network.
pub trait Remote {
    /// The bytes at `url`.
    fn get(
        &self,
        url: &str,
    ) -> Result<Vec<u8>, String> {
        Ok(self.get_page(url)?.0)
    }

    /// The bytes at `url`, and the URL of the next page when the response
    /// links one (`Link: <…>; rel="next"`).
    fn get_page(
        &self,
        url: &str,
    ) -> Result<(Vec<u8>, Option<String>), String>;

    /// The size of what is at `url`, from a `HEAD`'s `Content-Length`.
    fn head_size(
        &self,
        url: &str,
    ) -> Result<Option<u64>, String>;

    /// The SHA-256 of what is at `url`, streamed.
    fn sha256(
        &self,
        url: &str,
    ) -> Result<String, String>;
}

/// The network, through `ureq`.
pub struct Http(ureq::Agent);

impl Http {
    /// A client with `ureq`'s defaults.
    pub fn new() -> Self {
        Self(ureq::Agent::new_with_defaults())
    }
}

impl Default for Http {
    fn default() -> Self {
        Self::new()
    }
}

/// The `rel="next"` target of a `Link` header.
pub fn next_link(link: &str) -> Option<String> {
    link.split(',').find_map(|part| {
        let (target, params) = part.split_once(';')?;
        params.contains("rel=\"next\"").then(|| {
            target
                .trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_string()
        })
    })
}

impl Remote for Http {
    fn get_page(
        &self,
        url: &str,
    ) -> Result<(Vec<u8>, Option<String>), String> {
        let mut response = self
            .0
            .get(url)
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?;
        let next = response
            .headers()
            .get("link")
            .and_then(|l| l.to_str().ok())
            .and_then(next_link);
        let body = response
            .body_mut()
            .with_config()
            .limit(64 << 20)
            .read_to_vec()
            .map_err(|e| format!("GET {url}: {e}"))?;
        Ok((body, next))
    }

    fn head_size(
        &self,
        url: &str,
    ) -> Result<Option<u64>, String> {
        // A compressed response's length is not the file's.
        let response = self
            .0
            .head(url)
            .header("Accept-Encoding", "identity")
            .call()
            .map_err(|e| format!("HEAD {url}: {e}"))?;
        Ok(response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok()))
    }

    fn sha256(
        &self,
        url: &str,
    ) -> Result<String, String> {
        let mut response = self
            .0
            .get(url)
            .call()
            .map_err(|e| format!("GET {url}: {e}"))?;
        let mut reader = response.body_mut().as_reader();
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| format!("GET {url}: {e}"))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(hex(&hasher.finalize()))
    }
}

/// `bytes` as lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The SHA-256 of `bytes`, as lowercase hex.
pub fn sha256_of(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// A Hugging Face Hub repo at a revision.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct HubRepo {
    /// `org/name`.
    pub repo: String,

    /// A commit.
    pub revision: String,
}

impl HubRepo {
    /// The URL `file` is downloaded from.
    pub fn file_url(
        &self,
        file: &str,
    ) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{file}",
            self.repo, self.revision
        )
    }

    /// The URL of the repo's recursive tree listing.
    pub fn listing_url(&self) -> String {
        format!(
            "https://huggingface.co/api/models/{}/tree/{}?recursive=true",
            self.repo, self.revision
        )
    }
}

/// One file in a Hub repo's tree listing.
#[derive(Clone, Debug, Deserialize)]
pub struct HubFile {
    /// `file` or `directory`.
    #[serde(rename = "type")]
    pub kind: String,

    /// The path in the repo.
    pub path: String,

    /// The size of what git stores: an LFS pointer's, for an LFS file.
    #[serde(default)]
    pub size: u64,

    /// The LFS object, for a file stored in LFS.
    #[serde(default)]
    pub lfs: Option<HubLfs>,
}

/// An LFS file's object: its digest and size.
#[derive(Clone, Debug, Deserialize)]
pub struct HubLfs {
    /// The SHA-256 of the file.
    pub oid: String,

    /// The file's size.
    pub size: u64,
}

/// The files of `repo`, by path, across the listing's pages.
///
/// # Errors
/// A page that can't be fetched, or doesn't read as a listing.
pub fn hub_listing(
    remote: &dyn Remote,
    repo: &HubRepo,
) -> Result<BTreeMap<String, HubFile>, String> {
    let mut url = Some(repo.listing_url());
    let mut files = BTreeMap::new();
    while let Some(page) = url {
        let (body, next) = remote.get_page(&page)?;
        let entries: Vec<HubFile> = serde_json::from_slice(&body)
            .map_err(|e| format!("{page}: not a tree listing: {e}"))?;
        for entry in entries {
            if entry.kind == "file" {
                files.insert(entry.path.clone(), entry);
            }
        }
        url = next;
    }
    Ok(files)
}

/// The dtype a `config.json` names: `torch_dtype`, `dtype`, or either under
/// `text_config`.
pub fn config_dtype(config: &[u8]) -> Option<String> {
    let json: serde_json::Value = serde_json::from_slice(config).ok()?;
    let find = |v: &serde_json::Value| {
        ["torch_dtype", "dtype"]
            .iter()
            .find_map(|k| v.get(k).and_then(|d| d.as_str()).map(str::to_string))
    };
    find(&json).or_else(|| json.get("text_config").and_then(find))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_link() {
        assert_eq!(
            next_link(r#"<https://h/x?cursor=2>; rel="next""#).as_deref(),
            Some("https://h/x?cursor=2")
        );
        assert_eq!(next_link(r#"<https://h/x>; rel="prev""#), None);
    }

    #[test]
    fn test_config_dtype() {
        assert_eq!(
            config_dtype(br#"{"dtype":"float32"}"#).as_deref(),
            Some("float32")
        );
        assert_eq!(
            config_dtype(br#"{"text_config":{"torch_dtype":"bfloat16"}}"#).as_deref(),
            Some("bfloat16")
        );
        assert_eq!(config_dtype(b"{}"), None);
    }
}

/// A network of fixed answers, for tests.
#[cfg(test)]
pub mod testing {
    use std::{
        cell::Cell,
        collections::BTreeMap,
    };

    use super::*;

    /// Fixed bodies and sizes by URL; it counts the requests it answers.
    #[derive(Default)]
    pub struct FakeRemote {
        /// What a `GET` of each URL returns.
        pub bodies: BTreeMap<String, Vec<u8>>,

        /// What a `HEAD` of each URL says the size is.
        pub sizes: BTreeMap<String, u64>,

        /// The requests answered.
        pub requests: Cell<usize>,
    }

    impl FakeRemote {
        /// A Hub repo whose files are `small` (fetched) and `lfs` (listed
        /// with their digest and size).
        pub fn with_repo(
            mut self,
            repo: &HubRepo,
            small: &[(&str, &[u8])],
            lfs: &[(&str, &[u8])],
        ) -> Self {
            let mut listing = Vec::new();
            for (path, bytes) in small {
                listing
                    .push(serde_json::json!({"type": "file", "path": path, "size": bytes.len()}));
                self.bodies.insert(repo.file_url(path), bytes.to_vec());
            }
            for (path, bytes) in lfs {
                listing.push(serde_json::json!({
                    "type": "file", "path": path, "size": 1,
                    "lfs": {"oid": sha256_of(bytes), "size": bytes.len() * 1000}
                }));
                self.bodies.insert(repo.file_url(path), bytes.to_vec());
            }
            listing.push(serde_json::json!({"type": "directory", "path": "sub"}));
            self.bodies
                .insert(repo.listing_url(), serde_json::to_vec(&listing).unwrap());
            self
        }
    }

    impl Remote for FakeRemote {
        fn get_page(
            &self,
            url: &str,
        ) -> Result<(Vec<u8>, Option<String>), String> {
            self.requests.set(self.requests.get() + 1);
            let body = self
                .bodies
                .get(url)
                .cloned()
                .ok_or_else(|| format!("no {url}"))?;
            Ok((body, None))
        }

        fn head_size(
            &self,
            url: &str,
        ) -> Result<Option<u64>, String> {
            self.requests.set(self.requests.get() + 1);
            Ok(self.sizes.get(url).copied())
        }

        fn sha256(
            &self,
            url: &str,
        ) -> Result<String, String> {
            Ok(sha256_of(&self.get(url)?))
        }
    }
}
