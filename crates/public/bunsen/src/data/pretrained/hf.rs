//! # Hugging Face as a provider
//!
//! `hf:{org}/{repo}` is a Hugging Face repo: `hf:openai/whisper-large-v3`
//! is <https://huggingface.co/openai/whisper-large-v3>. [`HfProvider`]
//! answers such a ref with the repo's safetensors checkpoint as a resource
//! map, and lists nothing: the hub is not enumerable, and a bare name
//! never reaches it.
//!
//! What a repo holds comes from the hub's file listing (its tree API),
//! fetched once through the cache and kept there, so a ref resolved once
//! is resolved again offline. The listing says which files there are and
//! pins each LFS file by its SHA-256, so the checkpoint is digest-addressed
//! in the cache like a well-known row's. `transformers` saves a model as
//! one `model.safetensors` until it passes a shard-size limit, then as
//! `model-00001-of-0000N.safetensors` and so on with
//! `model.safetensors.index.json` naming the shard each tensor is in; the
//! row is one resource or the index and every shard, under the kit's
//! checkpoint key (`checkpoint`, or `checkpoint.index` and
//! `checkpoint.00001`, ...), each labeled [`SAFETENSORS`] or
//! [`SAFETENSORS_INDEX`] for the kit's reader. `config.json` rides along
//! as `config` when the repo has one. Anything else in the repo (other
//! frameworks' weights, tokenizer files) is not a resource: a kit reads
//! what it knows.
//!
//! The listing is at a revision, `main` unless another is named; a repo
//! whose `main` moves is seen again only when the cache is cleared or the
//! revision is named. Resolving a ref without a cache is an error: the
//! provider guesses nothing about a repo.

use alloc::{
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

use serde::Deserialize;

#[cfg(feature = "cache")]
use super::PretrainedCache;
use super::{
    Pretrained,
    PretrainedProvider,
    Resource,
    ResourceMap,
    SAFETENSORS,
    SAFETENSORS_INDEX,
    Source,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The provider's default name: the `hf` of `hf:openai/whisper-tiny`, and
/// the namespace its files are cached under.
pub const HF: &str = "hf";

/// Where the hub is.
pub const HF_ORIGIN: &str = "https://huggingface.co";

/// The revision a ref resolves at unless another is named.
pub const HF_MAIN: &str = "main";

/// The file a `transformers` repo keeps its weights in, when they fit one.
pub const HF_SINGLE_FILE: &str = "model.safetensors";

/// The index of a sharded checkpoint: the shard each tensor is in.
pub const HF_INDEX_FILE: &str = "model.safetensors.index.json";

/// The model's config, as `transformers` writes it.
pub const HF_CONFIG_FILE: &str = "config.json";

/// The key `config.json` rides under.
pub const CONFIG: &str = "config";

/// The key the checkpoint is under unless the kit names another.
pub const DEFAULT_CHECKPOINT_KEY: &str = "checkpoint";

/// One entry of the hub's file listing, as its tree API spells it.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct HfTreeEntry {
    /// `file` or `directory`.
    #[serde(rename = "type")]
    pub kind: String,

    /// The path within the repo.
    pub path: String,

    /// The size in bytes.
    #[serde(default)]
    pub size: u64,

    /// The LFS pointer, for a file stored through LFS: its `oid` is the
    /// file's SHA-256.
    #[serde(default)]
    pub lfs: Option<HfLfs>,
}

/// The LFS part of a listing entry.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct HfLfs {
    /// The file's SHA-256, lowercase hex.
    pub oid: String,

    /// The size in bytes.
    #[serde(default)]
    pub size: u64,
}

impl HfTreeEntry {
    /// The digest that pins this file, when LFS holds it.
    pub fn sha256(&self) -> Option<&str> {
        let oid = self.lfs.as_ref()?.oid.as_str();
        (oid.len() == 64 && oid.bytes().all(|b| b.is_ascii_hexdigit())).then_some(oid)
    }
}

/// The shard number and count of `model-00001-of-00002.safetensors`.
fn shard_numbers(file: &str) -> Option<(usize, usize)> {
    let rest = file.strip_prefix("model-")?.strip_suffix(".safetensors")?;
    let (n, of) = rest.split_once("-of-")?;
    Some((n.parse().ok()?, of.parse().ok()?))
}

/// Hugging Face repos, by ref, as safetensors checkpoints.
#[derive(Clone, Debug)]
pub struct HfProvider {
    /// The provider's name, and the namespace its files are cached under.
    pub name: String,

    /// Where the hub is: [`HF_ORIGIN`], or a mirror.
    pub origin: String,

    /// The revision every ref resolves at: a branch, a tag, or a commit.
    pub revision: String,

    /// The key the checkpoint goes under: the kit's.
    pub checkpoint_key: String,
}

impl Default for HfProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl HfProvider {
    /// `hf`, at [`HF_MAIN`], the checkpoint under `checkpoint`.
    pub fn new() -> Self {
        Self {
            name: HF.to_string(),
            origin: HF_ORIGIN.to_string(),
            revision: HF_MAIN.to_string(),
            checkpoint_key: DEFAULT_CHECKPOINT_KEY.to_string(),
        }
    }

    /// Repos at a mirror of the hub, which serves the same paths.
    pub fn with_origin(
        mut self,
        origin: impl Into<String>,
    ) -> Self {
        self.origin = origin.into();
        self
    }

    /// Another name: two of these in one factory, at two revisions, say.
    pub fn with_name(
        mut self,
        name: impl Into<String>,
    ) -> Self {
        self.name = name.into();
        self
    }

    /// Repos at `revision`.
    pub fn with_revision(
        mut self,
        revision: impl Into<String>,
    ) -> Self {
        self.revision = revision.into();
        self
    }

    /// The checkpoint under `key`.
    pub fn with_checkpoint_key(
        mut self,
        key: impl Into<String>,
    ) -> Self {
        self.checkpoint_key = key.into();
        self
    }

    /// The URL `file` in `org/repo` is served at, at this provider's
    /// revision.
    pub fn url(
        &self,
        org: &str,
        repo: &str,
        file: &str,
    ) -> String {
        format!(
            "{}/{org}/{repo}/resolve/{}/{file}",
            self.origin, self.revision
        )
    }

    /// The URL of `org/repo`'s file listing at this provider's revision.
    pub fn tree_url(
        &self,
        org: &str,
        repo: &str,
    ) -> String {
        format!(
            "{}/api/models/{org}/{repo}/tree/{}",
            self.origin, self.revision
        )
    }

    /// The `org` and `repo` of a ref, which is exactly `org/repo`.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] for anything else.
    pub fn split<'a>(
        &self,
        name: &'a str,
    ) -> BunsenResult<(&'a str, &'a str)> {
        match name.split_once('/') {
            Some((org, repo)) if !org.is_empty() && !repo.is_empty() && !repo.contains('/') => {
                Ok((org, repo))
            }
            _ => Err(BunsenError::Invalid(format!(
                "{}:{name}: a Hugging Face ref is org/repo",
                self.name
            ))),
        }
    }

    /// The listing resource for `org/repo`: the tree API's JSON, unpinned,
    /// under this provider's namespace.
    pub fn listing_resource(
        &self,
        org: &str,
        repo: &str,
    ) -> Resource {
        Resource {
            key: "listing".to_string(),
            file: format!("{org}--{repo}--{}.json", self.revision.replace('/', "-")),
            sha256: None,
            kind: Some("json".to_string()),
            namespace: self.name.clone(),
            sources: vec![Source::Url(self.tree_url(org, repo))],
        }
    }

    /// One resource of the row: `file` of `org/repo`, pinned when the
    /// listing pins it.
    fn resource(
        &self,
        org: &str,
        repo: &str,
        key: &str,
        entry: &HfTreeEntry,
        kind: &str,
    ) -> Resource {
        Resource {
            key: key.to_string(),
            file: entry.path.clone(),
            sha256: entry.sha256().map(str::to_string),
            kind: Some(kind.to_string()),
            namespace: self.name.clone(),
            sources: vec![Source::Url(self.url(org, repo, &entry.path))],
        }
    }

    /// The row for `name` from the repo's file listing: the safetensors
    /// checkpoint, one file or the index and every shard, pinned by the
    /// hub's LFS digests, and `config.json` when there is one.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] when the listing has neither
    /// [`HF_SINGLE_FILE`] nor [`HF_INDEX_FILE`] with a complete set of
    /// shards, naming what it has instead.
    pub fn row_from_listing(
        &self,
        name: &str,
        listing: &[HfTreeEntry],
    ) -> BunsenResult<Pretrained> {
        let (org, repo) = self.split(name)?;
        let files: Vec<&HfTreeEntry> = listing.iter().filter(|e| e.kind == "file").collect();
        let file = |path: &str| files.iter().copied().find(|e| e.path == path);
        let origin = format!("{}/{org}/{repo}", self.origin);
        let key = self.checkpoint_key.as_str();

        let mut resources = ResourceMap::new(name);
        resources.origin = Some(origin.clone());
        let what;
        if let Some(single) = file(HF_SINGLE_FILE) {
            resources.insert(self.resource(org, repo, key, single, SAFETENSORS));
            what = format!("{HF_SINGLE_FILE}, {}", megabytes(single.size));
        } else if let Some(index) = file(HF_INDEX_FILE) {
            let mut shards: Vec<(usize, usize, &HfTreeEntry)> = files
                .iter()
                .filter_map(|e| shard_numbers(&e.path).map(|(n, of)| (n, of, *e)))
                .collect();
            shards.sort_by_key(|(n, _, _)| *n);
            let of = shards.first().map(|(_, of, _)| *of).unwrap_or(0);
            let complete = !shards.is_empty()
                && shards
                    .iter()
                    .enumerate()
                    .all(|(i, (n, o, _))| *n == i + 1 && *o == of);
            if !complete {
                return Err(BunsenError::Invalid(format!(
                    "{}:{name}: {HF_INDEX_FILE} with an incomplete set of shards: {}",
                    self.name,
                    shards
                        .iter()
                        .map(|(_, _, e)| e.path.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            resources.insert(self.resource(
                org,
                repo,
                &format!("{key}.index"),
                index,
                SAFETENSORS_INDEX,
            ));
            let mut total = 0;
            for (n, _, entry) in &shards {
                total += entry.size;
                resources.insert(self.resource(
                    org,
                    repo,
                    &format!("{key}.{n:05}"),
                    entry,
                    SAFETENSORS,
                ));
            }
            what = format!("{} shards, {}", shards.len(), megabytes(total));
        } else {
            let weights: Vec<&str> = files
                .iter()
                .map(|e| e.path.as_str())
                .filter(|p| {
                    p.ends_with(".safetensors")
                        || p.ends_with(".bin")
                        || p.ends_with(".h5")
                        || p.ends_with(".msgpack")
                        || p.ends_with(".pt")
                        || p.ends_with(".gguf")
                })
                .collect();
            return Err(BunsenError::Invalid(format!(
                "{}:{name}: no {HF_SINGLE_FILE} or {HF_INDEX_FILE} at {}; the repo has: {}",
                self.name,
                self.revision,
                if weights.is_empty() {
                    "no weights".to_string()
                } else {
                    weights.join(", ")
                }
            )));
        }
        let mut with_config = "";
        if let Some(config) = file(HF_CONFIG_FILE) {
            resources.insert(self.resource(org, repo, CONFIG, config, "json"));
            with_config = ", with config.json";
        }
        let description = format!(
            "{org}/{repo} on Hugging Face at {}: {what}{with_config}",
            self.revision
        );
        resources.description.clone_from(&description);
        Ok(Pretrained {
            name: name.to_string(),
            aliases: Vec::new(),
            description,
            license: None,
            origin: Some(origin),
            prefab: None,
            resources,
        })
    }

    /// The repo's file listing, through the cache: fetched once under
    /// `kit`, read from there after.
    #[cfg(feature = "cache")]
    fn listing(
        &self,
        name: &str,
        kit: &str,
        cache: &PretrainedCache,
    ) -> BunsenResult<Vec<HfTreeEntry>> {
        let (org, repo) = self.split(name)?;
        let resource = self.listing_resource(org, repo);
        let resolved = cache.resolve(kit, &resource).map_err(|e| match e {
            BunsenError::External(m) if m.contains("http status: 401") => BunsenError::External(
                format!(
                    "{}:{name}: {m} (a repo that is not there, or a gated one, answers 401 to an anonymous caller)",
                    self.name
                ),
            ),
            BunsenError::External(m) => BunsenError::External(format!("{}:{name}: {m}", self.name)),
            BunsenError::ResourceNotFound(m) => BunsenError::ResourceNotFound(format!(
                "{}:{name}: the repo's file listing is {m}",
                self.name
            )),
            other => other,
        })?;
        let file = std::fs::File::open(&resolved.path).map_err(BunsenError::external)?;
        serde_json::from_reader(std::io::BufReader::new(file)).map_err(|e| {
            BunsenError::Invalid(format!(
                "{}:{name}: {}: not a file listing: {e}",
                self.name,
                resolved.path.display()
            ))
        })
    }
}

impl PretrainedProvider for HfProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "Hugging Face repos by ref: `hf:org/repo` is its safetensors checkpoint at a revision, pinned by the hub's listing; lists nothing"
    }

    fn origin(&self) -> Option<&str> {
        Some(HF_ORIGIN)
    }

    /// Nothing: the hub is not enumerable.
    fn list(&self) -> Vec<Pretrained> {
        Vec::new()
    }

    /// An error: what a repo holds is the hub's to say, through a cache.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`], always.
    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<Option<Pretrained>> {
        self.split(name)?;
        Err(BunsenError::Invalid(format!(
            "{}:{name}: a Hugging Face ref is resolved through a cache, which asks the hub what the repo holds",
            self.name
        )))
    }

    /// The row for `org/repo` from its file listing, fetched once into
    /// `cache` under `kit` and read from there after.
    ///
    /// # Errors
    /// As [`split`](Self::split) and
    /// [`row_from_listing`](Self::row_from_listing); the cache's, naming
    /// the ref, for a listing that cannot be had: a repo that is not there
    /// answers 401, and an offline cache that never saw the repo has
    /// nothing.
    #[cfg(feature = "cache")]
    fn resolve(
        &self,
        name: &str,
        kit: &str,
        cache: &PretrainedCache,
    ) -> BunsenResult<Option<Pretrained>> {
        let listing = self.listing(name, kit, cache)?;
        self.row_from_listing(name, &listing).map(Some)
    }

    /// Never: a bare name means a row bunsen knows.
    fn answers_bare_names(&self) -> bool {
        false
    }
}

/// `151.1 MB` or `6.17 GB`.
fn megabytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.1} MB", bytes as f64 / 1e6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        path: &str,
        size: u64,
        sha256: Option<&str>,
    ) -> HfTreeEntry {
        HfTreeEntry {
            kind: "file".to_string(),
            path: path.to_string(),
            size,
            lfs: sha256.map(|oid| HfLfs {
                oid: oid.to_string(),
                size,
            }),
        }
    }

    const TINY_SHA: &str = "7ebd0e69e78190ff5b1b2e6c7c0a6a9b1a4e2f0f0b3d2c1a9f8e7d6c5b4a3f2e";
    const SHARD_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const SHARD_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";

    /// A single-file repo: the checkpoint pinned by its LFS digest, the
    /// config along, nothing else of the repo.
    #[test]
    fn test_a_single_file_repo_is_one_pinned_checkpoint() {
        let hf = HfProvider::new();
        let listing = vec![
            entry(".gitattributes", 1438, None),
            entry("config.json", 1983, None),
            entry("model.safetensors", 151_061_672, Some(TINY_SHA)),
            entry("pytorch_model.bin", 151_095_027, Some(SHARD_1)),
            entry("tokenizer.json", 2_480_466, None),
        ];
        let row = hf
            .row_from_listing("openai/whisper-tiny", &listing)
            .unwrap();
        assert_eq!(row.name, "openai/whisper-tiny");
        assert_eq!(hf.id(&row), "hf:openai/whisper-tiny");
        assert_eq!(
            row.origin.as_deref(),
            Some("https://huggingface.co/openai/whisper-tiny")
        );
        assert_eq!(
            row.description,
            "openai/whisper-tiny on Hugging Face at main: model.safetensors, 151.1 MB, with config.json"
        );
        assert_eq!(row.prefab, None);
        row.resources.validate().unwrap();
        assert_eq!(row.resources.keys(), ["checkpoint", "config"]);

        let checkpoint = row.resources.get("checkpoint").unwrap();
        assert_eq!(checkpoint.file, "model.safetensors");
        assert_eq!(checkpoint.sha256.as_deref(), Some(TINY_SHA));
        assert_eq!(checkpoint.kind.as_deref(), Some("safetensors"));
        assert_eq!(checkpoint.namespace, "hf");
        assert_eq!(
            checkpoint.sources,
            [Source::Url(
                "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors"
                    .to_string()
            )]
        );
        let config = row.resources.get("config").unwrap();
        assert_eq!(config.file, "config.json");
        assert_eq!(config.sha256, None);
        assert_eq!(config.kind.as_deref(), Some("json"));
    }

    /// A sharded repo: the index and every shard, in shard order, under
    /// the checkpoint key; an incomplete set is refused.
    #[test]
    fn test_a_sharded_repo_is_the_index_and_its_shards() {
        let hf = HfProvider::new().with_checkpoint_key("weights");
        let listing = vec![
            entry(
                "model-00002-of-00002.safetensors",
                1_170_000_000,
                Some(SHARD_2),
            ),
            entry(
                "model-00001-of-00002.safetensors",
                5_000_000_000,
                Some(SHARD_1),
            ),
            entry("model.safetensors.index.json", 71_000, None),
        ];
        let row = hf
            .row_from_listing("ivrit-ai/whisper-large-v3", &listing)
            .unwrap();
        assert_eq!(
            row.resources.keys(),
            ["weights.00001", "weights.00002", "weights.index"]
        );
        assert_eq!(
            row.description,
            "ivrit-ai/whisper-large-v3 on Hugging Face at main: 2 shards, 6.17 GB"
        );
        let index = row.resources.get("weights.index").unwrap();
        assert_eq!(index.file, "model.safetensors.index.json");
        assert_eq!(index.kind.as_deref(), Some("safetensors-index"));
        assert_eq!(index.sha256, None);
        let first = row.resources.get("weights.00001").unwrap();
        assert_eq!(first.file, "model-00001-of-00002.safetensors");
        assert_eq!(first.sha256.as_deref(), Some(SHARD_1));
        assert_eq!(first.kind.as_deref(), Some("safetensors"));

        let incomplete = vec![
            entry("model-00002-of-00002.safetensors", 1, Some(SHARD_2)),
            entry("model.safetensors.index.json", 71_000, None),
        ];
        let err = hf
            .row_from_listing("ivrit-ai/whisper-large-v3", &incomplete)
            .unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("incomplete set of shards")),
            "{err}"
        );
    }

    /// A repo with weights in another format only is refused naming them;
    /// a malformed ref is refused before any listing is read; a lookup
    /// without a cache is refused.
    #[test]
    fn test_what_is_refused() {
        let hf = HfProvider::new();
        let listing = vec![
            entry("pytorch_model.bin", 10, Some(SHARD_1)),
            entry("tf_model.h5", 10, Some(SHARD_2)),
            entry("README.md", 10, None),
        ];
        let err = hf.row_from_listing("some/repo", &listing).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("no model.safetensors") && m.contains("pytorch_model.bin, tf_model.h5")),
            "{err}"
        );

        for bad in [
            "whisper-tiny",
            "openai/",
            "/whisper-tiny",
            "openai/whisper/tiny",
            "",
        ] {
            let err = hf.row_from_listing(bad, &[]).unwrap_err();
            assert!(
                matches!(&err, BunsenError::Invalid(m) if m.contains("org/repo")),
                "{bad:?}: {err}"
            );
        }
        let err = hf.lookup("openai/whisper-tiny").unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("through a cache")),
            "{err}"
        );
        assert!(hf.list().is_empty());
        assert!(hf.ids().is_empty());
        assert!(!hf.answers_bare_names());
        assert_eq!(hf.origin(), Some("https://huggingface.co"));
    }

    /// The URLs: a file at a revision, and the listing.
    #[test]
    fn test_urls_carry_the_revision() {
        let hf = HfProvider::new().with_revision("06f233fe06e710322aca913c1bc4249a0d71fce1");
        assert_eq!(
            hf.url("openai", "whisper-large-v3", "model.safetensors"),
            "https://huggingface.co/openai/whisper-large-v3/resolve/06f233fe06e710322aca913c1bc4249a0d71fce1/model.safetensors"
        );
        assert_eq!(
            hf.tree_url("openai", "whisper-large-v3"),
            "https://huggingface.co/api/models/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1"
        );
        let listing = hf.listing_resource("openai", "whisper-large-v3");
        assert_eq!(
            listing.file,
            "openai--whisper-large-v3--06f233fe06e710322aca913c1bc4249a0d71fce1.json"
        );
        assert_eq!(listing.namespace, "hf");
        listing.validate().unwrap();

        let named = HfProvider::new().with_name("hf-dev").with_revision("dev");
        assert_eq!(named.name(), "hf-dev");
        assert_eq!(named.listing_resource("o", "r").namespace, "hf-dev");
    }

    /// The listing as the hub serves it: LFS files carry a digest, the
    /// rest do not.
    #[test]
    fn test_the_listing_parses() {
        let json = r#"[
          {"type":"file","oid":"abc","size":1983,"path":"config.json"},
          {"type":"file","oid":"def","size":151061672,"path":"model.safetensors",
           "lfs":{"oid":"7ebd0e69e78190ff5b1b2e6c7c0a6a9b1a4e2f0f0b3d2c1a9f8e7d6c5b4a3f2e","size":151061672,"pointerSize":135}},
          {"type":"directory","oid":"ghi","size":0,"path":"onnx"}
        ]"#;
        let listing: Vec<HfTreeEntry> = serde_json::from_str(json).unwrap();
        assert_eq!(listing.len(), 3);
        assert_eq!(listing[0].sha256(), None);
        assert_eq!(listing[1].sha256(), Some(TINY_SHA));
        assert_eq!(listing[2].kind, "directory");
        let row = HfProvider::new()
            .row_from_listing("openai/whisper-tiny", &listing)
            .unwrap();
        assert_eq!(row.resources.keys(), ["checkpoint", "config"]);
    }

    /// Through a cache, against a loopback hub: the listing is fetched
    /// once, the row is pinned by it, and an offline cache that has the
    /// listing resolves the ref again without the network; a hub that
    /// answers 401 is reported with the ref, the URL and what 401 means.
    #[cfg(all(feature = "cache", feature = "fetch"))]
    #[test]
    fn test_resolve_fetches_the_listing_once_and_keeps_it() {
        use crate::data::{
            cache::{
                BunsenDiskCacheOptions,
                testing::{
                    serve_once,
                    serve_status,
                },
            },
            pretrained::{
                CacheStatus,
                PretrainedCacheOptions,
            },
        };

        static LISTING: &[u8] = br#"[
          {"type":"file","oid":"abc","size":1983,"path":"config.json"},
          {"type":"file","oid":"def","size":3,"path":"model.safetensors",
           "lfs":{"oid":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","size":3,"pointerSize":135}}
        ]"#;
        let dir = tempfile::tempdir().unwrap();
        let cache_at = |offline: bool| {
            PretrainedCache::new(
                PretrainedCacheOptions::default()
                    .with_disk(
                        BunsenDiskCacheOptions::default()
                            .with_cache_dir(Some(dir.path().join("cache")))
                            .without_transfer_observers(),
                    )
                    .with_offline(offline),
            )
            .unwrap()
        };
        // The server answers one request, whatever the path: the listing.
        let origin_of = |url: String| url.rsplit_once('/').unwrap().0.to_string();
        let hf = HfProvider::new().with_origin(origin_of(serve_once("any", LISTING)));

        let online = cache_at(false);
        let row = hf.resolve("org/repo", "kit", &online).unwrap().unwrap();
        assert_eq!(row.name, "org/repo");
        let checkpoint = row.resources.get("checkpoint").unwrap();
        assert_eq!(
            checkpoint.sha256.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(
            checkpoint.sources,
            [Source::Url(format!(
                "{}/org/repo/resolve/main/model.safetensors",
                hf.origin
            ))]
        );
        assert_eq!(
            online.status("kit", &hf.listing_resource("org", "repo")),
            CacheStatus::Cached,
            "the listing stays in the cache"
        );

        // Offline, the same ref resolves from the cached listing; the
        // server took one request and is gone.
        let offline = cache_at(true);
        let again = hf.resolve("org/repo", "kit", &offline).unwrap().unwrap();
        assert_eq!(again, row);
        let err = hf.resolve("org/other", "kit", &offline).unwrap_err();
        assert!(
            matches!(&err, BunsenError::ResourceNotFound(m) if m.starts_with("hf:org/other: the repo's file listing is")),
            "{err}"
        );

        let gone = HfProvider::new().with_origin(origin_of(serve_status("any", 401)));
        let err = gone.resolve("nobody/nothing", "kit", &online).unwrap_err();
        assert!(
            matches!(&err, BunsenError::External(m) if m.starts_with("hf:nobody/nothing: http://") && m.contains("http status: 401") && m.contains("anonymous caller")),
            "{err}"
        );
    }
}
