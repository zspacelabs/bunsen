use alloc::fmt;

use serde::{
    Deserialize,
    Serialize,
};

use crate::data::catalog::document::MergeMode;

/// A resource declaration: one artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceEntry {
    /// The resource's own segment.
    #[serde(skip)]
    pub name: String,

    /// How this occurrence meets an existing declaration of its name.
    #[serde(rename = "$merge", default, skip_serializing_if = "MergeMode::is_deny")]
    pub merge: MergeMode,

    /// What the artifact is, for a person.
    #[serde(
        rename = "$description",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub description: Option<String>,

    /// The file's path, relative to the place it is had from: a bare file
    /// name, or `/`-separated segments (`encoder/config.json`). A place
    /// source appends it; the cache keeps it whole.
    #[serde(rename = "$file")]
    pub file: String,

    /// The file's format, as its loader names it.
    #[serde(rename = "$format")]
    pub format: String,

    /// The file's SHA-256, in lowercase hex, when it is pinned.
    #[serde(rename = "$sha256", default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,

    /// The file's size in bytes, when it is known.
    #[serde(rename = "$size", default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,

    /// The dtype a checkpoint stores, when it is one; its shards store it
    /// too.
    #[serde(rename = "$dtype", default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Where to fetch it, tried in order. A place source (`$hub`,
    /// `$base_url`, `$local_dir`) serves every file of the resource, its
    /// shards included, by their `$file`; a `$url` serves this file alone.
    #[serde(rename = "$sources")]
    pub sources: Vec<ResourceSource>,

    /// For a templated resource, a numbered set of files with no index: the
    /// zero-padded width of the index its `$file` template
    /// (`shard_{index}.parquet`) writes in place of
    /// [`INDEX_PLACEHOLDER`](crate::data::datasets::INDEX_PLACEHOLDER).
    #[serde(
        rename = "$index_width",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub index_width: Option<usize>,

    /// For a templated resource, how many files there are: indices
    /// `0..count`.
    #[serde(rename = "$count", default, skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,

    /// For a templated resource, that its files are not pinned: `cargo
    /// catalog` lists no `$shards` for it.
    #[serde(
        rename = "$unpinned",
        default,
        skip_serializing_if = "core::ops::Not::not"
    )]
    pub unpinned: bool,

    /// The files a sharded resource's `$file` (its index) lists: one
    /// artifact, bound to one slot. For a templated resource
    /// ([`index_width`](Self::index_width)), the template's files, in index
    /// order, as `cargo catalog` pins them.
    #[serde(rename = "$shards", default, skip_serializing_if = "Vec::is_empty")]
    pub shards: Vec<ResourceShard>,
}

/// One shard of a sharded resource.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceShard {
    /// The shard's path, relative to the place it is had from.
    #[serde(rename = "$file")]
    pub file: String,

    /// The shard's SHA-256, in lowercase hex.
    #[serde(rename = "$sha256")]
    pub sha256: String,

    /// The shard's size in bytes.
    #[serde(rename = "$size")]
    pub size: u64,

    /// Sources of this shard alone, tried before the resource's place
    /// sources.
    #[serde(rename = "$sources", default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<ResourceSource>,
}

impl ResourceEntry {
    /// One unsharded artifact, `file` in `format`, had from `sources`: what
    /// a provider that makes its builds itself declares (an `hf:` repo's
    /// files). Unpinned, of unknown size and dtype, until set.
    pub fn new(
        file: impl Into<String>,
        format: impl Into<String>,
        sources: Vec<ResourceSource>,
    ) -> Self {
        Self {
            name: String::new(),
            merge: MergeMode::Deny,
            description: None,
            file: file.into(),
            format: format.into(),
            sha256: None,
            size: None,
            dtype: None,
            sources,
            shards: Vec::new(),
            index_width: None,
            count: None,
            unpinned: false,
        }
    }

    /// Whether this is a templated resource: a numbered set of files, its
    /// `$file` their name's template.
    pub fn is_templated(&self) -> bool {
        self.index_width.is_some()
    }

    /// The file name of index `index` of a templated resource: its `$file`
    /// with the zero-padded index in place of the placeholder.
    pub fn templated_file(
        &self,
        index: usize,
    ) -> String {
        let width = self.index_width.unwrap_or(0);
        self.file.replace(
            crate::data::datasets::INDEX_PLACEHOLDER,
            &alloc::format!("{index:0width$}"),
        )
    }

    /// The sources `shard` is had from: its own, then the resource's place
    /// sources.
    pub fn shard_sources<'a>(
        &'a self,
        shard: &'a ResourceShard,
    ) -> impl Iterator<Item = &'a ResourceSource> + 'a {
        shard
            .sources
            .iter()
            .chain(self.sources.iter().filter(|s| s.is_place()))
    }
}

/// Where a resource can be fetched.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ResourceSource {
    /// A file in a Hugging Face Hub repo, at a revision: `{$hub: {$repo,
    /// $revision}}`, a dataset repo's with `$dataset: true`.
    #[serde(rename = "$hub")]
    Hub {
        /// `org/name`.
        #[serde(rename = "$repo")]
        repo: String,
        /// A commit.
        #[serde(rename = "$revision")]
        revision: String,
        /// Whether the repo is a dataset's, not a model's.
        #[serde(
            rename = "$dataset",
            default,
            skip_serializing_if = "core::ops::Not::not"
        )]
        dataset: bool,
    },
    /// A URL: `{$url: ..}`. It names one file.
    #[serde(rename = "$url")]
    Url(String),
    /// A URL a file's `$file` is appended to: `{$base_url: ..}`.
    #[serde(rename = "$base_url")]
    BaseUrl(String),
    /// A directory another tool keeps its files in: `{$local_dir: {$name,
    /// $default}}`. A caller overrides it by `$name`; otherwise it is the
    /// first of `$default`'s path templates that resolves (`~`, `$VAR`).
    #[serde(rename = "$local_dir")]
    LocalDir {
        /// The directory's name, for overrides and messages.
        #[serde(rename = "$name")]
        name: String,
        /// Where it is when nobody overrides it: path templates, the first
        /// that resolves.
        #[serde(rename = "$default", default, skip_serializing_if = "Vec::is_empty")]
        default: Vec<String>,
    },
}

impl ResourceSource {
    /// Whether it is a place, which serves any file by its path (`$hub`,
    /// `$base_url`, `$local_dir`), rather than one file (`$url`).
    pub fn is_place(&self) -> bool {
        !matches!(self, Self::Url(_))
    }
}

impl fmt::Display for ResourceSource {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Hub {
                repo,
                revision,
                dataset,
            } => {
                let kind = if *dataset { "datasets/" } else { "" };
                write!(f, "hub:{kind}{repo}@{revision}")
            }
            Self::Url(url) => f.write_str(url),
            Self::BaseUrl(base) => write!(f, "{base}/*"),
            Self::LocalDir { name, .. } => write!(f, "local-dir:{name}"),
        }
    }
}
