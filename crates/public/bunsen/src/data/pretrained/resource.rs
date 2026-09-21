//! # Resources
//!
//! One file of a [`ResourceMap`](super::ResourceMap): what its map calls it,
//! the file it lands as, the digest that pins it (when one does), a label for
//! a listing, and where its bytes can be had from. A static twin for
//! compiled-in tables, an owned twin for everything else.
//!
//! A resource's sources are its own first, then its map's bases with its file
//! name appended, each in the order listed. A directory another tool keeps
//! the file in is a "trust me" source, used in place; a URL is fetched into
//! the cache; bytes linked into the binary are written into the cache on
//! first use. The owned twin carries them all, so it stands alone, and a
//! fused map needs no memory of which map a resource came from.
//!
//! [`StaticSource`] and [`Source`] are one place a file can be had from;
//! [`StaticBase`] is a place a map's file names are appended to.

use alloc::{
    format,
    string::{
        String,
        ToString,
    },
    vec,
    vec::Vec,
};
use core::fmt;
use std::path::{
    Path,
    PathBuf,
};

use serde::{
    Deserialize,
    Serialize,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The namespace of a resource that arrived as a path rather than a row.
pub const GIVEN_NAMESPACE: &str = "given";

const X25: crc::Crc<u16> = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);

/// A cache key (a bare directory name) from a name and a URL: the URL's
/// checksum and its base name, so that two unpinned files with one name
/// from two places do not share a slot.
pub fn url_to_cache_key(
    name: Option<&str>,
    url: &str,
) -> String {
    let hash = X25.checksum(url.as_bytes()).to_string();
    let base_name = url.rsplit_once('/').map_or(url, |(_, name)| name);
    match name {
        Some(n) => format!("{}-{}-{}", n, hash, base_name),
        None => format!("{}-{}", hash, base_name),
    }
}

/// The label on a safetensors weights file: a whole checkpoint, or one
/// shard of a checkpoint split across several.
pub const SAFETENSORS: &str = "safetensors";

/// The label on the index of a sharded safetensors checkpoint
/// (`model.safetensors.index.json`): the shard each tensor is in.
pub const SAFETENSORS_INDEX: &str = "safetensors-index";

/// One place a file can be had from, as a compiled-in table spells it.
///
/// Sources are tried in the order listed: a directory another tool keeps
/// the file in is used in place, a URL is fetched into the cache, and
/// bytes linked into the binary are written into the cache on first use.
/// A bundle laid out on disk is a local directory; one compiled in is
/// [`Bundled`](Self::Bundled).
#[derive(Clone, Copy)]
pub enum StaticSource<'a> {
    /// A URL, fetched into the cache.
    Url(&'a str),

    /// A directory another tool keeps this file in, under its bare name:
    /// `openai-whisper`'s download root, a hub cache. `name` is what a caller
    /// overrides the directory by; `default` is where it is when nobody does.
    LocalDir {
        /// The source's name, for overrides and messages.
        name: &'a str,
        /// Where the directory is by default, when it can be resolved.
        default: fn() -> Option<PathBuf>,
    },

    /// The file's bytes, linked into the binary: `include_bytes!` in a
    /// bundle crate. Written into the cache under the resource's digest on
    /// first use, so the resource must be pinned.
    Bundled(&'static [u8]),
}

impl StaticSource<'_> {
    /// The owned twin, with the default directory resolved now.
    pub fn to_source(&self) -> Source {
        match self {
            Self::Url(url) => Source::Url(url.to_string()),
            Self::LocalDir { name, default } => Source::LocalDir {
                name: name.to_string(),
                dir: default(),
            },
            Self::Bundled(bytes) => Source::Bundled(BundledBytes(bytes)),
        }
    }
}

impl fmt::Debug for StaticSource<'_> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Url(url) => f.debug_tuple("Url").field(url).finish(),
            Self::LocalDir { name, .. } => f.debug_struct("LocalDir").field("name", name).finish(),
            Self::Bundled(bytes) => write!(f, "Bundled({} bytes)", bytes.len()),
        }
    }
}

impl fmt::Display for StaticSource<'_> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Url(url) => write!(f, "url {url}"),
            Self::LocalDir { name, .. } => write!(f, "local dir {name}"),
            Self::Bundled(bytes) => write!(f, "bundled ({} bytes)", bytes.len()),
        }
    }
}

/// Bytes linked into the binary: the payload of a bundled source.
///
/// No manifest form: serializing one is an error, and none can be read
/// back, since the bytes are the build's.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BundledBytes(pub &'static [u8]);

impl BundledBytes {
    /// How many bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// `true` for no bytes at all.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for BundledBytes {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "BundledBytes({} bytes)", self.len())
    }
}

impl Serialize for BundledBytes {
    fn serialize<S: serde::Serializer>(
        &self,
        _serializer: S,
    ) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "bundled bytes have no manifest form",
        ))
    }
}

impl<'de> Deserialize<'de> for BundledBytes {
    fn deserialize<D: serde::Deserializer<'de>>(_deserializer: D) -> Result<Self, D::Error> {
        Err(serde::de::Error::custom(
            "bundled bytes have no manifest form",
        ))
    }
}

/// One place a file can be had from.
///
/// The owned twin of [`StaticSource`]: a local directory carries the
/// directory it resolved to. A [`Bundled`](Self::Bundled) source has no
/// manifest form: serializing one is an error, and none can be read back.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Source {
    /// A URL, fetched into the cache.
    Url(String),

    /// A directory another tool keeps this file in, under its bare name.
    LocalDir {
        /// The source's name, for overrides and messages.
        name: String,
        /// The directory, when it could be resolved.
        dir: Option<PathBuf>,
    },

    /// The file's bytes, linked into the binary; written into the cache
    /// under the resource's digest on first use.
    Bundled(BundledBytes),
}

impl fmt::Debug for Source {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Url(url) => f.debug_tuple("Url").field(url).finish(),
            Self::LocalDir { name, dir } => f
                .debug_struct("LocalDir")
                .field("name", name)
                .field("dir", dir)
                .finish(),
            Self::Bundled(bytes) => write!(f, "Bundled({} bytes)", bytes.len()),
        }
    }
}

impl fmt::Display for Source {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Url(url) => write!(f, "url {url}"),
            Self::LocalDir {
                name,
                dir: Some(dir),
            } => write!(f, "local dir {name} ({})", dir.display()),
            Self::LocalDir { name, dir: None } => write!(f, "local dir {name} (unresolved)"),
            Self::Bundled(bytes) => write!(f, "bundled ({} bytes)", bytes.len()),
        }
    }
}

/// A place a map's file names are appended to, as a compiled-in table spells
/// it.
///
/// A map lists its bases in preference order, as mirrors. Each becomes one
/// [`Source`] per resource once the file name is appended:
/// [`Url`](Self::Url) a whole URL, [`LocalDir`](Self::LocalDir) the
/// directory, which the cache joins the file name to when it looks.
#[derive(Clone, Copy, Debug)]
pub enum StaticBase<'a> {
    /// A URL prefix: the file is at `<url>/<file>`.
    Url(&'a str),

    /// A directory another tool keeps the files in, under their bare names.
    /// `name` is what a caller overrides the directory by; `default` is
    /// where it is when nobody does.
    LocalDir {
        /// The base's name, for overrides and messages.
        name: &'a str,
        /// Where the directory is by default, when it can be resolved.
        default: fn() -> Option<PathBuf>,
    },
}

impl StaticBase<'_> {
    /// The source `file` has under this base.
    pub fn source_for(
        &self,
        file: &str,
    ) -> Source {
        match self {
            Self::Url(url) => Source::Url(format!("{}/{file}", url.trim_end_matches('/'))),
            Self::LocalDir { name, default } => Source::LocalDir {
                name: name.to_string(),
                dir: default(),
            },
        }
    }
}

impl fmt::Display for StaticBase<'_> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Url(url) => write!(f, "url {url}"),
            Self::LocalDir { name, .. } => write!(f, "local dir {name}"),
        }
    }
}

/// One file of a map, as a compiled-in table spells it.
#[derive(Debug)]
pub struct StaticResource<'a> {
    /// The map's name for it: `checkpoint`, `vocabulary`. Opaque here; a
    /// kit's constant.
    pub key: &'a str,

    /// The file name, under every base.
    pub file: &'a str,

    /// Lowercase hex SHA-256 of the file, which pins every source but a
    /// bundled one; `None` leaves the file unpinned.
    pub sha256: Option<&'a str>,

    /// A label for a listing: `"pytorch fp16"`, `"tiktoken"`. Nothing in
    /// this layer reads it; a kit's hook may.
    pub kind: Option<&'a str>,

    /// Sources of this resource alone, tried before the map's bases: a file
    /// a whole URL, or a directory of its own.
    pub sources: &'a [StaticSource<'a>],
}

impl StaticResource<'_> {
    /// The owned twin, under `namespace`, with `bases` appended after its
    /// own sources.
    pub fn to_resource(
        &self,
        namespace: &str,
        bases: &[StaticBase<'_>],
    ) -> Resource {
        Resource {
            key: self.key.to_string(),
            file: self.file.to_string(),
            sha256: self.sha256.map(str::to_string),
            kind: self.kind.map(str::to_string),
            namespace: namespace.to_string(),
            sources: self
                .sources
                .iter()
                .map(StaticSource::to_source)
                .chain(bases.iter().map(|b| b.source_for(self.file)))
                .collect(),
        }
    }
}

/// One file of a map.
///
/// The owned twin of [`StaticResource`], with its map's bases already
/// appended to its sources, so it stands alone.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resource {
    /// The map's name for it. Opaque here; a kit's constant.
    pub key: String,

    /// The file name, under every base.
    pub file: String,

    /// Lowercase hex SHA-256 of the file, which pins every source but a
    /// bundled one; `None` leaves the file unpinned.
    pub sha256: Option<String>,

    /// A label for a listing. Nothing in this layer reads it; a kit's hook
    /// may, and a resource that arrived without a row has none.
    pub kind: Option<String>,

    /// The cache segment its file lives under: who published it.
    pub namespace: String,

    /// Where the file can be had from, in preference order, its map's bases
    /// already appended.
    pub sources: Vec<Source>,
}

impl Resource {
    /// A resource for a file on disk already: its directory, as a local-dir
    /// source named [`GIVEN_NAMESPACE`], and its file name. What a path on a
    /// command line becomes: used in place, unpinned, unlabeled.
    pub fn given(
        key: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        let path = path.into();
        let file = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let dir = path.parent().map(Path::to_path_buf);
        Self {
            key: key.into(),
            file,
            sha256: None,
            kind: None,
            namespace: GIVEN_NAMESPACE.to_string(),
            sources: vec![Source::LocalDir {
                name: GIVEN_NAMESPACE.to_string(),
                dir,
            }],
        }
    }

    /// `true` when the file is pinned to a digest.
    pub fn is_pinned(&self) -> bool {
        self.sha256.is_some()
    }

    /// The URL sources, in order.
    pub fn urls(&self) -> Vec<&str> {
        self.sources
            .iter()
            .filter_map(|s| match s {
                Source::Url(url) => Some(url.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The bytes linked into the binary, when a source is bundled.
    pub fn bundled(&self) -> Option<&'static [u8]> {
        self.sources.iter().find_map(|s| match s {
            Source::Bundled(bytes) => Some(bytes.0),
            _ => None,
        })
    }

    /// Checks the resource hangs together: a key, a file name, a namespace,
    /// at least one source, a digest, if any, of 64 lowercase hex digits,
    /// and a digest for certain when a source is bundled, since an
    /// unpinned file is cached under a name key and a rebuilt bundle would
    /// be shadowed by a stale cached one.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] naming the first problem.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.key.is_empty() {
            return Err(BunsenError::Invalid(format!(
                "{}: a resource has no key",
                self.file
            )));
        }
        if self.file.is_empty() {
            return Err(BunsenError::Invalid(format!("{}: no file name", self.key)));
        }
        if self.namespace.is_empty() {
            return Err(BunsenError::Invalid(format!("{}: no namespace", self.key)));
        }
        if self.sources.is_empty() {
            return Err(BunsenError::Invalid(format!("{}: no source", self.key)));
        }
        if let Some(sha256) = &self.sha256
            && !is_sha256_hex(sha256)
        {
            return Err(BunsenError::Invalid(format!(
                "{}: sha256 {sha256:?} is not 64 lowercase hex digits",
                self.key
            )));
        }
        if self.bundled().is_some() && self.sha256.is_none() {
            return Err(BunsenError::Invalid(format!(
                "{}: a bundled source needs a digest to be cached under",
                self.key
            )));
        }
        Ok(())
    }

    /// The cache directory of an unpinned resource: keyed by its first URL,
    /// as the older unpinned layout is, or by its file name when it has
    /// none.
    pub fn cache_key(&self) -> String {
        match self.urls().first() {
            Some(url) => url_to_cache_key(None, url),
            None => self.file.clone(),
        }
    }
}

impl fmt::Display for Resource {
    /// `checkpoint: tiny.en.pt (pytorch fp16, pinned)`.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}: {}", self.key, self.file)?;
        let pin = if self.is_pinned() {
            "pinned"
        } else {
            "unpinned"
        };
        match &self.kind {
            Some(kind) => write!(f, " ({kind}, {pin})"),
            None => write!(f, " ({pin})"),
        }
    }
}

/// `true` for 64 lowercase hex digits.
pub(crate) fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upstream_dir_for_display() -> Option<PathBuf> {
        Some(PathBuf::from("/home/someone/.cache/upstream"))
    }

    /// Sources print as a listing shows them, the owned local dir with the
    /// directory it resolved to.
    #[test]
    fn test_source_display() {
        let sources = [
            StaticSource::LocalDir {
                name: "upstream",
                default: upstream_dir_for_display,
            },
            StaticSource::Url("https://a.example/my_model.pt"),
        ];
        let shown: Vec<String> = sources.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            shown,
            ["local dir upstream", "url https://a.example/my_model.pt"]
        );
        assert_eq!(
            sources[0].to_source().to_string(),
            "local dir upstream (/home/someone/.cache/upstream)"
        );
        assert_eq!(
            sources[1].to_source(),
            Source::Url("https://a.example/my_model.pt".to_string())
        );
        assert_eq!(
            Source::LocalDir {
                name: "x".to_string(),
                dir: None
            }
            .to_string(),
            "local dir x (unresolved)"
        );
        let bundled = StaticSource::Bundled(b"abc");
        assert_eq!(bundled.to_string(), "bundled (3 bytes)");
        assert_eq!(format!("{bundled:?}"), "Bundled(3 bytes)");
        assert_eq!(bundled.to_source(), Source::Bundled(BundledBytes(b"abc")));
        assert_eq!(format!("{:?}", bundled.to_source()), "Bundled(3 bytes)");
        assert!(
            serde_json::to_string(&Source::Bundled(BundledBytes(b"abc"))).is_err(),
            "in-binary bytes have no manifest form"
        );
        let bundled = StaticSource::Bundled(b"abc");
        assert_eq!(bundled.to_string(), "bundled (3 bytes)");
        assert_eq!(format!("{bundled:?}"), "Bundled(3 bytes)");
        assert_eq!(bundled.to_source(), Source::Bundled(BundledBytes(b"abc")));
        assert_eq!(format!("{:?}", bundled.to_source()), "Bundled(3 bytes)");
        assert!(
            serde_json::to_string(&Source::Bundled(BundledBytes(b"abc"))).is_err(),
            "in-binary bytes have no manifest form"
        );
        assert_eq!(
            url_to_cache_key(Some("m"), "https://a.example/m.pt"),
            format!("m-{}-m.pt", X25.checksum(b"https://a.example/m.pt"))
        );
    }

    /// A bundled source is cached under the digest, so it needs one.
    #[test]
    fn test_a_bundled_source_needs_a_digest() {
        let mut r = Resource {
            key: "burnpack".to_string(),
            file: "model.bpk".to_string(),
            sha256: None,
            kind: None,
            namespace: "bunsen".to_string(),
            sources: vec![Source::Bundled(BundledBytes(b"abc"))],
        };
        assert_eq!(r.bundled(), Some(&b"abc"[..]));
        assert!(r.urls().is_empty());
        let err = r.validate().unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("bundled source needs a digest")),
            "{err}"
        );
        r.sha256 =
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string());
        r.validate().unwrap();
        assert!(r.is_pinned());
    }

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn upstream_dir() -> Option<PathBuf> {
        Some(PathBuf::from("/home/someone/.cache/whisper"))
    }

    fn nowhere() -> Option<PathBuf> {
        None
    }

    static BASES: [StaticBase<'static>; 2] = [
        StaticBase::LocalDir {
            name: "upstream",
            default: upstream_dir,
        },
        StaticBase::Url("https://a.example/models/"),
    ];

    static CHECKPOINT: StaticResource<'static> = StaticResource {
        key: "checkpoint",
        file: "tiny.en.pt",
        sha256: Some(ABC_SHA256),
        kind: Some("pytorch fp16"),
        sources: &[StaticSource::Url("https://mirror.example/tiny.en.pt")],
    };

    /// A URL base loses its trailing slash before the file is appended; a
    /// local-dir base resolves its default now.
    #[test]
    fn test_base_source_for() {
        assert_eq!(
            BASES[1].source_for("tiny.en.pt"),
            Source::Url("https://a.example/models/tiny.en.pt".to_string())
        );
        assert_eq!(
            StaticBase::Url("https://a.example/models").source_for("tiny.en.pt"),
            Source::Url("https://a.example/models/tiny.en.pt".to_string())
        );
        assert_eq!(
            BASES[0].source_for("tiny.en.pt"),
            Source::LocalDir {
                name: "upstream".to_string(),
                dir: upstream_dir(),
            }
        );
        assert_eq!(
            StaticBase::LocalDir {
                name: "gone",
                default: nowhere,
            }
            .source_for("x"),
            Source::LocalDir {
                name: "gone".to_string(),
                dir: None,
            }
        );
        assert_eq!(BASES[0].to_string(), "local dir upstream");
        assert_eq!(BASES[1].to_string(), "url https://a.example/models/");
    }

    /// The owned twin carries the resource's own sources first, then the
    /// bases in order, under the namespace it was given.
    #[test]
    fn test_to_resource_orders_sources() {
        let r = CHECKPOINT.to_resource("openai", &BASES);
        assert_eq!(r.key, "checkpoint");
        assert_eq!(r.file, "tiny.en.pt");
        assert_eq!(r.sha256.as_deref(), Some(ABC_SHA256));
        assert_eq!(r.kind.as_deref(), Some("pytorch fp16"));
        assert_eq!(r.namespace, "openai");
        assert_eq!(
            r.sources,
            vec![
                Source::Url("https://mirror.example/tiny.en.pt".to_string()),
                Source::LocalDir {
                    name: "upstream".to_string(),
                    dir: upstream_dir(),
                },
                Source::Url("https://a.example/models/tiny.en.pt".to_string()),
            ]
        );
        assert!(r.is_pinned());
        assert_eq!(
            r.urls(),
            vec![
                "https://mirror.example/tiny.en.pt",
                "https://a.example/models/tiny.en.pt"
            ]
        );
        assert_eq!(
            r.cache_key(),
            url_to_cache_key(None, "https://mirror.example/tiny.en.pt")
        );
        assert_eq!(
            r.to_string(),
            "checkpoint: tiny.en.pt (pytorch fp16, pinned)"
        );
        r.validate().unwrap();

        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<Resource>(&json).unwrap(), r);
    }

    /// A given path is one local-dir source, its parent, with its file
    /// name; unpinned and unlabeled, under the `given` namespace.
    #[test]
    fn test_given() {
        let r = Resource::given("checkpoint", "/models/my.pt");
        assert_eq!(r.key, "checkpoint");
        assert_eq!(r.file, "my.pt");
        assert_eq!(r.namespace, GIVEN_NAMESPACE);
        assert!(!r.is_pinned());
        assert_eq!(r.kind, None);
        assert_eq!(
            r.sources,
            vec![Source::LocalDir {
                name: GIVEN_NAMESPACE.to_string(),
                dir: Some(PathBuf::from("/models")),
            }]
        );
        assert!(r.urls().is_empty());
        assert_eq!(r.cache_key(), "my.pt");
        assert_eq!(r.to_string(), "checkpoint: my.pt (unpinned)");
        r.validate().unwrap();

        let bare = Resource::given("checkpoint", "my.pt");
        assert_eq!(bare.file, "my.pt");
        assert_eq!(
            bare.sources,
            vec![Source::LocalDir {
                name: GIVEN_NAMESPACE.to_string(),
                dir: Some(PathBuf::from("")),
            }]
        );
    }

    #[test]
    fn test_validate_names_the_problem() {
        let r = CHECKPOINT.to_resource("openai", &BASES);

        let mut no_key = r.clone();
        no_key.key.clear();
        assert!(matches!(no_key.validate(), Err(BunsenError::Invalid(m)) if m.contains("no key")));

        let mut no_file = r.clone();
        no_file.file.clear();
        assert!(
            matches!(no_file.validate(), Err(BunsenError::Invalid(m)) if m == "checkpoint: no file name")
        );

        let mut no_namespace = r.clone();
        no_namespace.namespace.clear();
        assert!(
            matches!(no_namespace.validate(), Err(BunsenError::Invalid(m)) if m.contains("no namespace"))
        );

        let mut no_source = r.clone();
        no_source.sources.clear();
        assert!(
            matches!(no_source.validate(), Err(BunsenError::Invalid(m)) if m == "checkpoint: no source")
        );

        let mut bad_digest = r.clone();
        bad_digest.sha256 = Some("ABC".to_string());
        assert!(
            matches!(bad_digest.validate(), Err(BunsenError::Invalid(m)) if m.contains("sha256"))
        );

        let mut unpinned = r;
        unpinned.sha256 = None;
        assert!(!unpinned.is_pinned());
        unpinned.validate().unwrap();
        assert_eq!(
            unpinned.to_string(),
            "checkpoint: tiny.en.pt (pytorch fp16, unpinned)"
        );
    }

    #[test]
    fn test_is_sha256_hex() {
        assert!(is_sha256_hex(ABC_SHA256));
        assert!(!is_sha256_hex(&ABC_SHA256.to_uppercase()));
        assert!(!is_sha256_hex(&ABC_SHA256[1..]));
        assert!(!is_sha256_hex(""));
    }
}
