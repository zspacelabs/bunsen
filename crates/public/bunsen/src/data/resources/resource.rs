//! Resources: one file of a resource set, and the places it can be had from.

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
    ConstraintError,
};

/// The namespace of a resource that arrived as a path rather than a row.
pub const GIVEN_NAMESPACE: &str = "given";

/// The format of a safetensors weights file: a whole checkpoint, or one
/// shard of a checkpoint split across several.
pub const SAFETENSORS: &str = "safetensors";

/// The format of the index of a sharded safetensors checkpoint
/// (`model.safetensors.index.json`): the shard each tensor is in.
pub const SAFETENSORS_INDEX: &str = "safetensors-index";

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
/// A resource's local sources are tried first, in the order listed: a
/// directory another tool keeps the file in is used in place, and bytes
/// linked into the binary are written into the cache on first use. Its
/// URLs come after, in the order listed, each fetched into the cache. A
/// local directory carries the directory it resolved to. A
/// [`Bundled`](Self::Bundled) source has no manifest form: serializing one
/// is an error, and none can be read back.
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
            Self::Url(url) => write!(f, "url: {url}"),
            Self::LocalDir {
                name,
                dir: Some(dir),
            } => write!(f, "local_dir: {name} ({})", dir.display()),
            Self::LocalDir { name, dir: None } => write!(f, "local_dir: {name} (unresolved)"),
            Self::Bundled(bytes) => write!(f, "bundled: ({} bytes)", bytes.len()),
        }
    }
}

/// One file of a resource set.
///
/// Its resource set's bases are already appended to its sources, so it
/// stands alone. A resource's sources
/// are its own first, then its resource set's bases with its file name
/// appended, each in the order listed; a fused resource set needs no memory
/// of which resource set a resource came from.
///
/// It is what a resource set is keyed by, and what
/// [`ResourceCache::resolve`](super::ResourceCache::resolve) brings
/// local, in its [source order](super::ResourceCache#source-order). A
/// path on a command line becomes one through [`given`](Self::given).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resource {
    /// The slot it fills in its set (`checkpoint`, `vocabulary`). Opaque
    /// here; a kit's constant.
    pub slot: String,

    /// The file name, under every base.
    pub file: String,

    /// Lowercase hex SHA-256 of the file: the pin in its cache path. A
    /// download and bundled bytes are checked against it, and a local-dir
    /// file only when the cache's options ask. `None` leaves the file
    /// unpinned, which a bundled source may not be.
    pub sha256: Option<String>,

    /// The file's format, the catalog's `$format` (`safetensors`). Nothing in
    /// this layer reads it; a kit's hook may, and a resource that arrived
    /// without a build has none.
    pub format: Option<String>,

    /// The file's size in bytes, when its row says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,

    /// The element type a checkpoint stores its weights in, as its
    /// publisher names it (`bfloat16`), when its row says.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,

    /// Its full name, which the cache keeps its file under: a catalog
    /// resource's (`qwen/qwen3/qwen3-0.6b/checkpoint`),
    /// `hf/<org>/<repo>/<slot>` for an `hf:` repo's file, `given` for a
    /// path's.
    pub name: String,

    /// Where the file can be had from, in preference order, its resource
    /// set's bases already appended.
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
            size: None,
            dtype: None,
            slot: key.into(),
            file,
            sha256: None,
            format: None,
            name: GIVEN_NAMESPACE.to_string(),
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
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] on `Resource` naming the first field at fault,
    /// under a frame naming the resource: its key, or its file when it has no
    /// key. The resource is declared wrong; this is not
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource),
    /// since nothing outside the program was read. A caller that read the
    /// declaration from outside the program re-marks it
    /// [`as_policy`](crate::errors::BunsenError::as_policy).
    pub fn validate(&self) -> BunsenResult<()> {
        let problem = if self.slot.is_empty() {
            Some(ConstraintError::zero_or_empty("Resource", "key"))
        } else if self.file.is_empty() {
            Some(ConstraintError::zero_or_empty("Resource", "file"))
        } else if self.name.is_empty() {
            Some(ConstraintError::zero_or_empty("Resource", "namespace"))
        } else if self.sources.is_empty() {
            Some(ConstraintError::zero_or_empty("Resource", "sources"))
        } else if let Some(sha256) = &self.sha256
            && !is_sha256_hex(sha256)
        {
            Some(ConstraintError::custom(
                "Resource",
                "sha256",
                format!("{sha256:?} is not 64 lowercase hex digits"),
            ))
        } else if self.bundled().is_some() && self.sha256.is_none() {
            Some(Self::bundled_without_digest())
        } else {
            None
        };
        match problem {
            None => Ok(()),
            Some(problem) => {
                let name = if self.slot.is_empty() {
                    &self.file
                } else {
                    &self.slot
                };
                Err(BunsenError::from(problem).context(name))
            }
        }
    }

    /// The rule a bundled source without a digest breaks: an unpinned file is
    /// cached under a name key, and a rebuilt bundle would be shadowed by a
    /// stale cached one. [`validate`](Self::validate) checks it, and so does
    /// the cache, which bundled bytes reach without a validate.
    pub(crate) fn bundled_without_digest() -> ConstraintError {
        ConstraintError::custom(
            "Resource",
            "sha256",
            "a bundled source needs a digest to be cached under",
        )
    }
}

impl fmt::Display for Resource {
    /// `checkpoint: tiny.en.pt (pytorch fp16, pinned)`.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}: {}", self.slot, self.file)?;
        let pin = if self.is_pinned() {
            "pinned"
        } else {
            "unpinned"
        };
        match &self.format {
            Some(kind) => write!(f, " ({kind}, {pin})"),
            None => write!(f, " ({pin})"),
        }
    }
}

/// `true` for 64 lowercase hex digits.
pub(crate) fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The directory a path template names, or `None` when it does not
/// resolve.
///
/// A template is a path whose leading `~` is the home directory and whose
/// `$NAME` segments are environment variables: `~/.cache/whisper`,
/// `$XDG_CACHE_HOME/whisper`. A template that names an unset variable, or
/// `~` without a home directory, does not resolve.
pub fn resolve_dir_template(template: &str) -> Option<PathBuf> {
    resolve_dir_template_with(template, &|name| std::env::var(name).ok(), &|| {
        std::env::home_dir()
    })
}

fn resolve_dir_template_with(
    template: &str,
    env: &dyn Fn(&str) -> Option<String>,
    home: &dyn Fn() -> Option<PathBuf>,
) -> Option<PathBuf> {
    let mut path = PathBuf::new();
    for (i, segment) in template.split('/').enumerate() {
        if i == 0 && segment == "~" {
            path.push(home()?);
        } else if let Some(name) = segment.strip_prefix('$') {
            path.push(env(name).filter(|v| !v.is_empty())?);
        } else if i == 0 && segment.is_empty() {
            path.push("/");
        } else if !segment.is_empty() {
            path.push(segment);
        }
    }
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::{
            ErrorMatcher,
            predicate,
            text,
        },
    };

    /// Sources print as a listing shows them, a local dir with the
    /// directory it resolved to; bundled bytes have no manifest form.
    #[test]
    fn test_source_display() {
        assert_eq!(
            Source::LocalDir {
                name: "upstream".to_string(),
                dir: Some(PathBuf::from("/home/someone/.cache/upstream")),
            }
            .to_string(),
            "local_dir: upstream (/home/someone/.cache/upstream)"
        );
        assert_eq!(
            Source::LocalDir {
                name: "x".to_string(),
                dir: None
            }
            .to_string(),
            "local_dir: x (unresolved)"
        );
        assert_eq!(
            Source::Url("https://a.example/my_model.pt".to_string()).to_string(),
            "url: https://a.example/my_model.pt"
        );
        let bundled = Source::Bundled(BundledBytes(b"abc"));
        assert_eq!(format!("{bundled:?}"), "Bundled(3 bytes)");
        assert!(
            serde_json::to_string(&bundled).is_err(),
            "in-binary bytes have no manifest form"
        );
    }

    /// A bundled source is cached under the digest, so it needs one.
    #[test]
    fn test_a_bundled_source_needs_a_digest() {
        let mut r = Resource {
            size: None,
            dtype: None,
            slot: "burnpack".to_string(),
            file: "model.bpk".to_string(),
            sha256: None,
            format: None,
            name: "bunsen".to_string(),
            sources: vec![Source::Bundled(BundledBytes(b"abc"))],
        };
        assert_eq!(r.bundled(), Some(&b"abc"[..]));
        assert!(r.urls().is_empty());
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("bundled source needs a digest")
            .frame_contains("burnpack")
            .has_cause::<ConstraintError>()
            .assert_err(&r.validate());
        r.sha256 =
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".to_string());
        r.validate().unwrap();
        assert!(r.is_pinned());
    }

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn upstream_dir() -> Option<PathBuf> {
        Some(PathBuf::from("/home/someone/.cache/whisper"))
    }

    /// A pinned checkpoint with a mirror of its own, then an upstream
    /// directory, then a base URL.
    fn checkpoint() -> Resource {
        Resource {
            size: None,
            dtype: None,
            slot: "checkpoint".to_string(),
            file: "tiny.en.pt".to_string(),
            sha256: Some(ABC_SHA256.to_string()),
            format: Some("pytorch fp16".to_string()),
            name: "openai".to_string(),
            sources: vec![
                Source::Url("https://mirror.example/tiny.en.pt".to_string()),
                Source::LocalDir {
                    name: "upstream".to_string(),
                    dir: upstream_dir(),
                },
                Source::Url("https://a.example/models/tiny.en.pt".to_string()),
            ],
        }
    }

    /// A resource's URLs are its URL sources in order; it prints as a listing
    /// shows it, and round-trips.
    #[test]
    fn test_resource_urls_key_and_display() {
        let r = checkpoint();
        assert!(r.is_pinned());
        assert_eq!(
            r.urls(),
            vec![
                "https://mirror.example/tiny.en.pt",
                "https://a.example/models/tiny.en.pt"
            ]
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
        assert_eq!(r.slot, "checkpoint");
        assert_eq!(r.file, "my.pt");
        assert_eq!(r.name, GIVEN_NAMESPACE);
        assert!(!r.is_pinned());
        assert_eq!(r.format, None);
        assert_eq!(
            r.sources,
            vec![Source::LocalDir {
                name: GIVEN_NAMESPACE.to_string(),
                dir: Some(PathBuf::from("/models")),
            }]
        );
        assert!(r.urls().is_empty());
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
        let r = checkpoint();

        let broken = |field: &'static str| {
            ErrorMatcher::kind(BunsenErrorKind::Illegal).cause(predicate(
                format!("a broken Resource.{field}"),
                move |c: &ConstraintError| c.owner == "Resource" && c.field == field,
            ))
        };

        let mut no_key = r.clone();
        no_key.slot.clear();
        broken("key")
            .frame_contains("tiny.en.pt")
            .assert_err(&no_key.validate());

        let mut no_file = r.clone();
        no_file.file.clear();
        assert_eq!(
            no_file.validate().unwrap_err().to_string(),
            "checkpoint: Resource.file: must not be zero or empty"
        );

        let mut no_namespace = r.clone();
        no_namespace.name.clear();
        broken("namespace").assert_err(&no_namespace.validate());

        let mut no_source = r.clone();
        no_source.sources.clear();
        broken("sources")
            .display(text::eq(
                "checkpoint: Resource.sources: must not be zero or empty",
            ))
            .assert_err(&no_source.validate());

        let mut bad_digest = r.clone();
        bad_digest.sha256 = Some("ABC".to_string());
        broken("sha256")
            .message_contains("\"ABC\" is not 64 lowercase hex digits")
            .assert_err(&bad_digest.validate());

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

    /// A template resolves `~` and `$NAME` segments, and does not resolve
    /// when a variable is unset or empty, or there is no home.
    #[test]
    fn test_resolve_dir_template() {
        let env = |name: &str| match name {
            "XDG_CACHE_HOME" => Some("/x/cache".to_string()),
            "EMPTY" => Some(String::new()),
            _ => None,
        };
        let home = || Some(PathBuf::from("/home/u"));
        let resolve = |t: &str| resolve_dir_template_with(t, &env, &home);
        assert_eq!(
            resolve("$XDG_CACHE_HOME/whisper"),
            Some(PathBuf::from("/x/cache/whisper"))
        );
        assert_eq!(
            resolve("~/.cache/whisper"),
            Some(PathBuf::from("/home/u/.cache/whisper"))
        );
        assert_eq!(resolve("/abs/dir"), Some(PathBuf::from("/abs/dir")));
        assert_eq!(resolve("$UNSET/whisper"), None);
        assert_eq!(resolve("$EMPTY/whisper"), None);
        assert_eq!(resolve_dir_template_with("~/x", &env, &|| None), None);
    }
}
