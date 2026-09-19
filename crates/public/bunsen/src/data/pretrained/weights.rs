//! # Pretrained weights descriptors
//!
//! What a set of pretrained weights is: its name and aliases, the file it
//! lands as, the digest that pins it (when one does), its format, and the
//! sources its bytes can be had from, in preference order. A static twin for
//! compiled-in tables, an owned twin for everything else.

use alloc::{
    collections::BTreeMap,
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

#[cfg(feature = "fetch")]
use crate::data::cache::BunsenDiskCache;
use crate::errors::{
    BunsenError,
    BunsenResult,
};

const X25: crc::Crc<u16> = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);

/// Builds a cache key (bare cache file name) from a name and URL.
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

/// Returns the cache resource key for a pretrained weights file.
///
/// # Arguments
///
/// - `cache_key`: the cache key (the bare cache file name).
///
/// # Returns
///
/// The cache resource key.
pub fn pretrained_weights_resource_key(cache_key: &str) -> Vec<String> {
    vec!["weights".to_string(), cache_key.to_string()]
}

/// The element type a weights file stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightsDType {
    /// IEEE half precision.
    F16,
    /// bfloat16.
    BF16,
    /// IEEE single precision.
    F32,
}

impl fmt::Display for WeightsDType {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(match self {
            Self::F16 => "fp16",
            Self::BF16 => "bf16",
            Self::F32 => "fp32",
        })
    }
}

/// How a weights file is laid out: what a kit's loader is keyed on.
///
/// A closed set, on purpose: a kit matches on it, and a new layout is a new
/// variant here rather than a string every kit parses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightsFormat {
    /// A `PyTorch` checkpoint (`torch.save`, a pickle of tensors).
    Pytorch {
        /// The tensors' element type.
        dtype: WeightsDType,
    },

    /// A `safetensors` file.
    Safetensors {
        /// The tensors' element type.
        dtype: WeightsDType,
    },

    /// A `burn` `burnpack` store.
    Burnpack,
}

impl WeightsFormat {
    /// A `PyTorch` checkpoint of fp16 tensors.
    pub const PYTORCH_F16: Self = Self::Pytorch {
        dtype: WeightsDType::F16,
    };
    /// A `PyTorch` checkpoint of fp32 tensors.
    pub const PYTORCH_F32: Self = Self::Pytorch {
        dtype: WeightsDType::F32,
    };
}

impl fmt::Display for WeightsFormat {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Pytorch { dtype } => write!(f, "pytorch {dtype}"),
            Self::Safetensors { dtype } => write!(f, "safetensors {dtype}"),
            Self::Burnpack => f.write_str("burnpack"),
        }
    }
}

/// One place a weights file can be had from, as a compiled-in table spells
/// it.
///
/// Sources are tried in the order listed. A digest-pinned descriptor checks
/// every source but a [`File`](Self::File) against its digest, since a file
/// a crate bundled was checked when it was built.
#[derive(Clone, Copy, Debug)]
pub enum StaticWeightsSource<'a> {
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

    /// A file a crate fetched and checked at build time, used in place.
    File(fn() -> &'static Path),
}

impl StaticWeightsSource<'_> {
    /// The owned twin, with the default directory and the file resolved now.
    pub fn to_source(&self) -> WeightsSource {
        match self {
            Self::Url(url) => WeightsSource::Url(url.to_string()),
            Self::LocalDir { name, default } => WeightsSource::LocalDir {
                name: name.to_string(),
                dir: default(),
            },
            Self::File(path) => WeightsSource::File(path().to_path_buf()),
        }
    }
}

impl fmt::Display for StaticWeightsSource<'_> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Url(url) => write!(f, "url {url}"),
            Self::LocalDir { name, .. } => write!(f, "local dir {name}"),
            Self::File(path) => write!(f, "file {}", path().display()),
        }
    }
}

/// One place a weights file can be had from.
///
/// The owned twin of [`StaticWeightsSource`]: a local directory carries the
/// directory it resolved to, a bundled file its path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightsSource {
    /// A URL, fetched into the cache.
    Url(String),

    /// A directory another tool keeps this file in, under its bare name.
    LocalDir {
        /// The source's name, for overrides and messages.
        name: String,
        /// The directory, when it could be resolved.
        dir: Option<PathBuf>,
    },

    /// A file on disk already, used in place.
    File(PathBuf),
}

impl fmt::Display for WeightsSource {
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
            Self::File(path) => write!(f, "file {}", path.display()),
        }
    }
}

/// Static [`PretrainedWeightsDescriptor`] provider.
#[derive(Debug)]
pub struct StaticPretrainedWeightsDescriptor<'a> {
    /// Name of the weights, unique within their map.
    pub name: &'a str,

    /// Description of the weights.
    pub description: &'a str,

    /// License.
    pub license: Option<&'a str>,

    /// Where the weights are published.
    pub origin: Option<&'a str>,

    /// The prefab these weights instantiate, by name in the kit's prefab map.
    pub prefab: &'a str,

    /// Other names the same bytes answer to.
    pub aliases: &'a [&'a str],

    /// The file name, under every source.
    pub file: &'a str,

    /// Lowercase hex SHA-256 of the file, which pins every source; `None`
    /// leaves the file unpinned.
    pub sha256: Option<&'a str>,

    /// How the file is laid out.
    pub format: WeightsFormat,

    /// Where the file can be had from, in preference order.
    pub sources: &'a [StaticWeightsSource<'a>],
}

impl StaticPretrainedWeightsDescriptor<'_> {
    /// Does `name` name these weights, by their name or an alias?
    pub fn matches(
        &self,
        name: &str,
    ) -> bool {
        self.name == name || self.aliases.contains(&name)
    }

    /// Converts to a [`PretrainedWeightsDescriptor`].
    pub fn to_descriptor(&self) -> PretrainedWeightsDescriptor {
        PretrainedWeightsDescriptor {
            name: self.name.to_string(),
            description: self.description.to_string(),
            license: self.license.map(|s| s.to_string()),
            origin: self.origin.map(|s| s.to_string()),
            prefab: self.prefab.to_string(),
            aliases: self.aliases.iter().map(|s| s.to_string()).collect(),
            file: self.file.to_string(),
            sha256: self.sha256.map(|s| s.to_string()),
            format: self.format,
            sources: self
                .sources
                .iter()
                .map(StaticWeightsSource::to_source)
                .collect(),
        }
    }
}

impl From<&StaticPretrainedWeightsDescriptor<'_>> for PretrainedWeightsDescriptor {
    fn from(descriptor: &StaticPretrainedWeightsDescriptor) -> Self {
        descriptor.to_descriptor()
    }
}

/// A descriptor for a pretrained weights file.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct PretrainedWeightsDescriptor {
    /// Name of the weights, unique within their map.
    pub name: String,

    /// Description of the weights.
    pub description: String,

    /// License.
    pub license: Option<String>,

    /// Where the weights are published.
    pub origin: Option<String>,

    /// The prefab these weights instantiate, by name in the kit's prefab map.
    pub prefab: String,

    /// Other names the same bytes answer to.
    pub aliases: Vec<String>,

    /// The file name, under every source.
    pub file: String,

    /// Lowercase hex SHA-256 of the file, which pins every source; `None`
    /// leaves the file unpinned.
    pub sha256: Option<String>,

    /// How the file is laid out.
    pub format: WeightsFormat,

    /// Where the file can be had from, in preference order.
    pub sources: Vec<WeightsSource>,
}

impl PretrainedWeightsDescriptor {
    /// Does `name` name these weights, by their name or an alias?
    pub fn matches(
        &self,
        name: &str,
    ) -> bool {
        self.name == name || self.aliases.iter().any(|a| a == name)
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
                WeightsSource::Url(url) => Some(url.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Checks the descriptor hangs together: a file name, at least one
    /// source, and a digest, if any, of 64 lowercase hex digits.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] naming the first problem.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.file.is_empty() {
            return Err(BunsenError::Invalid(format!("{}: no file name", self.name)));
        }
        if self.sources.is_empty() {
            return Err(BunsenError::Invalid(format!("{}: no source", self.name)));
        }
        if let Some(sha256) = &self.sha256
            && !(sha256.len() == 64
                && sha256
                    .bytes()
                    .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        {
            return Err(BunsenError::Invalid(format!(
                "{}: sha256 {sha256:?} is not 64 lowercase hex digits",
                self.name
            )));
        }
        Ok(())
    }

    /// Cache Key
    ///
    /// The key is ``{name}-{url crc hash}-{url basename}`` for the first URL,
    /// or ``{name}-{file}`` for weights with no URL.
    pub fn cache_key(&self) -> String {
        match self.urls().first() {
            Some(url) => url_to_cache_key(Some(&self.name), url),
            None => format!("{}-{}", self.name, self.file),
        }
    }

    /// Read-Through Cache the Model Weights
    ///
    /// The URL sources are tried in order; a pinned descriptor's file is
    /// checked against its digest as it lands, an unpinned one against its
    /// `Content-Length` alone. Local sources are not consulted here.
    ///
    /// # Returns
    ///
    /// The disk location of the cached weights.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] with no URL source; otherwise as
    /// [`BunsenDiskCache::load_cached_path`].
    #[cfg(feature = "fetch")]
    pub fn fetch_weights(
        &self,
        disk_cache: &BunsenDiskCache,
    ) -> BunsenResult<PathBuf> {
        let urls = self.urls();
        if urls.is_empty() {
            return Err(BunsenError::Invalid(format!(
                "{}: no URL to fetch from",
                self.name
            )));
        }
        let cache_key = &self.cache_key();
        let resource = pretrained_weights_resource_key(cache_key);

        disk_cache.load_cached_path(&resource, &urls, true, self.sha256.as_deref())
    }
}

/// Static [`PretrainedWeightsMap`] builder.
#[derive(Debug)]
pub struct StaticPretrainedWeightsMap<'a> {
    /// List of static descriptors.
    pub items: &'a [&'a StaticPretrainedWeightsDescriptor<'a>],
}

impl<'a> StaticPretrainedWeightsMap<'a> {
    /// Converts to a [`PretrainedWeightsMap`].
    pub fn to_directory(&self) -> PretrainedWeightsMap {
        PretrainedWeightsMap {
            items: self
                .items
                .iter()
                .map(|d| {
                    let desc = d.to_descriptor();
                    (desc.name.clone(), desc)
                })
                .collect(),
        }
    }
}

impl<'a> From<&StaticPretrainedWeightsMap<'a>> for PretrainedWeightsMap {
    fn from(directory: &StaticPretrainedWeightsMap) -> Self {
        directory.to_directory()
    }
}

/// Directory of [`PretrainedWeightsDescriptor`]s.
#[derive(Debug, Clone)]
pub struct PretrainedWeightsMap {
    /// Map of descriptors.
    pub items: BTreeMap<String, PretrainedWeightsDescriptor>,
}

impl PretrainedWeightsMap {
    /// Looks up a descriptor by name, or by one of its aliases.
    pub fn lookup_by_name(
        &self,
        name: &str,
    ) -> Option<PretrainedWeightsDescriptor> {
        self.items
            .get(name)
            .or_else(|| self.items.values().find(|d| d.matches(name)))
            .cloned()
    }

    /// Looks up a descriptor.
    pub fn try_lookup_by_name(
        &self,
        name: &str,
    ) -> BunsenResult<PretrainedWeightsDescriptor> {
        match self.lookup_by_name(name) {
            Some(d) => Ok(d),
            None => Err(BunsenError::ResourceNotFound(format!(
                "Descriptor not found: {}",
                name
            ))),
        }
    }

    /// Looks up a descriptor.
    pub fn expect_lookup_by_name(
        &self,
        name: &str,
    ) -> PretrainedWeightsDescriptor {
        match self.try_lookup_by_name(name) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn bundled_path() -> &'static Path {
        Path::new("/bundled/my_model.pt")
    }

    fn upstream_dir() -> Option<PathBuf> {
        Some(PathBuf::from("/home/someone/.cache/upstream"))
    }

    static MY_MODEL: StaticPretrainedWeightsDescriptor<'static> =
        StaticPretrainedWeightsDescriptor {
            name: "my_model",
            description: "some description of my model.",
            license: Some("MIT"),
            origin: Some("https://github.com/my_org/my_model"),
            prefab: "my_prefab",
            aliases: &["my-model", "latest"],
            file: "my_model.pt",
            sha256: Some(ABC_SHA256),
            format: WeightsFormat::PYTORCH_F16,
            sources: &[
                StaticWeightsSource::File(bundled_path),
                StaticWeightsSource::LocalDir {
                    name: "upstream",
                    default: upstream_dir,
                },
                StaticWeightsSource::Url("https://a.example/my_model.pt"),
                StaticWeightsSource::Url("https://b.example/my_model.pt"),
            ],
        };

    #[test]
    fn test_static_descriptor_to_descriptor() {
        let d = MY_MODEL.to_descriptor();
        assert_eq!(d.name, "my_model");
        assert_eq!(d.description, MY_MODEL.description);
        assert_eq!(d.license.as_deref(), Some("MIT"));
        assert_eq!(d.prefab, "my_prefab");
        assert_eq!(
            d.aliases,
            vec!["my-model".to_string(), "latest".to_string()]
        );
        assert_eq!(d.file, "my_model.pt");
        assert_eq!(d.sha256.as_deref(), Some(ABC_SHA256));
        assert_eq!(d.format, WeightsFormat::PYTORCH_F16);
        assert_eq!(
            d.sources,
            vec![
                WeightsSource::File(PathBuf::from("/bundled/my_model.pt")),
                WeightsSource::LocalDir {
                    name: "upstream".to_string(),
                    dir: upstream_dir(),
                },
                WeightsSource::Url("https://a.example/my_model.pt".to_string()),
                WeightsSource::Url("https://b.example/my_model.pt".to_string()),
            ]
        );
        assert_eq!(PretrainedWeightsDescriptor::from(&MY_MODEL), d);
        d.validate().unwrap();

        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(
            serde_json::from_str::<PretrainedWeightsDescriptor>(&json).unwrap(),
            d
        );
    }

    #[test]
    fn test_names_urls_and_keys() {
        let d = MY_MODEL.to_descriptor();
        assert!(d.matches("my_model"));
        assert!(d.matches("latest"));
        assert!(!d.matches("other"));
        assert!(d.is_pinned());
        assert_eq!(
            d.urls(),
            vec![
                "https://a.example/my_model.pt",
                "https://b.example/my_model.pt"
            ]
        );
        assert_eq!(
            d.cache_key(),
            url_to_cache_key(Some("my_model"), "https://a.example/my_model.pt")
        );

        let mut local_only = d.clone();
        local_only.sources.truncate(2);
        assert!(local_only.urls().is_empty());
        assert_eq!(local_only.cache_key(), "my_model-my_model.pt");
    }

    #[test]
    fn test_validate_names_the_problem() {
        let d = MY_MODEL.to_descriptor();

        let mut no_file = d.clone();
        no_file.file.clear();
        assert!(
            matches!(no_file.validate(), Err(BunsenError::Invalid(m)) if m.contains("no file name"))
        );

        let mut no_source = d.clone();
        no_source.sources.clear();
        assert!(
            matches!(no_source.validate(), Err(BunsenError::Invalid(m)) if m.contains("no source"))
        );

        let mut bad_digest = d.clone();
        bad_digest.sha256 = Some("ABC".to_string());
        assert!(
            matches!(bad_digest.validate(), Err(BunsenError::Invalid(m)) if m.contains("sha256"))
        );

        let mut unpinned = d.clone();
        unpinned.sha256 = None;
        assert!(!unpinned.is_pinned());
        unpinned.validate().unwrap();
    }

    #[test]
    fn test_display() {
        assert_eq!(WeightsFormat::PYTORCH_F16.to_string(), "pytorch fp16");
        assert_eq!(WeightsFormat::PYTORCH_F32.to_string(), "pytorch fp32");
        assert_eq!(
            WeightsFormat::Safetensors {
                dtype: WeightsDType::BF16
            }
            .to_string(),
            "safetensors bf16"
        );
        assert_eq!(WeightsFormat::Burnpack.to_string(), "burnpack");
        let sources: Vec<String> = MY_MODEL.sources.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            sources,
            vec![
                "file /bundled/my_model.pt",
                "local dir upstream",
                "url https://a.example/my_model.pt",
                "url https://b.example/my_model.pt",
            ]
        );
        let owned: Vec<String> = MY_MODEL
            .to_descriptor()
            .sources
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            owned[1],
            "local dir upstream (/home/someone/.cache/upstream)"
        );
        assert_eq!(
            WeightsSource::LocalDir {
                name: "x".to_string(),
                dir: None
            }
            .to_string(),
            "local dir x (unresolved)"
        );
    }

    #[test]
    fn test_map_lookup_by_name_or_alias() {
        let map = StaticPretrainedWeightsMap {
            items: &[&MY_MODEL],
        }
        .to_directory();
        assert_eq!(map.lookup_by_name("my_model").unwrap().name, "my_model");
        assert_eq!(map.lookup_by_name("latest").unwrap().name, "my_model");
        assert!(map.lookup_by_name("other").is_none());
        assert!(matches!(
            map.try_lookup_by_name("other"),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }
}
