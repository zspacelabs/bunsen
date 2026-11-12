//! What a pretrained cache holds, and removing it.
use alloc::fmt;
use core::str::FromStr;
use std::{
    fs,
    path::{
        Path,
        PathBuf,
    },
    time::SystemTime,
};

use crate::{
    data::{
        builds::HF,
        resources::{
            GIVEN_NAMESPACE,
            PRETRAINED_DIR,
            PretrainedCache,
        },
    },
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
};

/// A file in a [`PretrainedCache`]:
/// `<cache>/pretrained/<namespace>/<pin>/<file>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedFile {
    /// The namespace it is cached under: a catalog resource's full name
    /// (`openai/whisper/base/checkpoint`), or `hf` or `given` for a file off
    /// the catalog.
    pub namespace: String,

    /// The directory pinning it: its SHA-256, or a cache key for a file
    /// fetched unpinned.
    pub pin: String,

    /// Its path under the pin, without the `.partial` of an unfinished
    /// download: a file name, or a relative path (`encoder/config.json`).
    pub file: String,

    /// Where it is.
    pub path: PathBuf,

    /// Its size.
    pub bytes: u64,

    /// An unfinished download (`<file>.partial`), which a fetch resumes.
    pub partial: bool,

    /// When it came into the cache (its creation time), where the file
    /// system records it.
    pub created: Option<SystemTime>,

    /// When it was last loaded (its modification time): a load that finds
    /// it in the cache sets it ([`PretrainedCache`]'s "Use"), and a download
    /// is its first use.
    pub last_used: Option<SystemTime>,
}

/// Which of a [`PretrainedCache`]'s files: every one, or those whose
/// namespace is under a name.
///
/// Parsed from `""` (every file) or a `/`-separated name: `qwen` selects
/// every Qwen resource, `openai/whisper/base` one build's, `hf` every file
/// fetched from the Hub off the catalog.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheSelector {
    /// The name the namespaces are under, or every namespace.
    pub prefix: Option<String>,
}

impl CacheSelector {
    /// Whether a file cached under `namespace` is selected: the namespace is
    /// the prefix, or under it.
    pub fn matches(
        &self,
        namespace: &str,
    ) -> bool {
        self.prefix.as_deref().is_none_or(|p| {
            namespace == p
                || namespace
                    .strip_prefix(p)
                    .is_some_and(|rest| rest.starts_with('/'))
        })
    }
}

impl FromStr for CacheSelector {
    type Err = BunsenError;

    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for an empty
    /// segment.
    fn from_str(s: &str) -> BunsenResult<Self> {
        if s.is_empty() {
            return Ok(Self::default());
        }
        if s.split('/').any(str::is_empty) {
            return Err(BunsenError::illegal(format!(
                "cache selector {s:?}: expected a `/`-separated name, with no empty segment"
            )));
        }
        Ok(Self {
            prefix: Some(s.to_string()),
        })
    }
}

impl fmt::Display for CacheSelector {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match &self.prefix {
            None => write!(f, "every resource"),
            Some(prefix) => write!(f, "{prefix}"),
        }
    }
}

/// Whether `segment` is a SHA-256 pin: 64 lowercase hex digits.
fn is_sha256(segment: &str) -> bool {
    segment.len() == 64
        && segment
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The files under `dir`, at any depth, in path order; none when `dir` does
/// not exist.
fn files_under(dir: &Path) -> BunsenResult<Vec<(PathBuf, fs::Metadata)>> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return Ok(out);
    }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in fs::read_dir(&at).map_err(sys_at("read the directory", &at))? {
            let entry = entry.map_err(sys_at("read the directory", &at))?;
            let path = entry.path();
            let meta = entry
                .metadata()
                .map_err(sys_at("read the metadata of", &path))?;
            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file() {
                out.push((path, meta));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

impl PretrainedCache {
    /// The directory holding every cached file: `<cache>/pretrained`.
    pub fn pretrained_dir(&self) -> PathBuf {
        self.cache_dir().join(PRETRAINED_DIR)
    }

    /// The files the cache holds under `selector`, in path order.
    ///
    /// A file's pin is the first segment of its path that is a SHA-256; the
    /// namespace is what precedes it, and the file what follows. Off the
    /// catalog, under `hf` or `given`, an unpinned file's pin is the segment
    /// after the namespace. Anything else under the pretrained directory is
    /// not the cache's, and is not listed.
    ///
    /// # Errors
    /// A [`Sys`](crate::errors::BunsenErrorKind::Sys) error if a directory or
    /// a file's metadata cannot be read.
    pub fn cached_files(
        &self,
        selector: &CacheSelector,
    ) -> BunsenResult<Vec<CachedFile>> {
        let root = self.pretrained_dir();
        let mut out = Vec::new();
        for (path, meta) in files_under(&root)? {
            let Ok(relative) = path.strip_prefix(&root) else {
                continue;
            };
            let Some(segments) = relative
                .components()
                .map(|c| c.as_os_str().to_str().map(str::to_string))
                .collect::<Option<Vec<String>>>()
            else {
                continue;
            };
            let at = match segments.iter().position(|s| is_sha256(s)) {
                Some(at) if at > 0 => at,
                _ if segments.len() > 2
                    && [HF, GIVEN_NAMESPACE].contains(&segments[0].as_str()) =>
                {
                    1
                }
                _ => continue,
            };
            if at + 1 >= segments.len() {
                continue;
            }
            let namespace = segments[..at].join("/");
            if !selector.matches(&namespace) {
                continue;
            }
            let name = segments[at + 1..].join("/");
            let (file, partial) = match name.strip_suffix(".partial") {
                Some(file) => (file.to_string(), true),
                None => (name, false),
            };
            out.push(CachedFile {
                namespace,
                pin: segments[at].clone(),
                file,
                path,
                bytes: meta.len(),
                partial,
                created: meta.created().ok(),
                last_used: meta.modified().ok(),
            });
        }
        Ok(out)
    }

    /// Removes `files` from the cache, and the directories they leave empty
    /// under its pretrained directory.
    ///
    /// # Returns
    /// The bytes freed.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), removing
    /// nothing, if a file is not under the cache's pretrained directory; a
    /// [`Sys`](crate::errors::BunsenErrorKind::Sys) error if one cannot be
    /// removed, after those before it were.
    pub fn remove_cached(
        &self,
        files: &[CachedFile],
    ) -> BunsenResult<u64> {
        let root = self.pretrained_dir();
        if let Some(stray) = files.iter().find(|f| !f.path.starts_with(&root)) {
            return Err(BunsenError::illegal(format!(
                "{} is not in the pretrained cache at {}",
                stray.path.display(),
                root.display()
            )));
        }
        let mut freed = 0;
        for file in files {
            fs::remove_file(&file.path).map_err(sys_at("remove", &file.path))?;
            freed += file.bytes;
            // The directories above it, up to the pretrained directory, while
            // each is empty.
            let mut dir = file.path.parent();
            while let Some(d) = dir.filter(|d| *d != root && d.starts_with(&root)) {
                if fs::remove_dir(d).is_err() {
                    break;
                }
                dir = d.parent();
            }
        }
        Ok(freed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        cache::BunsenDiskCacheOptions,
        resources::PretrainedCacheOptions,
    };

    fn cache_in(dir: &Path) -> PretrainedCache {
        PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap()
    }

    fn put(
        cache: &PretrainedCache,
        rel: &str,
        bytes: usize,
    ) {
        let path = cache.pretrained_dir().join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, vec![0u8; bytes]).unwrap();
    }

    #[test]
    fn test_selector() {
        assert_eq!(
            "".parse::<CacheSelector>().unwrap(),
            CacheSelector::default()
        );
        let qwen: CacheSelector = "qwen".parse().unwrap();
        assert_eq!(qwen.prefix.as_deref(), Some("qwen"));
        assert!(qwen.matches("qwen") && qwen.matches("qwen/qwen3/qwen3-0.6b/checkpoint"));
        assert!(!qwen.matches("qwen3") && !qwen.matches("openai/whisper/base/checkpoint"));
        let build: CacheSelector = "openai/whisper/base".parse().unwrap();
        assert!(build.matches("openai/whisper/base/checkpoint"));
        assert!(!build.matches("openai/whisper/base.en/checkpoint"));
        assert_eq!(build.to_string(), "openai/whisper/base");
        for bad in ["/b", "a/", "a//b"] {
            assert!(bad.parse::<CacheSelector>().is_err(), "{bad}");
        }
    }

    /// Files are listed in path order, each by its namespace and pin (a
    /// SHA-256, or the segment after `hf` or `given`), selected by a name
    /// prefix, with an unfinished download marked and a relative path kept
    /// whole; removing them frees their bytes and the directories they
    /// leave empty.
    #[test]
    fn test_list_and_remove() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache_in(dir.path());
        assert!(
            cache
                .cached_files(&CacheSelector::default())
                .unwrap()
                .is_empty()
        );

        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let d = "d".repeat(64);
        put(
            &cache,
            &format!("openai/whisper/tiny/checkpoint/{a}/tiny.pt"),
            3,
        );
        put(
            &cache,
            &format!("openai/whisper/base/checkpoint/{b}/base.pt.partial"),
            5,
        );
        put(&cache, "hf/ccc/model.safetensors", 7);
        put(
            &cache,
            &format!("laya/laya/laya/encoder_config/{d}/encoder/config.json"),
            11,
        );
        // No pin in its path: not the cache's.
        put(&cache, "openai/stray.txt", 1);

        let all = cache.cached_files(&CacheSelector::default()).unwrap();
        let names: Vec<(&str, &str, bool)> = all
            .iter()
            .map(|f| (f.namespace.as_str(), f.file.as_str(), f.partial))
            .collect();
        assert_eq!(
            names,
            [
                ("hf", "model.safetensors", false),
                (
                    "laya/laya/laya/encoder_config",
                    "encoder/config.json",
                    false
                ),
                ("openai/whisper/base/checkpoint", "base.pt", true),
                ("openai/whisper/tiny/checkpoint", "tiny.pt", false),
            ]
        );
        assert_eq!(all[0].pin, "ccc");
        assert_eq!(all[2].bytes, 5);
        assert_eq!(all[3].pin, a);

        let whisper = cache
            .cached_files(&"openai/whisper".parse().unwrap())
            .unwrap();
        assert_eq!(whisper.len(), 2);
        assert_eq!(cache.remove_cached(&whisper).unwrap(), 8);
        assert!(!cache.pretrained_dir().join("openai/whisper").exists());
        // The stray file keeps `openai` itself.
        assert!(cache.pretrained_dir().join("openai/stray.txt").exists());
        assert!(cache.pretrained_dir().join("hf/ccc").exists());

        let hf = cache.cached_files(&"hf".parse().unwrap()).unwrap();
        assert_eq!(cache.remove_cached(&hf).unwrap(), 7);
        assert!(!cache.pretrained_dir().join("hf").exists());
        assert_eq!(
            cache.cached_files(&CacheSelector::default()).unwrap().len(),
            1
        );

        let mut outside = all[0].clone();
        outside.path = dir.path().join("elsewhere");
        assert!(cache.remove_cached(&[outside]).is_err());
    }
}
