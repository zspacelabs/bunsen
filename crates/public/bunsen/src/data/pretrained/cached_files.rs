//! What a pretrained cache holds, and removing it.
use std::{
    fmt,
    fs,
    path::{
        Path,
        PathBuf,
    },
    str::FromStr,
};

use super::{
    PRETRAINED_DIR,
    PretrainedCache,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
    sys_at,
};

/// A file in a [`PretrainedCache`]:
/// `<cache>/pretrained/<kit>/<namespace>/<pin>/<file>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CachedFile {
    /// The kit it belongs to: `whisper`, `qwen3-asr`.
    pub kit: String,

    /// Its namespace within the kit: `openai`, `fermion`.
    pub namespace: String,

    /// The directory pinning it: its SHA-256, or a cache key for a file
    /// fetched unpinned.
    pub pin: String,

    /// Its file name, without the `.partial` of an unfinished download.
    pub file: String,

    /// Where it is.
    pub path: PathBuf,

    /// Its size.
    pub bytes: u64,

    /// An unfinished download (`<file>.partial`), which a fetch resumes.
    pub partial: bool,
}

/// Which of a [`PretrainedCache`]'s files: every one, a kit's, or one
/// namespace of a kit's.
///
/// Parsed from `""` (every file), `kit` or `kit/namespace`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheSelector {
    /// The kit, or every kit.
    pub kit: Option<String>,

    /// The namespace within [`kit`](Self::kit), or every namespace; never
    /// set without a kit.
    pub namespace: Option<String>,
}

impl CacheSelector {
    /// Whether a file of `kit` and `namespace` is selected.
    pub fn matches(
        &self,
        kit: &str,
        namespace: &str,
    ) -> bool {
        self.kit.as_deref().is_none_or(|k| k == kit)
            && self.namespace.as_deref().is_none_or(|n| n == namespace)
    }
}

impl FromStr for CacheSelector {
    type Err = BunsenError;

    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for more than two
    /// parts, or an empty one.
    fn from_str(s: &str) -> BunsenResult<Self> {
        if s.is_empty() {
            return Ok(Self::default());
        }
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() > 2 || parts.iter().any(|p| p.is_empty()) {
            return Err(BunsenError::illegal(format!(
                "cache selector {s:?}: expected `kit` or `kit/namespace`"
            )));
        }
        Ok(Self {
            kit: Some(parts[0].to_string()),
            namespace: parts.get(1).map(|n| n.to_string()),
        })
    }
}

impl fmt::Display for CacheSelector {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match (&self.kit, &self.namespace) {
            (None, _) => write!(f, "every kit"),
            (Some(kit), None) => write!(f, "{kit}"),
            (Some(kit), Some(namespace)) => write!(f, "{kit}/{namespace}"),
        }
    }
}

/// The directories in `dir`, by name, in name order; none when `dir` does
/// not exist.
fn subdirs(dir: &Path) -> BunsenResult<Vec<(String, PathBuf)>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(sys_at("read the directory", dir))? {
        let entry = entry.map_err(sys_at("read the directory", dir))?;
        let path = entry.path();
        if path.is_dir()
            && let Some(name) = entry.file_name().to_str()
        {
            out.push((name.to_string(), path));
        }
    }
    out.sort();
    Ok(out)
}

impl PretrainedCache {
    /// The directory holding every kit's files: `<cache>/pretrained`.
    pub fn pretrained_dir(&self) -> PathBuf {
        self.cache_dir().join(PRETRAINED_DIR)
    }

    /// The files the cache holds under `selector`, in path order.
    ///
    /// Only files at the cache's layout depth are listed; anything else
    /// under the pretrained directory is not the cache's.
    ///
    /// # Errors
    /// A [`Sys`](crate::errors::BunsenErrorKind::Sys) error if a directory or
    /// a file's metadata cannot be read.
    pub fn cached_files(
        &self,
        selector: &CacheSelector,
    ) -> BunsenResult<Vec<CachedFile>> {
        let mut out = Vec::new();
        for (kit, kit_dir) in subdirs(&self.pretrained_dir())? {
            for (namespace, ns_dir) in subdirs(&kit_dir)? {
                if !selector.matches(&kit, &namespace) {
                    continue;
                }
                for (pin, pin_dir) in subdirs(&ns_dir)? {
                    let entries =
                        fs::read_dir(&pin_dir).map_err(sys_at("read the directory", &pin_dir))?;
                    let mut files = Vec::new();
                    for entry in entries {
                        let entry = entry.map_err(sys_at("read the directory", &pin_dir))?;
                        let path = entry.path();
                        let meta = entry
                            .metadata()
                            .map_err(sys_at("read the metadata of", &path))?;
                        let Some(name) = entry.file_name().to_str().map(String::from) else {
                            continue;
                        };
                        if !meta.is_file() {
                            continue;
                        }
                        let (file, partial) = match name.strip_suffix(".partial") {
                            Some(file) => (file.to_string(), true),
                            None => (name, false),
                        };
                        files.push(CachedFile {
                            kit: kit.clone(),
                            namespace: namespace.clone(),
                            pin: pin.clone(),
                            file,
                            path,
                            bytes: meta.len(),
                            partial,
                        });
                    }
                    files.sort_by(|a, b| a.path.cmp(&b.path));
                    out.extend(files);
                }
            }
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
            // The pin, namespace and kit directories, while each is empty.
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
        pretrained::PretrainedCacheOptions,
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
        let kit: CacheSelector = "whisper".parse().unwrap();
        assert_eq!(kit.kit.as_deref(), Some("whisper"));
        assert_eq!(kit.namespace, None);
        assert!(kit.matches("whisper", "openai") && !kit.matches("parakeet", "nvidia"));
        let ns: CacheSelector = "whisper/hf".parse().unwrap();
        assert!(ns.matches("whisper", "hf") && !ns.matches("whisper", "openai"));
        assert_eq!(ns.to_string(), "whisper/hf");
        for bad in ["a/b/c", "/b", "a/", "a//b"] {
            assert!(bad.parse::<CacheSelector>().is_err(), "{bad}");
        }
    }

    /// Files at the layout's depth are listed, in path order, selected by
    /// kit and namespace, with an unfinished download marked; removing them
    /// frees their bytes and the directories they leave empty.
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

        put(&cache, "whisper/openai/aaa/tiny.pt", 3);
        put(&cache, "whisper/openai/bbb/base.pt.partial", 5);
        put(&cache, "whisper/hf/ccc/model.safetensors", 7);
        put(&cache, "parakeet/nvidia/ddd/model.safetensors", 11);
        // Not at the layout's depth: not the cache's.
        put(&cache, "whisper/stray.txt", 1);

        let all = cache.cached_files(&CacheSelector::default()).unwrap();
        let names: Vec<(&str, &str, &str, bool)> = all
            .iter()
            .map(|f| {
                (
                    f.kit.as_str(),
                    f.namespace.as_str(),
                    f.file.as_str(),
                    f.partial,
                )
            })
            .collect();
        assert_eq!(
            names,
            [
                ("parakeet", "nvidia", "model.safetensors", false),
                ("whisper", "hf", "model.safetensors", false),
                ("whisper", "openai", "tiny.pt", false),
                ("whisper", "openai", "base.pt", true),
            ]
        );
        assert_eq!(all[3].bytes, 5);
        assert_eq!(all[2].pin, "aaa");

        let openai = cache
            .cached_files(&"whisper/openai".parse().unwrap())
            .unwrap();
        assert_eq!(openai.len(), 2);
        assert_eq!(cache.remove_cached(&openai).unwrap(), 8);
        assert!(!cache.pretrained_dir().join("whisper/openai").exists());
        assert!(cache.pretrained_dir().join("whisper/hf/ccc").exists());

        let whisper = cache.cached_files(&"whisper".parse().unwrap()).unwrap();
        assert_eq!(cache.remove_cached(&whisper).unwrap(), 7);
        // The stray file keeps `whisper` itself.
        assert!(cache.pretrained_dir().join("whisper/stray.txt").exists());
        assert!(!cache.pretrained_dir().join("whisper/hf").exists());
        assert_eq!(
            cache.cached_files(&CacheSelector::default()).unwrap().len(),
            1
        );

        let mut outside = all[0].clone();
        outside.path = dir.path().join("elsewhere");
        assert!(cache.remove_cached(&[outside]).is_err());
    }
}
