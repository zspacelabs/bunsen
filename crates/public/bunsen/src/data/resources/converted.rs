use std::{
    fs,
    path::PathBuf,
};

use sha2::{
    Digest,
    Sha256,
};

use super::{
    RESOURCES_DIR,
    ResourceCache,
    ResourceSet,
};

/// The revision of the converted-weights layout: part of every key, so a
/// change to how weights are converted retires every copy made before it.
pub const CONVERTED_WEIGHTS_REVISION: u32 = 1;

/// The directory segment, under a build's name, its converted copies are in:
/// `resources/<build>/converted/<key>/<kit>.bpk`.
pub const CONVERTED_DIR: &str = "converted";

impl ResourceCache {
    /// Whether loads keep and use converted copies of the weights they read
    /// ([`ResourceCacheOptions::converted_weights`](super::ResourceCacheOptions::converted_weights)).
    pub fn converted_weights(&self) -> bool {
        self.converted_weights
    }

    /// The key of a kit's converted copy of `weights`, under a load whose
    /// other choices `facets` spells (its load policy, the device's float
    /// dtypes): a lowercase hex SHA-256 over them, the kit, every weight
    /// resource's key, name and pin, the bunsen version, and
    /// [`CONVERTED_WEIGHTS_REVISION`]. Read from the resource set alone: the
    /// weights' files need not be here.
    ///
    /// `None` when a weight resource is unpinned, so a copy could not be
    /// told from one of other weights.
    pub fn converted_key(
        &self,
        kit: &str,
        weights: &ResourceSet,
        facets: &str,
    ) -> Option<String> {
        let mut hasher = Sha256::new();
        hasher.update(format!(
            "converted-weights {CONVERTED_WEIGHTS_REVISION}\nbunsen {}\nkit {kit}\n{facets}\n",
            env!("CARGO_PKG_VERSION")
        ));
        for (key, resource) in &weights.resources {
            let pin = resource.sha256.as_deref()?;
            hasher.update(format!("{key} {} {pin}\n", resource.name));
        }
        Some(
            hasher
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        )
    }

    /// Where a kit's converted copy of `weights` under `key` lives, present
    /// or not: `resources/<build>/converted/<key>/<kit>.bpk`, the build being
    /// the name the weights' resources share, their slot aside
    /// (`qwen/qwen3/qwen3-4b` for `qwen/qwen3/qwen3-4b/checkpoint`). So
    /// `cache list` shows it under `<build>/converted`, and clearing the
    /// build's `checkpoint` leaves it.
    ///
    /// `None` for no weights.
    pub fn converted_path(
        &self,
        kit: &str,
        weights: &ResourceSet,
        key: &str,
    ) -> Option<PathBuf> {
        let name = &weights.resources.values().next()?.name;
        let build = name
            .rsplit_once('/')
            .map_or(name.as_str(), |(build, _)| build);
        let dirs: Vec<&str> = core::iter::once(RESOURCES_DIR)
            .chain(build.split('/'))
            .chain([CONVERTED_DIR, key])
            .collect();
        Some(self.disk().cache_path(&dirs, format!("{kit}.bpk")))
    }
}

/// Marks a converted copy used now, as a cache hit marks a file: its
/// modification time, which `cache list` and `cache clear --age` read.
pub fn mark_converted_used(path: &std::path::Path) {
    let now = std::time::SystemTime::now();
    if let Ok(file) = fs::OpenOptions::new().write(true).open(path) {
        let _ = file.set_times(fs::FileTimes::new().set_accessed(now).set_modified(now));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        cache::BunsenDiskCacheOptions,
        resources::{
            Resource,
            ResourceCacheOptions,
        },
    };

    fn cache(dir: &std::path::Path) -> ResourceCache {
        ResourceCache::new(
            ResourceCacheOptions::default().with_disk(
                BunsenDiskCacheOptions::default().with_cache_dir(Some(dir.to_path_buf())),
            ),
        )
        .unwrap()
    }

    fn weights(pin: Option<&str>) -> ResourceSet {
        let mut set = ResourceSet::new("toy");
        let mut resource = Resource::given("checkpoint", "/nowhere/model.safetensors");
        resource.name = "toy/lm/tiny/checkpoint".to_string();
        resource.sha256 = pin.map(str::to_string);
        set.insert(resource);
        set
    }

    #[test]
    fn test_key_and_path() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache(dir.path());
        assert!(!cache.converted_weights(), "off by default");
        let pin = "a".repeat(64);
        let key = cache
            .converted_key("qwen3", &weights(Some(&pin)), "policy x")
            .unwrap();
        assert_eq!(key.len(), 64);
        assert_eq!(
            cache
                .converted_key("qwen3", &weights(Some(&pin)), "policy x")
                .unwrap(),
            key
        );
        assert_ne!(
            cache
                .converted_key("qwen3", &weights(Some(&pin)), "policy y")
                .unwrap(),
            key
        );
        assert_ne!(
            cache
                .converted_key("gemma", &weights(Some(&pin)), "policy x")
                .unwrap(),
            key
        );
        assert_ne!(
            cache
                .converted_key("qwen3", &weights(Some(&"b".repeat(64))), "policy x")
                .unwrap(),
            key
        );
        assert!(
            cache
                .converted_key("qwen3", &weights(None), "policy x")
                .is_none(),
            "unpinned"
        );

        let path = cache
            .converted_path("qwen3", &weights(Some(&pin)), &key)
            .unwrap();
        assert!(
            path.ends_with(format!("resources/toy/lm/tiny/converted/{key}/qwen3.bpk")),
            "{path:?}"
        );
    }

    /// A converted copy is listed under its build's `converted` name, and
    /// clearing the build's `checkpoint` leaves it.
    #[test]
    fn test_listed_and_kept_apart() {
        let dir = tempfile::tempdir().unwrap();
        let cache = cache(dir.path());
        let pin = "a".repeat(64);
        let set = weights(Some(&pin));
        let key = cache.converted_key("qwen3", &set, "").unwrap();
        let path = cache.converted_path("qwen3", &set, &key).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"pack").unwrap();
        let files = cache.cached_files(&"toy/lm/tiny".parse().unwrap()).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "toy/lm/tiny/converted");
        assert_eq!(files[0].pin, key);
        assert!(
            cache
                .cached_files(&"toy/lm/tiny/checkpoint".parse().unwrap())
                .unwrap()
                .is_empty()
        );
    }
}
