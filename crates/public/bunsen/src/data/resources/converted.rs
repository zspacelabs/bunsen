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
    /// What loads do with converted copies of the weights they read
    /// ([`ResourceCacheOptions::converted_weights`](super::ResourceCacheOptions::converted_weights)).
    pub fn converted_weights(&self) -> super::ConvertedWeights {
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
/// modification time, which `cache list`, `cache clear --age` and `cache gc`
/// read; and its manifest's ([`converted_manifest_path`]), so that a manifest
/// is as new as the copy's last use.
pub fn mark_converted_used(path: &std::path::Path) {
    let now = std::time::SystemTime::now();
    for path in [path.to_path_buf(), converted_manifest_path(path)] {
        if let Ok(file) = fs::OpenOptions::new().write(true).open(&path) {
            let _ = file.set_times(fs::FileTimes::new().set_accessed(now).set_modified(now));
        }
    }
}

/// The manifest beside a converted copy: `<kit>.bpk.json` for `<kit>.bpk`.
pub fn converted_manifest_path(copy: &std::path::Path) -> PathBuf {
    let mut name = copy.as_os_str().to_os_string();
    name.push(".json");
    PathBuf::from(name)
}

/// A cached resource a converted copy stands for: what the next load with the
/// copy's choices does not read. Its cache file is
/// `resources/<name>/<pin>/<file>`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ShadowedResource {
    /// The resource's full name (`qwen/qwen3/qwen3-4b/checkpoint`).
    pub name: String,

    /// Its SHA-256.
    pub pin: String,

    /// Its file under the pin.
    pub file: String,
}

/// The record beside a converted copy ([`converted_manifest_path`]): what it
/// was made from and under which choices, and which cached resources it
/// shadows. `cache gc` reads it to clear sources no load with the copy's
/// choices reads.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ConvertedManifest {
    /// The converted-weights layout ([`CONVERTED_WEIGHTS_REVISION`]).
    pub revision: u32,

    /// The bunsen version that wrote the copy: part of its key, so another
    /// version's loads never read it.
    pub bunsen: String,

    /// The kit whose copy it is.
    pub kit: String,

    /// The copy's key ([`ResourceCache::converted_key`]).
    pub key: String,

    /// The load's choices the key spells (its policy, the device's floats).
    pub facets: String,

    /// The resources the copy stands for.
    pub shadows: Vec<ShadowedResource>,
}

impl ConvertedManifest {
    /// The manifest of a kit's copy of `weights` under `key`, saved now by this
    /// bunsen, under `facets`.
    pub fn new(
        kit: &str,
        key: &str,
        facets: &str,
        weights: &ResourceSet,
    ) -> Self {
        Self {
            revision: CONVERTED_WEIGHTS_REVISION,
            bunsen: env!("CARGO_PKG_VERSION").to_string(),
            kit: kit.to_string(),
            key: key.to_string(),
            facets: facets.to_string(),
            shadows: weights
                .resources
                .values()
                .filter_map(|r| {
                    Some(ShadowedResource {
                        name: r.name.clone(),
                        pin: r.sha256.clone()?,
                        file: r.file.clone(),
                    })
                })
                .collect(),
        }
    }

    /// Whether this bunsen's loads can read the copy: it was written by this
    /// version, under this layout ([`CONVERTED_WEIGHTS_REVISION`]). A copy that
    /// is not current is stale: no load computes its key.
    pub fn is_current(&self) -> bool {
        self.revision == CONVERTED_WEIGHTS_REVISION && self.bunsen == env!("CARGO_PKG_VERSION")
    }

    /// Writes the manifest of the copy at `copy` ([`converted_manifest_path`]):
    /// to a `.partial` beside it first, then renamed.
    ///
    /// # Errors
    /// A [`Sys`](crate::errors::BunsenErrorKind::Sys) error when it cannot be
    /// written.
    pub fn write(
        &self,
        copy: &std::path::Path,
    ) -> crate::errors::BunsenResult<()> {
        use crate::errors::sys_at;
        let path = converted_manifest_path(copy);
        let partial = path.with_extension("json.partial");
        let text = serde_json::to_string_pretty(self).map_err(|e| {
            crate::errors::BunsenError::internal("a converted manifest").with_cause(e)
        })?;
        fs::write(&partial, text).map_err(sys_at("write", &partial))?;
        fs::rename(&partial, &path).map_err(sys_at("rename", &partial))
    }

    /// The manifest of the copy at `copy`, when it has one.
    ///
    /// # Errors
    /// A [`Sys`](crate::errors::BunsenErrorKind::Sys) error when it cannot be
    /// read, [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// when it does not parse.
    pub fn read(copy: &std::path::Path) -> crate::errors::BunsenResult<Option<Self>> {
        use crate::errors::sys_at;
        let path = converted_manifest_path(copy);
        if !path.is_file() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path).map_err(sys_at("read", &path))?;
        serde_json::from_str(&text).map(Some).map_err(|e| {
            crate::errors::BunsenError::invalid_resource(format!("{}", path.display()))
                .with_cause(e)
        })
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
        assert_eq!(
            cache.converted_weights(),
            super::super::ConvertedWeights::Off,
            "off by default"
        );
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

    /// A manifest names the copy's weights by name, pin and file, round-trips
    /// beside its copy, and is touched with it; another version's is stale.
    #[test]
    fn test_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let copy = dir.path().join("qwen3.bpk");
        fs::write(&copy, b"bpk").unwrap();
        assert_eq!(ConvertedManifest::read(&copy).unwrap(), None);

        let pin = "a".repeat(64);
        let manifest = ConvertedManifest::new("qwen3", "k", "policy x", &weights(Some(&pin)));
        assert_eq!(
            manifest.shadows,
            [ShadowedResource {
                name: "toy/lm/tiny/checkpoint".into(),
                pin: pin.clone(),
                file: "model.safetensors".into(),
            }]
        );
        assert_eq!(manifest.bunsen, env!("CARGO_PKG_VERSION"));
        assert!(manifest.is_current());
        let older = ConvertedManifest {
            bunsen: "0.0.1".into(),
            ..manifest.clone()
        };
        assert!(!older.is_current());
        manifest.write(&copy).unwrap();
        assert_eq!(
            ConvertedManifest::read(&copy).unwrap().as_ref(),
            Some(&manifest)
        );
        assert_eq!(
            converted_manifest_path(&copy),
            dir.path().join("qwen3.bpk.json")
        );

        let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000);
        for path in [copy.clone(), converted_manifest_path(&copy)] {
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(old)
                .unwrap();
        }
        mark_converted_used(&copy);
        let modified = |p: &std::path::Path| fs::metadata(p).unwrap().modified().unwrap();
        assert!(
            modified(&converted_manifest_path(&copy)) > old,
            "touched with its copy"
        );
    }
}
