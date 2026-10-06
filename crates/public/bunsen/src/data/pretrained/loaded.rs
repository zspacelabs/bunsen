//! Loaded resources: a resource map with every file local.

use std::{
    collections::BTreeMap,
    fs,
    path::{
        Path,
        PathBuf,
    },
};

use super::{
    ResolvedResource,
    ResourceMap,
    not_found,
};
use crate::{
    data::cache::link_or_copy,
    errors::{
        BunsenError,
        BunsenResult,
        sys_at,
    },
};

/// A resource map with every resource local: key to path and provenance.
///
/// What [`PretrainedCache::load`](super::PretrainedCache::load) returns,
/// and what a kit's [`Construct::construct`](super::Construct::construct)
/// is handed: all it is handed, besides the ref. The hook reads its parts
/// by key, with [`expect`](Self::expect), [`get`](Self::get), or
/// [`family`](Self::family) for a checkpoint split across files (as
/// [`SafetensorsCheckpoint::from_loaded`](super::SafetensorsCheckpoint::from_loaded)
/// does). The [`Loaded`](super::Loaded) handle keeps it as `resources`,
/// so a caller can say where every part came from
/// ([`Provenance`](super::Provenance)).
///
/// [`materialize`](Self::materialize) is a directory view over it, for a
/// loader that reads a directory rather than paths: every part linked or
/// copied into one place under its resource's file name. That is an
/// operation, not the storage truth, which stays per file under its digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedResources {
    /// The map that was loaded.
    pub map: ResourceMap,

    /// Where each resource is, and where it came from, by key.
    pub parts: BTreeMap<String, ResolvedResource>,
}

impl LoadedResources {
    /// The resource called `key`: its path and provenance.
    pub fn get(
        &self,
        key: &str,
    ) -> Option<&ResolvedResource> {
        self.parts.get(key)
    }

    /// The path of the resource called `key`.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup), with a
    /// [`LookupError`](crate::errors::LookupError) naming the keys there
    /// are, under a frame naming the map.
    pub fn expect(
        &self,
        key: &str,
    ) -> BunsenResult<&Path> {
        self.get(key)
            .map(|r| r.path.as_path())
            .ok_or_else(|| not_found(Some(&self.map.name), "resource", key, &self.keys()))
    }

    /// The label of the resource called `key`, for a listing or a hook's
    /// dispatch; `None` for one that arrived without a row, or without a
    /// label.
    pub fn kind(
        &self,
        key: &str,
    ) -> Option<&str> {
        self.map.get(key).and_then(|r| r.kind.as_deref())
    }

    /// The parts keyed `key` or `key.<part>`, in key order: a checkpoint
    /// that is one file, or one split across several.
    pub fn family(
        &self,
        key: &str,
    ) -> Vec<(&str, &ResolvedResource)> {
        let prefix = format!("{key}.");
        self.parts
            .iter()
            .filter(|(k, _)| *k == key || k.starts_with(&prefix))
            .map(|(k, r)| (k.as_str(), r))
            .collect()
    }

    /// Every key, in key order.
    pub fn keys(&self) -> Vec<&str> {
        self.parts.keys().map(String::as_str).collect()
    }

    /// Every resource with its key, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ResolvedResource)> {
        self.parts.iter().map(|(k, r)| (k.as_str(), r))
    }

    /// Lays every part out under `dir`, each under its resource's file
    /// name, by link where the platform has them and by copy otherwise:
    /// the directory a loader that reads directories is handed.
    ///
    /// The parts stay where they are; `dir` is a view of them. A file
    /// already at a part's place is replaced.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if two parts
    /// would land under one file name: the map is declared wrong. An
    /// `io::Error` sorted by [`sys_at`], naming the path, if the directory
    /// cannot be made or a file in the way removed; as [`link_or_copy`] if a
    /// link or copy cannot be made.
    pub fn materialize(
        &self,
        dir: impl Into<PathBuf>,
    ) -> BunsenResult<PathBuf> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(sys_at("create", &dir))?;
        let mut placed: BTreeMap<&str, &str> = BTreeMap::new();
        for (key, part) in self.iter() {
            let file = self
                .map
                .get(key)
                .map(|r| r.file.as_str())
                .unwrap_or_else(|| key);
            if let Some(other) = placed.insert(file, key) {
                return Err(BunsenError::illegal(format!(
                    "{key} and {other} would both land as {file:?}"
                ))
                .context(&self.map.name));
            }
            let dest = dir.join(file);
            if dest.symlink_metadata().is_ok() {
                fs::remove_file(&dest).map_err(sys_at("remove", &dest))?;
            }
            link_or_copy(&part.path, &dest)?;
        }
        Ok(dir)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        data::pretrained::{
            Provenance,
            Resource,
        },
        errors::{
            BunsenErrorKind,
            LookupError,
            testing::{
                ErrorMatcher,
                predicate,
            },
        },
    };

    fn loaded() -> LoadedResources {
        let mut checkpoint = Resource::given("checkpoint", "/models/tiny.pt");
        checkpoint.kind = Some("pytorch fp16".to_string());
        let map = ResourceMap::new("m")
            .with_resource(checkpoint)
            .with_resource(Resource::given("vocabulary", "/models/gpt2.tiktoken"));
        let parts = [
            ("checkpoint", "/models/tiny.pt", Provenance::LocalDir),
            ("vocabulary", "/models/gpt2.tiktoken", Provenance::LocalDir),
        ]
        .into_iter()
        .map(|(key, path, provenance)| {
            (
                key.to_string(),
                ResolvedResource {
                    path: PathBuf::from(path),
                    provenance,
                },
            )
        })
        .collect();
        LoadedResources { map, parts }
    }

    #[test]
    fn test_lookups() {
        let loaded = loaded();
        assert_eq!(loaded.keys(), ["checkpoint", "vocabulary"]);
        assert_eq!(
            loaded.get("checkpoint").map(|r| r.provenance),
            Some(Provenance::LocalDir)
        );
        assert_eq!(
            loaded.expect("vocabulary").unwrap(),
            Path::new("/models/gpt2.tiktoken")
        );
        assert_eq!(loaded.kind("checkpoint"), Some("pytorch fp16"));
        assert_eq!(loaded.kind("vocabulary"), None);
        assert_eq!(loaded.kind("config"), None);
        assert_eq!(
            loaded.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            ["checkpoint", "vocabulary"]
        );
    }

    #[test]
    fn test_expect_names_the_keys_there_are() {
        let loaded = loaded();
        let err = loaded.expect("config").unwrap_err();
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .cause(predicate("a missing \"config\"", |c: &LookupError| {
                c.key == "config" && c.candidates == ["checkpoint", "vocabulary"]
            }))
            .assert(&err);
        assert_eq!(
            err.to_string(),
            "m: no resource \"config\"; there are: checkpoint, vocabulary"
        );

        let empty = LoadedResources {
            map: ResourceMap::new("empty"),
            parts: BTreeMap::new(),
        };
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_eq("no resource \"x\"")
            .frame_contains("empty")
            .assert_err(&empty.expect("x"));
    }

    /// The view holds every part under its resource's file name, reads as
    /// the part does, and can be laid out again over itself; two parts
    /// with one file name are refused.
    #[test]
    fn test_materialize_lays_the_parts_out() {
        let dir = tempfile::tempdir().unwrap();
        let checkpoint = dir.path().join("src").join("tiny.pt");
        let vocabulary = dir.path().join("src").join("gpt2.tiktoken");
        fs::create_dir_all(checkpoint.parent().unwrap()).unwrap();
        fs::write(&checkpoint, b"weights").unwrap();
        fs::write(&vocabulary, b"ranks").unwrap();
        let map = ResourceMap::new("m")
            .with_resource(Resource::given("checkpoint", &checkpoint))
            .with_resource(Resource::given("vocabulary", &vocabulary));
        let parts = [("checkpoint", &checkpoint), ("vocabulary", &vocabulary)]
            .into_iter()
            .map(|(key, path)| {
                (
                    key.to_string(),
                    ResolvedResource {
                        path: path.clone(),
                        provenance: Provenance::LocalDir,
                    },
                )
            })
            .collect();
        let loaded = LoadedResources { map, parts };

        let view = loaded.materialize(dir.path().join("view")).unwrap();
        assert_eq!(view, dir.path().join("view"));
        assert_eq!(fs::read(view.join("tiny.pt")).unwrap(), b"weights");
        assert_eq!(fs::read(view.join("gpt2.tiktoken")).unwrap(), b"ranks");
        assert!(checkpoint.is_file(), "the part stays where it is");
        let again = loaded.materialize(&view).unwrap();
        assert_eq!(fs::read(again.join("tiny.pt")).unwrap(), b"weights");

        let clash = LoadedResources {
            map: ResourceMap::new("clash")
                .with_resource(Resource::given(
                    "a",
                    dir.path().join("one").join("same.bin"),
                ))
                .with_resource(Resource::given(
                    "b",
                    dir.path().join("two").join("same.bin"),
                )),
            parts: [("a", &checkpoint), ("b", &vocabulary)]
                .into_iter()
                .map(|(key, path)| {
                    (
                        key.to_string(),
                        ResolvedResource {
                            path: path.clone(),
                            provenance: Provenance::LocalDir,
                        },
                    )
                })
                .collect(),
        };
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("same.bin")
            .frame_contains("clash")
            .assert_err(&clash.materialize(dir.path().join("clash")));
    }
}
