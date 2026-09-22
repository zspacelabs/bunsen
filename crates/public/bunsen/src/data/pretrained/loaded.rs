//! # Loaded resources
//!
//! A [`ResourceMap`] with every resource local: key to path and provenance.
//! What a construction hook is handed, and all it is handed.
//!
//! [`LoadedResources::materialize`] is a directory view over it, for a
//! loader that reads a directory rather than paths: every part linked or
//! copied into one place under its resource's file name. An operation, not
//! the storage truth, which stays per file under its digest.

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
};
use crate::{
    data::cache::link_or_copy,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// A resource map with every resource local.
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
    /// [`BunsenError::ResourceNotFound`] naming the keys there are.
    pub fn expect(
        &self,
        key: &str,
    ) -> BunsenResult<&Path> {
        self.get(key).map(|r| r.path.as_path()).ok_or_else(|| {
            let keys = if self.parts.is_empty() {
                "(none)".to_string()
            } else {
                self.keys().join(", ")
            };
            BunsenError::ResourceNotFound(format!(
                "{}: no resource {key:?}; there are: {keys}",
                self.map.name
            ))
        })
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
    /// [`BunsenError::Invalid`] if two parts would land under one file
    /// name; [`BunsenError::External`] if the directory or a link cannot
    /// be made.
    pub fn materialize(
        &self,
        dir: impl Into<PathBuf>,
    ) -> BunsenResult<PathBuf> {
        let dir = dir.into();
        fs::create_dir_all(&dir).map_err(BunsenError::external)?;
        let mut placed: BTreeMap<&str, &str> = BTreeMap::new();
        for (key, part) in self.iter() {
            let file = self
                .map
                .get(key)
                .map(|r| r.file.as_str())
                .unwrap_or_else(|| key);
            if let Some(other) = placed.insert(file, key) {
                return Err(BunsenError::Invalid(format!(
                    "{}: {key} and {other} would both land as {file:?}",
                    self.map.name
                )));
            }
            let dest = dir.join(file);
            if dest.symlink_metadata().is_ok() {
                fs::remove_file(&dest).map_err(BunsenError::external)?;
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
    use crate::data::pretrained::{
        Provenance,
        Resource,
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
        match loaded.expect("config") {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert_eq!(
                    m,
                    "m: no resource \"config\"; there are: checkpoint, vocabulary"
                );
            }
            other => panic!("{other:?}"),
        }

        let empty = LoadedResources {
            map: ResourceMap::new("empty"),
            parts: BTreeMap::new(),
        };
        assert!(matches!(
            empty.expect("x"),
            Err(BunsenError::ResourceNotFound(m)) if m.ends_with("there are: (none)")
        ));
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
        assert!(matches!(
            clash.materialize(dir.path().join("clash")),
            Err(BunsenError::Invalid(m)) if m.contains("same.bin")
        ));
    }
}
