//! # Loaded resources
//!
//! A [`ResourceMap`] with every resource local: key to path and provenance.
//! What a construction hook is handed, and all it is handed.

use std::{
    collections::BTreeMap,
    path::Path,
};

use super::{
    ResolvedResource,
    ResourceMap,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
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

    /// Every key, in key order.
    pub fn keys(&self) -> Vec<&str> {
        self.parts.keys().map(String::as_str).collect()
    }

    /// Every resource with its key, in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ResolvedResource)> {
        self.parts.iter().map(|(k, r)| (k.as_str(), r))
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
}
