//! Resource maps: named sets of resources, and how two of them fuse.

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
use std::path::{
    Path,
    PathBuf,
};

use serde::{
    Deserialize,
    Serialize,
};

use super::{
    Resource,
    Source,
    StaticBase,
    StaticResource,
    not_found,
};
use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    LookupError,
    ResultContext,
};

/// A named set of resources under a set of bases, as a compiled-in table
/// spells it.
///
/// [`to_map`](Self::to_map) appends every [base](StaticBase) to every
/// file, so each owned [`Resource`] stands alone.
#[derive(Debug)]
pub struct StaticResourceMap<'a> {
    /// The map's name, for messages.
    pub name: &'a str,

    /// A line for a listing.
    pub description: &'a str,

    /// The license the files are distributed under.
    pub license: Option<&'a str>,

    /// Where the files are published.
    pub origin: Option<&'a str>,

    /// The cache segment the files live under: who published them.
    pub namespace: &'a str,

    /// Mirrors, in preference order. Each is appended to every file name,
    /// after the file's own sources.
    pub bases: &'a [StaticBase<'a>],

    /// The files, in listing order.
    pub resources: &'a [StaticResource<'a>],
}

impl StaticResourceMap<'_> {
    /// The owned twin, every base appended to every file.
    pub fn to_map(&self) -> ResourceMap {
        ResourceMap {
            name: self.name.to_string(),
            description: self.description.to_string(),
            license: self.license.map(str::to_string),
            origin: self.origin.map(str::to_string),
            resources: self
                .resources
                .iter()
                .map(|r| (r.key.to_string(), r.to_resource(self.namespace, self.bases)))
                .collect(),
        }
    }

    /// Every key, in listing order.
    pub fn keys(&self) -> Vec<&str> {
        self.resources.iter().map(|r| r.key).collect()
    }

    /// The resource called `key`.
    pub fn get(
        &self,
        key: &str,
    ) -> Option<&StaticResource<'_>> {
        self.resources.iter().find(|r| r.key == key)
    }
}

impl From<&StaticResourceMap<'_>> for ResourceMap {
    fn from(map: &StaticResourceMap<'_>) -> Self {
        map.to_map()
    }
}

/// How two maps combine when a key is in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fuse {
    /// A repeated key is an error: for authoring, where it is a mistake.
    Strict,

    /// A repeated key takes the right map's resource: for an override.
    Overlay,
}

/// A named set of resources, by key.
///
/// The unit a pretrained is described in, cached by, and fused from. The
/// keys are the kit's; nothing here knows which one is the model. A
/// [`Pretrained`](super::Pretrained) row is one map, fused from the maps
/// it lists; [`ResourceMap::given`] is a path as a map;
/// [`PretrainedCache::load`](super::PretrainedCache::load) brings a whole
/// map local, as [`LoadedResources`](super::LoadedResources).
///
/// Built from a [`StaticResourceMap`], deserialized, or assembled at
/// runtime; [`validate`](Self::validate) checks a hand-built one. Two maps
/// [fuse](Self::fuse) by key, with no memory of where a resource came
/// from: [`Fuse::Strict`] for authoring, where a repeated key is a
/// mistake, or [`Fuse::Overlay`] for an override, where the right-hand map
/// wins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceMap {
    /// The map's name, for messages.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The license the files are distributed under.
    pub license: Option<String>,

    /// Where the files are published.
    pub origin: Option<String>,

    /// The resources, by key.
    pub resources: BTreeMap<String, Resource>,
}

impl ResourceMap {
    /// An empty map called `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            license: None,
            origin: None,
            resources: BTreeMap::new(),
        }
    }

    /// A map of one resource: a file on disk already, under `key`. What a
    /// path on a command line becomes.
    pub fn given(
        name: impl Into<String>,
        key: impl Into<String>,
        path: impl Into<PathBuf>,
    ) -> Self {
        Self::new(name).with_resource(Resource::given(key, path))
    }

    /// Adds `resource` under its key, replacing one already there.
    pub fn with_resource(
        mut self,
        resource: Resource,
    ) -> Self {
        self.insert(resource);
        self
    }

    /// Adds `resource` under its key, and returns the one it replaced.
    pub fn insert(
        &mut self,
        resource: Resource,
    ) -> Option<Resource> {
        self.resources.insert(resource.key.clone(), resource)
    }

    /// Every key, in key order.
    pub fn keys(&self) -> Vec<&str> {
        self.resources.keys().map(String::as_str).collect()
    }

    /// The resources keyed `key` or `key.<part>`, as a map of their own
    /// under this map's name: a checkpoint that is one file, or one split
    /// across several (`checkpoint.index`, `checkpoint.00001`, ...).
    pub fn family(
        &self,
        key: &str,
    ) -> ResourceMap {
        let prefix = format!("{key}.");
        let mut family = Self::new(self.name.clone());
        family.description.clone_from(&self.description);
        family.license.clone_from(&self.license);
        family.origin.clone_from(&self.origin);
        for (k, resource) in &self.resources {
            if k == key || k.starts_with(&prefix) {
                family.insert(resource.clone());
            }
        }
        family
    }

    /// The number of resources.
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Whether there are no resources.
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// The resource called `key`.
    pub fn get(
        &self,
        key: &str,
    ) -> Option<&Resource> {
        self.resources.get(key)
    }

    /// The resource called `key`.
    ///
    /// # Errors
    /// [`Lookup`](BunsenErrorKind::Lookup), with a [`LookupError`] naming the
    /// keys there are, under a frame naming the map.
    pub fn try_get(
        &self,
        key: &str,
    ) -> BunsenResult<&Resource> {
        self.get(key)
            .ok_or_else(|| not_found(Some(&self.name), "resource", key, &self.keys()))
    }

    /// Checks the map hangs together: every resource validates and sits
    /// under its own key.
    ///
    /// # Errors
    /// [`Illegal`](BunsenErrorKind::Illegal) naming the first problem,
    /// under a frame naming the map: a resource filed under a key not its
    /// own, or one that does not [validate](Resource::validate). The map is
    /// declared wrong.
    pub fn validate(&self) -> BunsenResult<()> {
        for (key, resource) in &self.resources {
            if key != &resource.key {
                return Err(BunsenError::illegal(format!(
                    "resource {:?} is filed under {key:?}",
                    resource.key
                ))
                .context(&self.name));
            }
            resource.validate().context(&self.name)?;
        }
        Ok(())
    }

    /// The union of this map and `other`, by key, keeping this map's name
    /// and listing fields.
    ///
    /// # Errors
    /// Under [`Fuse::Strict`], [`Illegal`](BunsenErrorKind::Illegal), with a
    /// [`LookupError`] for the key in both, under a frame naming the maps:
    /// the maps were authored to be disjoint.
    pub fn fuse(
        mut self,
        other: Self,
        how: Fuse,
    ) -> BunsenResult<Self> {
        for (key, resource) in other.resources {
            if how == Fuse::Strict && self.resources.contains_key(&key) {
                return Err(BunsenError::from_cause(
                    BunsenErrorKind::Illegal,
                    LookupError::duplicate("resource", key),
                )
                .context(format!("fusing {} into {}", other.name, self.name)));
            }
            self.resources.insert(key, resource);
        }
        Ok(self)
    }

    /// This map with the named resources served from files on disk, in
    /// place: each resource's sources become one local-dir source called
    /// `name` at its file's directory. What a bundle laid out at build time
    /// becomes a provider through.
    ///
    /// # Errors
    /// As [`try_get`](Self::try_get) for a key the map lacks;
    /// [`Illegal`](BunsenErrorKind::Illegal), under a frame naming the map,
    /// when a path's file name is not the resource's.
    pub fn with_local_files(
        mut self,
        name: &str,
        files: &[(&str, &Path)],
    ) -> BunsenResult<Self> {
        for (key, path) in files {
            let resource = self.try_get(key)?;
            let file_name = path.file_name().map(|n| n.to_string_lossy().into_owned());
            if file_name.as_deref() != Some(resource.file.as_str()) {
                return Err(BunsenError::illegal(format!(
                    "{key} is {}, and {} is not it",
                    resource.file,
                    path.display()
                ))
                .context(&self.name));
            }
            let dir = path.parent().map(Path::to_path_buf);
            let resource = self.resources.get_mut(*key).expect("try_get found it");
            resource.sources = vec![Source::LocalDir {
                name: name.to_string(),
                dir,
            }];
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        ConstraintError,
        LookupProblem,
        testing::{
            ErrorMatcher,
            predicate,
            text,
        },
    };

    /// `with_local_files` serves the named resources in place, keeping
    /// their digests and namespaces, and checks each file name.
    #[test]
    fn test_with_local_files_replaces_sources_and_checks_names() {
        let map = CHECKPOINT
            .to_map()
            .fuse(VOCABULARY.to_map(), Fuse::Strict)
            .unwrap();
        let bundled = map
            .clone()
            .with_local_files(
                "bundled",
                &[("checkpoint", Path::new("/build/out/tiny.en.pt"))],
            )
            .unwrap();
        let r = bundled.get("checkpoint").unwrap();
        assert_eq!(
            r.sources,
            vec![Source::LocalDir {
                name: "bundled".to_string(),
                dir: Some(PathBuf::from("/build/out")),
            }]
        );
        assert_eq!(r.sha256, map.get("checkpoint").unwrap().sha256);
        assert_eq!(r.namespace, "a");
        assert_eq!(
            bundled.get("vocabulary").unwrap().sources,
            map.get("vocabulary").unwrap().sources,
            "an unnamed resource is untouched"
        );

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("tiny.en.pt")
            .message_contains("other.pt")
            .assert_err(
                &map.clone()
                    .with_local_files("bundled", &[("checkpoint", Path::new("/x/other.pt"))]),
            );
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .has_cause::<LookupError>()
            .assert_err(
                &map.with_local_files("bundled", &[("config", Path::new("/x/config.json"))]),
            );
    }
    use crate::data::pretrained::{
        GIVEN_NAMESPACE,
        Source,
        StaticSource,
    };

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn upstream_dir() -> Option<PathBuf> {
        Some(PathBuf::from("/home/someone/.cache/whisper"))
    }

    /// A one-file map with two bases: a checkpoint, as the whisper table
    /// spells one.
    static CHECKPOINT: StaticResourceMap<'static> = StaticResourceMap {
        name: "a/tiny.en.pt",
        description: "a checkpoint",
        license: Some("MIT"),
        origin: Some("https://a.example"),
        namespace: "a",
        bases: &[
            StaticBase::LocalDir {
                name: "upstream",
                default: upstream_dir,
            },
            StaticBase::Url("https://a.example/models"),
        ],
        resources: &[StaticResource {
            key: "checkpoint",
            file: "tiny.en.pt",
            sha256: Some(ABC_SHA256),
            kind: Some("pytorch fp16"),
            sources: &[],
        }],
    };

    /// A one-file map with a mirror of its own and a URL base: a
    /// vocabulary.
    static VOCABULARY: StaticResourceMap<'static> = StaticResourceMap {
        name: "a/gpt2.tiktoken",
        description: "a vocabulary",
        license: None,
        origin: None,
        namespace: "a",
        bases: &[StaticBase::Url("https://raw.example/assets/")],
        resources: &[StaticResource {
            key: "vocabulary",
            file: "gpt2.tiktoken",
            sha256: None,
            kind: Some("tiktoken"),
            sources: &[StaticSource::Url("https://mirror.example/gpt2.tiktoken")],
        }],
    };

    #[test]
    fn test_static_to_map() {
        assert_eq!(CHECKPOINT.keys(), ["checkpoint"]);
        assert_eq!(
            CHECKPOINT.get("checkpoint").map(|r| r.file),
            Some("tiny.en.pt")
        );
        assert!(CHECKPOINT.get("vocabulary").is_none());

        let map = CHECKPOINT.to_map();
        assert_eq!(map, ResourceMap::from(&CHECKPOINT));
        assert_eq!(map.name, "a/tiny.en.pt");
        assert_eq!(map.license.as_deref(), Some("MIT"));
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
        assert_eq!(map.keys(), ["checkpoint"]);
        let r = map.get("checkpoint").unwrap();
        assert_eq!(r.namespace, "a");
        assert_eq!(
            r.sources,
            vec![
                Source::LocalDir {
                    name: "upstream".to_string(),
                    dir: upstream_dir(),
                },
                Source::Url("https://a.example/models/tiny.en.pt".to_string()),
            ]
        );
        map.validate().unwrap();

        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(serde_json::from_str::<ResourceMap>(&json).unwrap(), map);
    }

    /// A strict fuse takes the union and keeps the left map's name; a
    /// repeated key is refused and named.
    #[test]
    fn test_fuse_strict() {
        let fused = CHECKPOINT
            .to_map()
            .fuse(VOCABULARY.to_map(), Fuse::Strict)
            .unwrap();
        assert_eq!(fused.name, "a/tiny.en.pt");
        assert_eq!(fused.keys(), ["checkpoint", "vocabulary"]);
        assert_eq!(
            fused.get("vocabulary").unwrap().sources,
            vec![
                Source::Url("https://mirror.example/gpt2.tiktoken".to_string()),
                Source::Url("https://raw.example/assets/gpt2.tiktoken".to_string()),
            ]
        );
        fused.validate().unwrap();

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_eq("duplicate resource \"checkpoint\"")
            .frame_contains("fusing a/tiny.en.pt into a/tiny.en.pt")
            .cause(predicate(
                "a duplicate \"checkpoint\"",
                |c: &LookupError| c.problem == LookupProblem::Duplicate && c.key == "checkpoint",
            ))
            .assert_err(&CHECKPOINT.to_map().fuse(CHECKPOINT.to_map(), Fuse::Strict));
    }

    /// An overlay lets the right map replace: a given path over a row.
    #[test]
    fn test_fuse_overlay() {
        let row = CHECKPOINT
            .to_map()
            .fuse(VOCABULARY.to_map(), Fuse::Strict)
            .unwrap();
        let overlaid = row
            .fuse(
                ResourceMap::given("flags", "vocabulary", "/mine/vocab.tiktoken"),
                Fuse::Overlay,
            )
            .unwrap();
        assert_eq!(overlaid.name, "a/tiny.en.pt");
        assert_eq!(overlaid.keys(), ["checkpoint", "vocabulary"]);
        let vocab = overlaid.get("vocabulary").unwrap();
        assert_eq!(vocab.namespace, GIVEN_NAMESPACE);
        assert_eq!(vocab.file, "vocab.tiktoken");
        assert_eq!(vocab.kind, None);
        assert_eq!(
            vocab.sources,
            vec![Source::LocalDir {
                name: GIVEN_NAMESPACE.to_string(),
                dir: Some(PathBuf::from("/mine")),
            }]
        );
        overlaid.validate().unwrap();
    }

    #[test]
    fn test_given_insert_and_lookup() {
        let mut map = ResourceMap::given("flags", "checkpoint", "/models/my.pt");
        assert_eq!(map.keys(), ["checkpoint"]);
        assert_eq!(map.try_get("checkpoint").unwrap().file, "my.pt");
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::eq(
                "flags: no resource \"vocabulary\"; there are: checkpoint",
            ))
            .cause(predicate("candidates: checkpoint", |c: &LookupError| {
                c.candidates == ["checkpoint"]
            }))
            .assert_err(&map.try_get("vocabulary"));

        let replaced = map.insert(Resource::given("checkpoint", "/models/other.pt"));
        assert_eq!(replaced.map(|r| r.file), Some("my.pt".to_string()));
        assert_eq!(map.get("checkpoint").unwrap().file, "other.pt");

        let empty = ResourceMap::new("empty");
        assert!(empty.is_empty());
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::eq("empty: no resource \"x\""))
            .assert_err(&empty.try_get("x"));
        empty.validate().unwrap();
    }

    /// A hand-built map is checked: a resource filed under the wrong key,
    /// or one that does not validate itself, is named with the map.
    #[test]
    fn test_validate_names_the_problem() {
        let mut misfiled = ResourceMap::new("m");
        misfiled.resources.insert(
            "vocabulary".to_string(),
            Resource::given("checkpoint", "/x.pt"),
        );
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .display(text::eq(
                "m: resource \"checkpoint\" is filed under \"vocabulary\"",
            ))
            .assert_err(&misfiled.validate());

        let mut no_source = ResourceMap::given("m", "checkpoint", "/x.pt");
        no_source
            .resources
            .get_mut("checkpoint")
            .unwrap()
            .sources
            .clear();
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .display(text::eq(
                "m: checkpoint: Resource.sources: must not be zero or empty",
            ))
            .has_cause::<ConstraintError>()
            .assert_err(&no_source.validate());
    }
}
