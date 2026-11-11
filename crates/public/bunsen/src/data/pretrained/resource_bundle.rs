//! Resource bundles: named sets of resources, and how two of them fuse.

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
    not_found,
};
use crate::errors::{
    BunsenError,
    BunsenErrorKind,
    BunsenResult,
    LookupError,
    ResultContext,
};

/// How two resource bundles combine when a key is in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fuse {
    /// A repeated key is an error: for authoring, where it is a mistake.
    Strict,

    /// A repeated key takes the right resource bundle's resource: for an
    /// override.
    Overlay,
}

/// A named set of resources, by key.
///
/// The unit a pretrained is described in, cached by, and fused from. The
/// keys are the kit's; nothing here knows which one is the model. A
/// [`Pretrained`](super::Pretrained) row is one resource bundle, fused from the
/// resource bundles it lists; [`ResourceBundle::given`] is a path as a resource
/// bundle; [`PretrainedCache::load`](super::PretrainedCache::load) brings a
/// whole resource bundle local, as [`LoadedResources`](super::LoadedResources).
///
/// Built from a build catalog's build, deserialized, or assembled at runtime;
/// [`validate`](Self::validate) checks a hand-built one. Two resource bundles
/// [fuse](Self::fuse) by key, with no memory of where a resource came from:
/// [`Fuse::Strict`] for authoring, where a repeated key is a mistake, or
/// [`Fuse::Overlay`] for an override, where the right-hand resource bundle
/// wins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceBundle {
    /// The resource bundle's name, for messages.
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

impl ResourceBundle {
    /// An empty resource bundle called `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            license: None,
            origin: None,
            resources: BTreeMap::new(),
        }
    }

    /// A resource bundle of one resource: a file on disk already, under `key`.
    /// What a path on a command line becomes.
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

    /// The bundle's size in bytes: the sum of its resources' sizes, when
    /// every one is known.
    pub fn size(&self) -> Option<u64> {
        self.resources.values().map(|r| r.size).sum()
    }

    /// The distinct dtypes its resources declare, in key order.
    pub fn dtypes(&self) -> Vec<&str> {
        let mut dtypes = Vec::new();
        for dtype in self.resources.values().filter_map(|r| r.dtype.as_deref()) {
            if !dtypes.contains(&dtype) {
                dtypes.push(dtype);
            }
        }
        dtypes
    }

    /// Every key, in key order.
    pub fn keys(&self) -> Vec<&str> {
        self.resources.keys().map(String::as_str).collect()
    }

    /// The resources keyed `key` or `key.<part>`, as a resource bundle of their
    /// own under this resource bundle's name: a checkpoint that is one file, or
    /// one split across several (`checkpoint.index`, `checkpoint.00001`, ...).
    pub fn family(
        &self,
        key: &str,
    ) -> ResourceBundle {
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
    /// keys there are, under a frame naming the resource bundle.
    pub fn try_get(
        &self,
        key: &str,
    ) -> BunsenResult<&Resource> {
        self.get(key)
            .ok_or_else(|| not_found(Some(&self.name), "resource", key, &self.keys()))
    }

    /// Checks the resource bundle hangs together: every resource validates and
    /// sits under its own key.
    ///
    /// # Errors
    /// [`Illegal`](BunsenErrorKind::Illegal) naming the first problem,
    /// under a frame naming the resource bundle: a resource filed under a key
    /// not its own, or one that does not [validate](Resource::validate). The
    /// resource bundle is declared wrong.
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

    /// The union of this resource bundle and `other`, by key, keeping this
    /// resource bundle's name and listing fields.
    ///
    /// # Errors
    /// Under [`Fuse::Strict`], [`Illegal`](BunsenErrorKind::Illegal), with a
    /// [`LookupError`] for the key in both, under a frame naming the resource
    /// bundles: the resource bundles were authored to be disjoint.
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

    /// This resource bundle with the named resources served from files on disk,
    /// in place: each resource's sources become one local-dir source called
    /// `name` at its file's directory. What a bundle laid out at build time
    /// becomes a provider through.
    ///
    /// # Errors
    /// As [`try_get`](Self::try_get) for a key the resource bundle lacks;
    /// [`Illegal`](BunsenErrorKind::Illegal), under a frame naming the resource
    /// bundle, when a path's file name is not the resource's.
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
        let map = checkpoint().fuse(vocabulary(), Fuse::Strict).unwrap();
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
    use crate::data::pretrained::GIVEN_NAMESPACE;

    const ABC_SHA256: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    fn upstream_dir() -> Option<PathBuf> {
        Some(PathBuf::from("/home/someone/.cache/whisper"))
    }

    /// A one-file resource bundle with two sources: a checkpoint, as the
    /// whisper catalog spells one.
    fn checkpoint() -> ResourceBundle {
        ResourceBundle {
            name: "a/tiny.en.pt".to_string(),
            description: "a checkpoint".to_string(),
            license: Some("MIT".to_string()),
            origin: Some("https://a.example".to_string()),
            resources: BTreeMap::new(),
        }
        .with_resource(Resource {
            size: None,
            dtype: None,
            key: "checkpoint".to_string(),
            file: "tiny.en.pt".to_string(),
            sha256: Some(ABC_SHA256.to_string()),
            kind: Some("pytorch fp16".to_string()),
            namespace: "a".to_string(),
            sources: vec![
                Source::LocalDir {
                    name: "upstream".to_string(),
                    dir: upstream_dir(),
                },
                Source::Url("https://a.example/models/tiny.en.pt".to_string()),
            ],
        })
    }

    /// A one-file resource bundle with a mirror and a URL: a vocabulary.
    fn vocabulary() -> ResourceBundle {
        ResourceBundle::new("a/gpt2.tiktoken").with_resource(Resource {
            size: None,
            dtype: None,
            key: "vocabulary".to_string(),
            file: "gpt2.tiktoken".to_string(),
            sha256: None,
            kind: Some("tiktoken".to_string()),
            namespace: "a".to_string(),
            sources: vec![
                Source::Url("https://mirror.example/gpt2.tiktoken".to_string()),
                Source::Url("https://raw.example/assets/gpt2.tiktoken".to_string()),
            ],
        })
    }

    /// A bundle answers by key, validates, and survives a round trip
    /// through JSON.
    #[test]
    fn test_accessors_and_round_trip() {
        let map = checkpoint();
        assert_eq!(map.name, "a/tiny.en.pt");
        assert_eq!(map.license.as_deref(), Some("MIT"));
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
        assert_eq!(map.keys(), ["checkpoint"]);
        assert_eq!(map.get("checkpoint").unwrap().namespace, "a");
        assert!(map.get("vocabulary").is_none());
        map.validate().unwrap();

        let json = serde_json::to_string(&map).unwrap();
        assert_eq!(serde_json::from_str::<ResourceBundle>(&json).unwrap(), map);
    }

    /// A bundle's size is known when every resource's is; its dtypes are
    /// the distinct ones its resources declare.
    #[test]
    fn test_size_and_dtypes() {
        let map = checkpoint().fuse(vocabulary(), Fuse::Strict).unwrap();
        assert_eq!(map.size(), None);
        assert!(map.dtypes().is_empty());

        let mut sized = map.clone();
        for (i, r) in sized.resources.values_mut().enumerate() {
            r.size = Some(10 + i as u64);
            r.dtype = Some("bfloat16".to_string());
        }
        assert_eq!(sized.size(), Some(21));
        assert_eq!(sized.dtypes(), ["bfloat16"]);
        sized.resources.get_mut("vocabulary").unwrap().dtype = Some("float32".to_string());
        assert_eq!(sized.dtypes(), ["bfloat16", "float32"]);
        assert_eq!(ResourceBundle::new("empty").size(), Some(0));
    }

    /// A strict fuse takes the union and keeps the left resource bundle's name;
    /// a repeated key is refused and named.
    #[test]
    fn test_fuse_strict() {
        let fused = checkpoint().fuse(vocabulary(), Fuse::Strict).unwrap();
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
            .assert_err(&checkpoint().fuse(checkpoint(), Fuse::Strict));
    }

    /// An overlay lets the right resource bundle replace: a given path over a
    /// row.
    #[test]
    fn test_fuse_overlay() {
        let row = checkpoint().fuse(vocabulary(), Fuse::Strict).unwrap();
        let overlaid = row
            .fuse(
                ResourceBundle::given("flags", "vocabulary", "/mine/vocab.tiktoken"),
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
        let mut map = ResourceBundle::given("flags", "checkpoint", "/models/my.pt");
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

        let empty = ResourceBundle::new("empty");
        assert!(empty.is_empty());
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::eq("empty: no resource \"x\""))
            .assert_err(&empty.try_get("x"));
        empty.validate().unwrap();
    }

    /// A hand-built resource bundle is checked: a resource filed under the
    /// wrong key, or one that does not validate itself, is named with the
    /// resource bundle.
    #[test]
    fn test_validate_names_the_problem() {
        let mut misfiled = ResourceBundle::new("m");
        misfiled.resources.insert(
            "vocabulary".to_string(),
            Resource::given("checkpoint", "/x.pt"),
        );
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .display(text::eq(
                "m: resource \"checkpoint\" is filed under \"vocabulary\"",
            ))
            .assert_err(&misfiled.validate());

        let mut no_source = ResourceBundle::given("m", "checkpoint", "/x.pt");
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
