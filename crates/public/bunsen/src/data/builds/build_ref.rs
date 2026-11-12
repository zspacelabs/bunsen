//! Build references: what a spec resolved to.

use alloc::{
    boxed::Box,
    string::String,
};

use crate::{
    data::{
        builds::qualified_id,
        catalog::{
            ResolvedBuild,
            presets::PresetSelection,
        },
        resources::{
            Fuse,
            ResourceSet,
        },
    },
    errors::BunsenResult,
};

/// What a spec resolved to: a build a provider answered with, or a resource
/// set from somewhere else, such as a path.
///
/// The first step of the name-to-model pathway, shared by every kit. A
/// [`BuildFactory`](crate::data::builds::BuildFactory) makes a
/// [`Named`](Self::Named) ref from a spec; a path or a manifest becomes a
/// [`Given`](Self::Given) one through `From<ResourceSet>`. A
/// [`Deferred`](crate::data::builds::Deferred) model pairs the ref with the
/// kit's hook and loads it, and the hook plans and builds from the ref:
/// [`to_resource_set`](Self::to_resource_set) for its files,
/// [`stated`](Self::stated) for an argument the build binds, such as the
/// geometry it promises.
///
/// A ref is plain data: the build and the resource set are values, and the
/// provider is held by name, so a ref outlives the factory that made it.
/// The caller's resources ride on it through
/// [`with_overlay`](Self::with_overlay), so a hook sees one thing.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildRef {
    /// A build a provider answered with.
    Named {
        /// The provider's name: the `provider` of `provider:ref`.
        provider: String,
        /// The build. Boxed: a build is much larger than a given resource
        /// set.
        build: Box<ResolvedBuild>,
        /// What the lookup asked of the build's presets: the presets its box
        /// named, and its patch.
        selection: PresetSelection,
        /// The caller's resources, laid over the build's by slot
        /// ([`with_overlay`](Self::with_overlay)).
        overlay: Option<ResourceSet>,
    },

    /// A resource set from somewhere else: a path, a manifest. No geometry
    /// is promised; what it is, the kit finds out by looking.
    Given(ResourceSet),
}

impl From<ResourceSet> for BuildRef {
    fn from(map: ResourceSet) -> Self {
        Self::Given(map)
    }
}

impl BuildRef {
    /// `build`, as `provider` answered it, with what the lookup asked of its
    /// presets.
    pub fn named(
        provider: impl Into<String>,
        build: ResolvedBuild,
        selection: PresetSelection,
    ) -> Self {
        Self::Named {
            provider: provider.into(),
            build: Box::new(build),
            selection,
            overlay: None,
        }
    }

    /// The qualified id, `provider:ref`
    /// ([`qualified_id`](crate::data::builds::qualified_id)), or the given
    /// resource set's name.
    pub fn id(&self) -> String {
        match self {
            Self::Named {
                provider, build, ..
            } => qualified_id(provider, build),
            Self::Given(map) => map.name.clone(),
        }
    }

    /// What the lookup asked of the build's presets, if the spec was a name.
    pub fn selection(&self) -> Option<&PresetSelection> {
        match self {
            Self::Named { selection, .. } => Some(selection),
            Self::Given(_) => None,
        }
    }

    /// The provider's name and the build, if the spec was a name.
    pub fn build(&self) -> Option<(&str, &ResolvedBuild)> {
        match self {
            Self::Named {
                provider, build, ..
            } => Some((provider, build)),
            Self::Given(_) => None,
        }
    }

    /// The value the build binds its contract's argument slot `slot` to,
    /// read as `C`: a stated geometry, say. `None` for a given set, or a
    /// build that binds no such slot.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) when the value
    /// doesn't read as `C`.
    pub fn stated<C: serde::de::DeserializeOwned>(
        &self,
        slot: &str,
    ) -> crate::errors::BunsenResult<Option<C>> {
        let Some(arg) = self.build().and_then(|(_, build)| build.args.get(slot)) else {
            return Ok(None);
        };
        serde_json::from_value(arg.value.clone())
            .map(Some)
            .map_err(|e| {
                crate::errors::BunsenError::illegal(alloc::format!(
                    "{}: the argument `{slot}` doesn't read as its config",
                    self.id()
                ))
                .with_cause(e)
            })
    }

    /// The resource set: the build's files in the loader's form
    /// ([`ResolvedBuild::resource_set`]), under the overlay and named by the
    /// qualified id, or the given resource set.
    pub fn to_resource_set(&self) -> ResourceSet {
        match self {
            Self::Named { build, overlay, .. } => {
                let mut map = build.resource_set();
                if let Some(overlay) = overlay {
                    for resource in overlay.resources.values() {
                        map.insert(resource.clone());
                    }
                }
                map.name = self.id();
                map
            }
            Self::Given(map) => map.clone(),
        }
    }

    /// The caller's resources over the model's, by slot
    /// ([`Fuse::Overlay`]): a `--vocab` over a build. The id and the build
    /// are unchanged; only what [`to_resource_set`](Self::to_resource_set)
    /// gives is.
    ///
    /// # Errors
    /// As [`ResourceSet::fuse`], which does not fail under an overlay.
    pub fn with_overlay(
        self,
        map: ResourceSet,
    ) -> BunsenResult<Self> {
        Ok(match self {
            Self::Named {
                provider,
                build,
                selection,
                overlay,
            } => {
                let overlay = match overlay {
                    Some(mine) => mine.fuse(map, Fuse::Overlay)?,
                    None => map,
                };
                Self::Named {
                    provider,
                    build,
                    selection,
                    overlay: Some(overlay),
                }
            }
            Self::Given(mine) => Self::Given(mine.fuse(map, Fuse::Overlay)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::data::{
        catalog::{
            CatalogName,
            document::{
                ResourceEntry,
                ResourceSource,
            },
        },
        resources::GIVEN_NAMESPACE,
    };

    fn n(name: &str) -> CatalogName {
        CatalogName::parse(name).unwrap()
    }

    fn entry(file: &str) -> ResourceEntry {
        ResourceEntry::new(
            file,
            "pytorch",
            vec![ResourceSource::Url(alloc::format!(
                "https://a.example/{file}"
            ))],
        )
    }

    /// `catalog:a/small`, as a factory has it from a provider.
    fn named() -> BuildRef {
        let mut build = ResolvedBuild::new(n("a/small"), "small weights")
            .with_resource("checkpoint", n("a/small/checkpoint"), entry("small.pt"))
            .with_resource("vocabulary", n("a/vocabulary"), entry("vocab.txt"));
        build.args.insert(
            "geometry".into(),
            crate::data::catalog::resolve::ResolvedArg {
                value: serde_json::json!({ "width": 4 }),
                set_by: n("a/small"),
            },
        );
        BuildRef::named("catalog", build, Default::default())
    }

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Shape {
        width: usize,
    }

    /// A named ref is its provider's build: the id is `provider:ref`, the
    /// resource set is the build's under that id, and a stated argument is
    /// the build's.
    #[test]
    fn test_a_named_ref_is_its_build() {
        let model = named();
        assert_eq!(model.id(), "catalog:a/small");
        let (provider, build) = model.build().unwrap();
        assert_eq!(provider, "catalog");
        assert_eq!(build.name, n("a/small"));
        assert_eq!(
            model.stated::<Shape>("geometry").unwrap(),
            Some(Shape { width: 4 })
        );
        assert_eq!(model.stated::<Shape>("other").unwrap(), None);
        assert!(model.stated::<String>("geometry").is_err());

        let map = model.to_resource_set();
        assert_eq!(
            map.name, "catalog:a/small",
            "a build's map is named by its id"
        );
        assert_eq!(map.keys(), ["checkpoint", "vocabulary"]);
        assert_eq!(map.get("checkpoint").unwrap().file, "small.pt");
        assert_eq!(map.get("checkpoint").unwrap().name, "a/small/checkpoint");
        assert_eq!(model.clone(), model);
    }

    /// A given ref is the resource set it was given: no provider, nothing
    /// stated.
    #[test]
    fn test_a_given_ref_is_its_map() {
        let map = ResourceSet::given("/models/ckpt.pt", "checkpoint", "/models/ckpt.pt");
        let model = BuildRef::from(map.clone());
        assert_eq!(model, BuildRef::Given(map.clone()));
        assert_eq!(model.id(), "/models/ckpt.pt");
        assert!(model.build().is_none());
        assert_eq!(model.stated::<Shape>("geometry").unwrap(), None);
        assert_eq!(model.to_resource_set(), map);
        assert_eq!(
            model.to_resource_set().get("checkpoint").unwrap().file,
            "ckpt.pt"
        );
    }

    /// An overlay replaces by slot on either variant and leaves the id and
    /// the build alone; the overlaid resource keeps its own name, `given`.
    #[test]
    fn test_with_overlay_replaces_by_slot_on_both_variants() {
        let vocab = ResourceSet::given("--vocab", "vocabulary", "/mine/gpt2.tiktoken");

        let model = named().with_overlay(vocab.clone()).unwrap();
        assert_eq!(model.id(), "catalog:a/small");
        let map = model.to_resource_set();
        assert_eq!(map.name, "catalog:a/small");
        assert_eq!(map.keys(), ["checkpoint", "vocabulary"]);
        let overlaid = map.get("vocabulary").unwrap();
        assert_eq!(overlaid.file, "gpt2.tiktoken");
        assert_eq!(overlaid.name, GIVEN_NAMESPACE);
        assert_eq!(map.get("checkpoint").unwrap().file, "small.pt", "untouched");
        assert_eq!(
            model.build().unwrap().1.resources["vocabulary"].entry.file,
            "vocab.txt",
            "the build is unchanged"
        );

        let given = BuildRef::from(ResourceSet::given("mine", "checkpoint", "/m/ckpt.pt"))
            .with_overlay(vocab)
            .unwrap();
        assert_eq!(given.id(), "mine");
        assert_eq!(given.to_resource_set().keys(), ["checkpoint", "vocabulary"]);
    }
}
