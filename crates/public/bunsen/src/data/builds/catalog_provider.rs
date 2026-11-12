//! A build catalog's builds of one contract, as a provider's builds.

use alloc::{
    format,
    string::String,
    vec::Vec,
};

use crate::{
    data::{
        builds::{
            BuildProvider,
            CATALOG,
        },
        catalog::{
            BuiltinCatalog,
            BunsenCatalogStack,
            CatalogName,
            CatalogRef,
            CatalogRule,
            ResolvedBuild,
            builtin::find_spelling,
            catalog_error,
        },
    },
    errors::BunsenResult,
};

/// The argument slot a contract states a geometry in
/// ([`BuildRef::stated`](super::BuildRef::stated)): for weights
/// that carry none (resnet's, and the `stated` variants of qwen3-asr's and
/// voxtral's), the geometry they are built in; for weights that carry their
/// own (whisper's), the one they must have.
pub const GEOMETRY: &str = "geometry";

/// A provider over the builds of one contract in a build catalog, each
/// named in full (`qwen/qwen3/qwen3-0.6b`).
///
/// A contract is what a kit's loader implements
/// ([`catalog_contracts`](crate::kits::support::catalog_contracts)), so the
/// builds of a kit's contract, its variants' among them, are the ones its
/// loader can build. The builds are resolved once, when the provider is
/// made: a stack is not shared between threads, and a provider is.
///
/// A lookup takes a name as the catalog's lookups do: a build's full name,
/// or a shorter spelling whose segments are among its name's in order
/// (`qwen/qwen3-0.6b`, `qwen3-0.6b`), or one of its aliases'; a spelling two
/// builds match is ambiguous.
#[derive(Clone, Debug)]
pub struct CatalogBuildProvider {
    name: String,
    description: String,
    builds: Vec<ResolvedBuild>,
    /// The names a lookup tests, each with its build: the build's full name
    /// and each of its aliases'.
    names: Vec<(CatalogName, usize)>,
}

impl CatalogBuildProvider {
    /// The builds of `contract` (`qwen3-lm`; `qwen3-asr` for both of its
    /// variants) in `stack`, as the builds of a provider called `name`.
    ///
    /// # Errors
    /// A contract `$contracts` doesn't have, or a build that doesn't
    /// resolve.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        stack: &BunsenCatalogStack,
        contract: &str,
    ) -> BunsenResult<Self> {
        let contract_name = stack
            .expect_contract(&CatalogRef::parse(contract)?)?
            .name()
            .clone();
        let mut builds = Vec::new();
        for build in stack.builds_of_contract(&contract_name) {
            builds.push(stack.resolve_build(&build)?);
        }
        Ok(Self::of_builds(name, description, builds))
    }

    fn of_builds(
        name: impl Into<String>,
        description: impl Into<String>,
        builds: Vec<ResolvedBuild>,
    ) -> Self {
        let mut names = Vec::new();
        for (index, build) in builds.iter().enumerate() {
            names.push((build.name.clone(), index));
            names.extend(build.aliases.iter().map(|a| (a.clone(), index)));
        }
        Self {
            name: name.into(),
            description: description.into(),
            builds,
            names,
        }
    }

    /// The builds, in the catalog's order.
    pub fn builds(&self) -> &[ResolvedBuild] {
        &self.builds
    }

    /// The build `name` names, if any.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) when two builds
    /// match it and neither is named in full.
    pub fn find(
        &self,
        name: &str,
    ) -> BunsenResult<Option<&ResolvedBuild>> {
        let found = find_spelling(&self.names, name, |i| self.builds[i].name.to_string())?;
        Ok(found.map(|i| &self.builds[i]))
    }
}

impl BuildProvider for CatalogBuildProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn list(&self) -> Vec<ResolvedBuild> {
        self.builds.clone()
    }

    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<Option<ResolvedBuild>> {
        Ok(self.find(name)?.cloned())
    }
}

impl BuiltinCatalog {
    /// The built-in catalog's builds of `contract` (`qwen3-lm`; `qwen3-asr`
    /// for both of its variants), as the [`CATALOG`] provider's, described by
    /// the contract.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a contract
    /// `$contracts` doesn't have.
    pub fn provider(
        &self,
        contract: &str,
    ) -> BunsenResult<CatalogBuildProvider> {
        let contract_name = CatalogName::contracts_root().join(CatalogRef::parse(contract)?.path());
        let description = self.contract_description(&contract_name).ok_or_else(|| {
            catalog_error(
                CatalogRule::Lookup,
                format!("the built-in catalog has no contract `{contract}`"),
            )
        })?;
        let builds = self
            .builds()
            .iter()
            .filter(|build| {
                build
                    .contract
                    .as_ref()
                    .is_some_and(|c| c == &contract_name || c.is_within(&contract_name))
            })
            .cloned()
            .collect();
        Ok(CatalogBuildProvider::of_builds(
            CATALOG,
            description,
            builds,
        ))
    }

    /// [`provider`](Self::provider) of the process's built-in catalog, for a
    /// kit's default providers: the built-in catalog is compiled in and
    /// tested whole.
    ///
    /// # Panics
    /// When [`get`](Self::get) or [`provider`](Self::provider) fails, a bug
    /// in the built-in catalog.
    pub fn expect_provider(contract: &str) -> CatalogBuildProvider {
        Self::get()
            .and_then(|catalog| catalog.provider(contract))
            .unwrap_or_else(|e| panic!("the built-in catalog's `{contract}` builds: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        catalog::fixtures,
        resources::Source,
    };

    fn provider(contract: &str) -> CatalogBuildProvider {
        CatalogBuildProvider::new(
            "examples",
            "the example catalogs",
            &fixtures::example_stack().unwrap(),
            contract,
        )
        .unwrap()
    }

    /// A contract's builds are the provider's, named in full, with the
    /// catalog's implied tags.
    #[test]
    fn test_the_rows_are_the_contracts_builds() {
        let whisper = provider("whisper-asr");
        let names: Vec<String> = whisper
            .builds()
            .iter()
            .map(|b| b.name.to_string())
            .collect();
        assert!(
            names.iter().any(|n| n == "openai/whisper/large/large-v3"),
            "{names:?}"
        );
        assert!(names.iter().all(|n| n.starts_with("openai/whisper/")));
        let v3 = whisper.find("large-v3").unwrap().unwrap();
        assert_eq!(v3.tag("contract"), Some(&serde_json::json!("whisper-asr")));
        assert!(v3.presets.axis_of("translate").is_some());
    }

    /// A resource's sources: a place serves the file by its path, a URL
    /// names it, a local dir resolves from its templates.
    #[test]
    fn test_sources() {
        let whisper = provider("whisper-asr");
        let v1 = whisper.find("large-v1").unwrap().unwrap();
        let set = v1.resource_set();
        let checkpoint = set.get("checkpoint").unwrap();
        // Cached under the resource's full name.
        assert_eq!(checkpoint.name, "openai/whisper/large/large-v1/checkpoint");
        assert_eq!(checkpoint.dtype.as_deref(), Some("float16"));
        assert!(
            matches!(&checkpoint.sources[0], Source::LocalDir { name, .. } if name == "openai-whisper")
        );
        let Source::Url(url) = &checkpoint.sources[1] else {
            panic!("{:?}", checkpoint.sources)
        };
        assert!(url.ends_with("/large-v1.pt"), "{url}");
        assert!(set.get("vocabulary").is_some());
    }

    /// Lookups take the shorter spellings the catalog's do; a name no row
    /// has is "not here"; one two rows match is ambiguous (the built-in
    /// catalog's two `resnet34`s).
    #[test]
    fn test_lookups() {
        let whisper = provider("whisper-asr");
        for name in [
            "openai/whisper/large/large-v3",
            "openai/large-v3",
            "large-v3",
            "LARGE-V3",
        ] {
            assert_eq!(
                whisper.lookup(name).unwrap().unwrap().name.to_string(),
                "openai/whisper/large/large-v3",
                "{name}"
            );
        }
        assert!(whisper.lookup("qwen3-0.6b").unwrap().is_none());
        assert!(whisper.lookup("large").unwrap().is_none(), "a namespace");

        // The resnet builds are in the built-in catalog's `kits` layer.
        if !cfg!(feature = "kits") {
            return;
        }

        let resnet = BuiltinCatalog::get()
            .unwrap()
            .provider("resnet-classifier")
            .unwrap();
        crate::errors::testing::ErrorMatcher::kind(crate::errors::BunsenErrorKind::Lookup)
            .message_contains("`resnet34` is ambiguous: torchvision/resnet34, timm/resnet34")
            .assert_err(&resnet.lookup("resnet34"));
        assert_eq!(
            resnet
                .lookup("timm/resnet34")
                .unwrap()
                .unwrap()
                .name
                .to_string(),
            "timm/resnet34"
        );
    }
}
