//! # The pretrained factory
//!
//! The one object a caller holds for a kit: its providers, in search
//! order, and the dispatch of a spec across them. `provider:ref` goes to
//! that provider and nowhere else; a spec with no `provider:` is offered to
//! each provider that [answers bare
//! names](PretrainedProvider::answers_bare_names), in registration order, and
//! the first row wins. A provider's "not here" is `Ok(None)`; any error a
//! provider returns aborts the lookup, since a hub that cannot be reached is
//! not the same as a hub that has no such row.
//!
//! There is no process-wide registry. A kit ships a `default_{kit}_factory()`
//! over its compiled-in providers, and a caller that wants more builds its
//! own:
//!
//! ```rust,ignore
//! let factory = Arc::new(
//!     PretrainedFactory::new(WHISPER_KIT)
//!         .with_providers(default_whisper_providers())?
//!         .with_provider(Arc::new(my_hub))?, // answers `hub:org/repo`, lists nothing
//! );
//! ```
//!
//! A factory is built for one kit, the `<kit>` segment of the cache path;
//! a hook whose `KIT` differs is refused at load.

use alloc::{
    string::{
        String,
        ToString,
    },
    sync::Arc,
    vec::Vec,
};
use std::path::Path;

use super::{
    Pretrained,
    PretrainedProvider,
    PretrainedRef,
    ResourceMap,
    not_found,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A kit's providers, in search order, and the dispatch of a spec across
/// them.
#[derive(Debug)]
pub struct PretrainedFactory {
    kit: String,
    providers: Vec<Arc<dyn PretrainedProvider>>,
}

impl PretrainedFactory {
    /// An empty factory for `kit`: the `<kit>` segment of the cache path,
    /// and what a hook's `KIT` must match.
    pub fn new(kit: impl Into<String>) -> Self {
        Self {
            kit: kit.into(),
            providers: Vec::new(),
        }
    }

    /// The kit this factory serves.
    pub fn kit(&self) -> &str {
        &self.kit
    }

    /// Adds a provider at the end of the search order.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] when a provider of that name is registered
    /// already; names are what a spec's `provider:` selects, so two cannot
    /// share one.
    pub fn register(
        &mut self,
        provider: Arc<dyn PretrainedProvider>,
    ) -> BunsenResult<()> {
        if self.provider(provider.name()).is_some() {
            return Err(BunsenError::Invalid(alloc::format!(
                "{}: a provider named {:?} is registered already",
                self.kit,
                provider.name()
            )));
        }
        self.providers.push(provider);
        Ok(())
    }

    /// [`register`](Self::register), builder style.
    ///
    /// # Errors
    /// As [`register`](Self::register).
    pub fn with_provider(
        mut self,
        provider: Arc<dyn PretrainedProvider>,
    ) -> BunsenResult<Self> {
        self.register(provider)?;
        Ok(self)
    }

    /// [`register`](Self::register) each, in order, builder style.
    ///
    /// # Errors
    /// As [`register`](Self::register).
    pub fn with_providers(
        mut self,
        providers: impl IntoIterator<Item = Arc<dyn PretrainedProvider>>,
    ) -> BunsenResult<Self> {
        for provider in providers {
            self.register(provider)?;
        }
        Ok(self)
    }

    /// Removes the provider called `name`, and returns it.
    pub fn remove(
        &mut self,
        name: &str,
    ) -> Option<Arc<dyn PretrainedProvider>> {
        let at = self.providers.iter().position(|p| p.name() == name)?;
        Some(self.providers.remove(at))
    }

    /// The providers, in search order.
    pub fn providers(&self) -> &[Arc<dyn PretrainedProvider>] {
        &self.providers
    }

    /// The provider called `name`.
    pub fn provider(
        &self,
        name: &str,
    ) -> Option<&Arc<dyn PretrainedProvider>> {
        self.providers.iter().find(|p| p.name() == name)
    }

    /// Every listed row with its provider's name, in registration then
    /// listing order.
    pub fn list(&self) -> Vec<(String, Pretrained)> {
        self.providers
            .iter()
            .flat_map(|provider| {
                let name = provider.name().to_string();
                provider
                    .list()
                    .into_iter()
                    .map(move |row| (name.clone(), row))
            })
            .collect()
    }

    /// Every listed id, `provider:ref`: the "did you mean" list.
    pub fn ids(&self) -> Vec<String> {
        self.providers.iter().flat_map(|p| p.ids()).collect()
    }

    /// Every listed row that instantiates `prefab`, with its provider's
    /// name.
    pub fn for_prefab(
        &self,
        prefab: &str,
    ) -> Vec<(String, Pretrained)> {
        self.providers
            .iter()
            .flat_map(|provider| {
                let name = provider.name().to_string();
                provider
                    .for_prefab(prefab)
                    .into_iter()
                    .map(move |row| (name.clone(), row))
            })
            .collect()
    }

    /// The row `spec` names, with its provider's name.
    ///
    /// `provider:ref` asks that provider alone. Anything else is offered to
    /// the providers that answer bare names, in order; the first row wins.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`] naming what there is: the
    /// provider's ids after a qualified miss, the providers after an
    /// unknown `provider:`, every id otherwise. A provider's own error, as
    /// it returned it.
    pub fn lookup(
        &self,
        spec: &str,
    ) -> BunsenResult<(String, Pretrained)> {
        match self.find(spec)? {
            Some(found) => Ok(found),
            None => Err(self.not_found(spec)),
        }
    }

    /// What `spec` refers to: the row it names, as [`lookup`](Self::lookup)
    /// finds it, or else, when `given_key` is `Some` and `spec` is a path to
    /// an existing file, a one-resource map under that key, the key the
    /// kit reads a bare checkpoint by. The index wins over the file
    /// system; `None` refuses a path.
    ///
    /// # Errors
    /// As [`lookup`](Self::lookup), the message noting that `spec` is not
    /// a file either when a path would have been taken.
    pub fn resolve(
        &self,
        spec: &str,
        given_key: Option<&str>,
    ) -> BunsenResult<PretrainedRef> {
        if let Some((provider, pretrained)) = self.find(spec)? {
            return Ok(PretrainedRef::Named {
                provider,
                pretrained,
            });
        }
        let Some(key) = given_key else {
            return Err(self.not_found(spec));
        };
        let path = Path::new(spec);
        if path.is_file() {
            return Ok(PretrainedRef::Given(ResourceMap::given(spec, key, path)));
        }
        Err(match self.not_found(spec) {
            BunsenError::ResourceNotFound(m) => {
                BunsenError::ResourceNotFound(alloc::format!("{m}; and {spec:?} is not a file"))
            }
            other => other,
        })
    }

    /// The dispatch behind [`lookup`](Self::lookup): `Ok(None)` when no
    /// provider has the row.
    pub(crate) fn find(
        &self,
        spec: &str,
    ) -> BunsenResult<Option<(String, Pretrained)>> {
        if let Some((head, rest)) = spec.split_once(':')
            && let Some(provider) = self.provider(head)
        {
            return Ok(provider
                .lookup(rest)?
                .map(|row| (provider.name().to_string(), row)));
        }
        for provider in &self.providers {
            if !provider.answers_bare_names() {
                continue;
            }
            if let Some(row) = provider.lookup(spec)? {
                return Ok(Some((provider.name().to_string(), row)));
            }
        }
        Ok(None)
    }

    /// The error for a spec no provider has.
    pub(crate) fn not_found(
        &self,
        spec: &str,
    ) -> BunsenError {
        if let Some((head, rest)) = spec.split_once(':') {
            if let Some(provider) = self.provider(head) {
                let ids = provider.ids();
                let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
                return not_found(Some(provider.name()), "pretrained", rest, &ids);
            }
            let names: Vec<&str> = self.providers.iter().map(|p| p.name()).collect();
            return not_found(Some(&self.kit), "provider", head, &names);
        }
        let ids = self.ids();
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        not_found(Some(&self.kit), "pretrained", spec, &ids)
    }
}

/// Providers with the shape of things the compiled-in tables are not: a
/// hub that answers a ref but lists nothing, and one that fails.
#[cfg(test)]
pub(crate) mod testing {
    use alloc::{
        format,
        vec,
    };
    use core::sync::atomic::{
        AtomicUsize,
        Ordering,
    };

    use super::*;
    use crate::data::pretrained::{
        Resource,
        ResourceMap,
        Source,
    };

    /// A provider that can look up but not list: the shape of a hub
    /// wrapper. Answers `hub:<org>/<repo>` with a two-file map of URLs, and
    /// counts how often it was asked, so a test can pin that a bare name
    /// never reached it.
    #[derive(Debug, Default)]
    pub struct ListsNothing {
        /// How many lookups reached this provider.
        pub lookups: AtomicUsize,
    }

    impl ListsNothing {
        /// How many lookups reached this provider.
        pub fn lookups(&self) -> usize {
            self.lookups.load(Ordering::SeqCst)
        }
    }

    impl PretrainedProvider for ListsNothing {
        fn name(&self) -> &str {
            "hub"
        }

        fn description(&self) -> &str {
            "answers `hub:<org>/<repo>`; lists nothing"
        }

        fn origin(&self) -> Option<&str> {
            Some("https://hub.example")
        }

        fn list(&self) -> Vec<Pretrained> {
            Vec::new()
        }

        fn lookup(
            &self,
            name: &str,
        ) -> BunsenResult<Option<Pretrained>> {
            self.lookups.fetch_add(1, Ordering::SeqCst);
            let Some((org, repo)) = name.split_once('/') else {
                return Ok(None);
            };
            let base = format!("https://hub.example/{org}/{repo}/resolve/main");
            let file = |key: &str, file: &str, kind: &str| Resource {
                key: key.to_string(),
                file: file.to_string(),
                sha256: None,
                kind: Some(kind.to_string()),
                namespace: "hub".to_string(),
                sources: vec![Source::Url(format!("{base}/{file}"))],
            };
            let resources = ResourceMap::new(name)
                .with_resource(file("checkpoint", "model.safetensors", "safetensors"))
                .with_resource(file("config", "config.json", "json"));
            Ok(Some(Pretrained {
                name: name.to_string(),
                aliases: Vec::new(),
                description: format!("{org}/{repo}, from the hub"),
                license: None,
                origin: Some(base),
                prefab: None,
                resources,
            }))
        }

        fn answers_bare_names(&self) -> bool {
            false
        }
    }

    /// A provider whose lookups fail: a hub that cannot be reached.
    #[derive(Debug, Default)]
    pub struct Failing;

    impl PretrainedProvider for Failing {
        fn name(&self) -> &str {
            "failing"
        }

        fn description(&self) -> &str {
            "cannot be reached"
        }

        fn list(&self) -> Vec<Pretrained> {
            Vec::new()
        }

        fn lookup(
            &self,
            _name: &str,
        ) -> BunsenResult<Option<Pretrained>> {
            Err(BunsenError::External("failing: unreachable".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::{
        testing::{
            Failing,
            ListsNothing,
        },
        *,
    };
    use crate::data::pretrained::{
        PretrainedGroup,
        PretrainedTable,
        Resource,
        ResourceMap,
        Source,
        WELL_KNOWN,
    };

    fn row(
        name: &str,
        aliases: &[&str],
        prefab: &str,
    ) -> Pretrained {
        Pretrained {
            name: name.to_string(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
            description: format!("row {name}"),
            license: None,
            origin: None,
            prefab: Some(prefab.to_string()),
            resources: ResourceMap::new(name).with_resource(Resource {
                key: "checkpoint".to_string(),
                file: format!("{name}.pt"),
                sha256: None,
                kind: Some("pytorch".to_string()),
                namespace: "t".to_string(),
                sources: vec![Source::Url(format!("https://t.example/{name}.pt"))],
            }),
        }
    }

    fn group(
        name: &str,
        items: Vec<Pretrained>,
    ) -> PretrainedGroup {
        PretrainedGroup {
            name: name.to_string(),
            description: format!("group {name}"),
            license: None,
            origin: None,
            items,
        }
    }

    fn table(
        name: &str,
        groups: Vec<PretrainedGroup>,
    ) -> Arc<dyn PretrainedProvider> {
        Arc::new(PretrainedTable {
            name: name.to_string(),
            description: format!("table {name}"),
            groups,
        })
    }

    /// `well-known` with groups `a` (small, large-v1, large-v2) and `b`
    /// (small, tiny).
    fn well_known() -> Arc<dyn PretrainedProvider> {
        table(
            WELL_KNOWN,
            vec![
                group(
                    "a",
                    vec![
                        row("small", &["s"], "small"),
                        row("large-v1", &[], "large"),
                        row("large-v2", &["large"], "large"),
                    ],
                ),
                group(
                    "b",
                    vec![row("small", &[], "small"), row("tiny", &[], "tiny")],
                ),
            ],
        )
    }

    fn factory() -> PretrainedFactory {
        PretrainedFactory::new("kit")
            .with_providers([well_known(), Arc::new(ListsNothing::default()) as _])
            .unwrap()
    }

    #[test]
    fn test_register_refuses_a_duplicate_name_and_remove_frees_it() {
        let mut factory = PretrainedFactory::new("kit")
            .with_provider(well_known())
            .unwrap();
        assert_eq!(factory.kit(), "kit");
        assert_eq!(factory.providers().len(), 1);
        assert!(factory.provider(WELL_KNOWN).is_some());

        let err = factory.register(well_known()).unwrap_err();
        assert!(
            matches!(&err, BunsenError::Invalid(m) if m.contains("\"well-known\" is registered already")),
            "{err}"
        );
        assert_eq!(factory.providers().len(), 1);

        assert!(factory.remove("nobody").is_none());
        let removed = factory.remove(WELL_KNOWN).unwrap();
        assert_eq!(removed.name(), WELL_KNOWN);
        assert!(factory.providers().is_empty());
        factory.register(well_known()).unwrap();
        assert!(format!("{factory:?}").contains("PretrainedTable"));

        // The builder form surfaces the same error through `?`.
        assert!(
            PretrainedFactory::new("kit")
                .with_providers([well_known(), well_known()])
                .is_err()
        );
    }

    /// `provider:ref` asks that provider and nothing else: a ref the
    /// provider has, a ref it does not, a provider that is not there.
    #[test]
    fn test_a_qualified_spec_goes_to_one_provider() {
        let factory = factory();

        let (provider, row) = factory.lookup("well-known:b/tiny").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.as_str()),
            ("well-known", "b/tiny")
        );
        let (provider, row) = factory.lookup("well-known:large").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.as_str()),
            ("well-known", "a/large-v2")
        );

        match factory.lookup("well-known:a/tiny") {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert!(m.starts_with("well-known: no pretrained \"a/tiny\""), "{m}");
                assert!(m.contains("well-known:a/small"), "{m}");
            }
            other => panic!("{other:?}"),
        }
        match factory.lookup("nobody:a/small") {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert!(m.starts_with("kit: no provider \"nobody\""), "{m}");
                assert!(m.contains("well-known, hub"), "{m}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// A bare spec is offered to the providers in order; the first row
    /// wins, and a provider that declines bare names is skipped.
    #[test]
    fn test_a_bare_spec_searches_in_order_and_skips_decliners() {
        let second = table(
            "second",
            vec![group(
                "a",
                vec![row("small", &[], "small"), row("only-here", &[], "x")],
            )],
        );
        let factory = PretrainedFactory::new("kit")
            .with_providers([Arc::new(ListsNothing::default()) as _, well_known(), second])
            .unwrap();

        let (provider, row) = factory.lookup("small").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.as_str()),
            ("well-known", "a/small")
        );
        let (provider, row) = factory.lookup("only-here").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.as_str()),
            ("second", "a/only-here")
        );
        let (provider, row) = factory.lookup("second:a/small").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.as_str()),
            ("second", "a/small")
        );
        let (_, row) = factory.lookup("tiny").unwrap();
        assert_eq!(row.name, "b/tiny");
        let (_, row) = factory.lookup("a/s").unwrap();
        assert_eq!(row.name, "a/small");

        match factory.lookup("gigantic") {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert!(m.starts_with("kit: no pretrained \"gigantic\""), "{m}");
                assert!(m.contains("well-known:a/small"), "{m}");
                assert!(m.contains("second:a/only-here"), "{m}");
            }
            other => panic!("{other:?}"),
        }
        let hub = factory.provider("hub").unwrap();
        assert!(!hub.answers_bare_names());
    }

    /// A provider that lists nothing answers only its qualified refs; a
    /// bare name never reaches it.
    #[test]
    fn test_a_lookup_only_provider_answers_qualified_names_only() {
        let hub = Arc::new(ListsNothing::default());
        let factory = PretrainedFactory::new("kit")
            .with_providers([well_known(), hub.clone() as _])
            .unwrap();

        assert_eq!(factory.list().len(), 5, "the hub lists nothing");
        assert_eq!(factory.ids().len(), 5);
        assert!(factory.ids().iter().all(|id| id.starts_with("well-known:")));

        let (provider, row) = factory.lookup("hub:openai/whisper-base").unwrap();
        assert_eq!(provider, "hub");
        assert_eq!(row.name, "openai/whisper-base");
        assert_eq!(row.resources.keys(), ["checkpoint", "config"]);
        assert_eq!(
            row.resources.get("checkpoint").unwrap().urls(),
            ["https://hub.example/openai/whisper-base/resolve/main/model.safetensors"]
        );
        assert_eq!(hub.lookups(), 1);

        assert!(matches!(
            factory.lookup("whisper-base"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        assert_eq!(hub.lookups(), 1, "a bare name never reached the hub");

        match factory.lookup("hub:nonsense") {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert_eq!(m, "hub: no pretrained \"nonsense\"; there are: (none)");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(hub.lookups(), 2);
    }

    /// A provider's own error is the lookup's error, qualified or bare: it
    /// is not "not here".
    #[test]
    fn test_a_providers_error_aborts_the_lookup() {
        let factory = PretrainedFactory::new("kit")
            .with_providers([Arc::new(Failing) as _, well_known()])
            .unwrap();
        assert!(matches!(
            factory.lookup("failing:x"),
            Err(BunsenError::External(_))
        ));
        assert!(
            matches!(factory.lookup("small"), Err(BunsenError::External(_))),
            "the failing provider answers bare names, and is asked first"
        );
        assert_eq!(
            factory.lookup("well-known:small").unwrap().1.name,
            "a/small"
        );
    }

    /// A spec resolves to the row it names, qualified, bare or by alias;
    /// failing that, to a file under the given key; and the index is tried
    /// before the file system.
    #[test]
    fn test_resolve_names_aliases_and_paths() {
        let factory = factory();

        let model = factory
            .resolve("well-known:a/small", Some("checkpoint"))
            .unwrap();
        assert_eq!(model.id(), "well-known:a/small");
        let model = factory.resolve("s", Some("checkpoint")).unwrap();
        assert_eq!(model.id(), "well-known:a/small");
        assert_eq!(
            model.named().map(|(p, row)| (p, row.name.as_str())),
            Some(("well-known", "a/small"))
        );
        assert_eq!(model.to_map().name, "well-known:a/small");
        let model = factory.resolve("hub:x/y", None).unwrap();
        assert_eq!(model.id(), "hub:x/y");

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("ckpt.pt");
        std::fs::write(&file, b"x").unwrap();
        let spec = file.to_str().unwrap();
        let model = factory.resolve(spec, Some("checkpoint")).unwrap();
        assert_eq!(model.id(), spec);
        assert!(model.named().is_none());
        let map = model.to_map();
        assert_eq!(map.keys(), ["checkpoint"]);
        assert_eq!(map.get("checkpoint").unwrap().file, "ckpt.pt");

        match factory.resolve(spec, None) {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert!(!m.contains("not a file"), "a path was never an option: {m}");
            }
            other => panic!("{other:?}"),
        }
        match factory.resolve("/no/such/file.pt", Some("checkpoint")) {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert!(m.contains("well-known:a/small"), "{m}");
                assert!(m.ends_with("\"/no/such/file.pt\" is not a file"), "{m}");
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            factory.resolve("well-known:a/gigantic", Some("checkpoint")),
            Err(BunsenError::ResourceNotFound(_))
        ));

        // A row named like an existing file is the row: the index first.
        let shadow = table("shadow", vec![group("g", vec![row(spec, &[], "x")])]);
        let factory = PretrainedFactory::new("kit").with_provider(shadow).unwrap();
        let model = factory.resolve(spec, Some("checkpoint")).unwrap();
        assert!(model.named().is_some());
        assert_eq!(model.id(), format!("shadow:g/{spec}"));
    }

    #[test]
    fn test_list_ids_and_for_prefab_cover_every_provider() {
        let factory = factory();
        let listed: Vec<String> = factory
            .list()
            .iter()
            .map(|(p, row)| format!("{p}:{}", row.name))
            .collect();
        assert_eq!(listed, factory.ids());
        assert_eq!(
            factory.ids(),
            [
                "well-known:a/small",
                "well-known:a/large-v1",
                "well-known:a/large-v2",
                "well-known:b/small",
                "well-known:b/tiny",
            ]
        );
        let small: Vec<String> = factory
            .for_prefab("small")
            .iter()
            .map(|(p, row)| format!("{p}:{}", row.name))
            .collect();
        assert_eq!(small, ["well-known:a/small", "well-known:b/small"]);
        assert!(factory.for_prefab("nothing").is_empty());
    }
}
