//! # The pretrained factory
//!
//! The one object a caller holds for a kit: its providers, in search
//! order. The interface is names and a listing:
//! `factory.load::<B>("[provider:]name", &cache, device)` is the whole
//! pathway, and [`resolve`](PretrainedFactory::resolve) is its index half,
//! a [`Deferred`] model that carries the hook its map calls for, so that a
//! caller overlays a row before loading it and never builds or passes a
//! hook. A file on disk is not the factory's business; that is a given
//! [`ResourceMap`] through [`Deferred::from_map`].
//!
//! Dispatch: `provider:ref` goes to that provider and nowhere else; a spec
//! with no `provider:` is offered to each provider that [answers bare
//! names](PretrainedProvider::answers_bare_names), in registration order,
//! and the first row wins. A provider's "not here" is `Ok(None)`; any error
//! a provider returns aborts the lookup, since a hub that cannot be reached
//! is not the same as a hub that has no such row. `resolve` goes through
//! the cache, [`PretrainedProvider::resolve`], so a hub can ask what a repo
//! holds once and answer from the cache after;
//! [`lookup`](PretrainedFactory::lookup) is the index alone, which a table
//! answers and a hub does not.
//!
//! There is no process-wide registry. A kit ships a `default_{kit}_factory()`
//! over its compiled-in providers, and a caller that wants more builds on
//! it:
//!
//! ```rust,ignore
//! let factory = default_whisper_factory()?
//!     .with_provider(Arc::new(my_hub))?; // answers `hub:org/repo`, lists nothing
//! let bundle = factory.load_bundle::<B>("openai/base", &cache, &device)?;
//! ```

use alloc::{
    string::{
        String,
        ToString,
    },
    sync::Arc,
    vec::Vec,
};
use core::marker::PhantomData;

use burn::prelude::Backend;

#[cfg(doc)]
use super::ResourceMap;
use super::{
    Construct,
    Deferred,
    Loaded,
    Pretrained,
    PretrainedCache,
    PretrainedProvider,
    PretrainedRef,
    not_found,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A kit's providers, in search order, resolving names to [`Deferred`]
/// models built through the kit's hook `H`.
#[derive(Debug)]
pub struct PretrainedFactory<H: Construct> {
    providers: Vec<Arc<dyn PretrainedProvider>>,
    hook: PhantomData<H>,
}

impl<H: Construct> Clone for PretrainedFactory<H> {
    fn clone(&self) -> Self {
        Self {
            providers: self.providers.clone(),
            hook: PhantomData,
        }
    }
}

impl<H: Construct> Default for PretrainedFactory<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: Construct> PretrainedFactory<H> {
    /// A factory with no providers yet.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            hook: PhantomData,
        }
    }

    /// The kit this factory serves: the hook's, the `<kit>` segment of the
    /// cache path.
    pub fn kit(&self) -> &'static str {
        H::KIT
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
                H::KIT,
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
    /// The index only: `provider:ref` asks that provider alone. Anything
    /// else is offered to the providers that answer bare names, in order;
    /// the first row wins. A provider that answers only through a cache, a
    /// hub, answers this with its own error.
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
        match self.find(spec, |provider, name| provider.lookup(name))? {
            Some(found) => Ok(found),
            None => Err(self.not_found(spec)),
        }
    }

    /// The row `spec` names, dispatched as [`lookup`](Self::lookup) is but
    /// through each provider's [`resolve`](PretrainedProvider::resolve)
    /// with `cache` to hand, as a [`Deferred`] model with the hook its map
    /// calls for: the index half of [`load`](Self::load), for a caller
    /// that overlays the row before loading it. A table answers from its
    /// rows; a hub asks what the repo holds, once, and keeps the answer in
    /// the cache.
    ///
    /// # Errors
    /// As [`lookup`](Self::lookup) and [`Deferred::new`].
    pub fn resolve(
        &self,
        spec: &str,
        cache: &PretrainedCache,
    ) -> BunsenResult<Deferred<H>> {
        let found = self.find(spec, |provider, name| provider.resolve(name, H::KIT, cache))?;
        let (provider, pretrained) = match found {
            Some(found) => found,
            None => return Err(self.not_found(spec)),
        };
        Deferred::new(PretrainedRef::Named {
            provider,
            pretrained,
        })
    }

    /// A name to what the kit builds: [`resolve`](Self::resolve), then
    /// [`Deferred::load`]. The whole pathway.
    ///
    /// # Errors
    /// As [`resolve`](Self::resolve) and [`Deferred::load`].
    pub fn load<B: Backend>(
        &self,
        spec: &str,
        cache: &PretrainedCache,
        device: &B::Device,
    ) -> BunsenResult<Loaded<H::Built<B>>> {
        self.resolve(spec, cache)?.load::<B>(cache, device)
    }

    /// The dispatch behind [`lookup`](Self::lookup) and
    /// [`resolve`](Self::resolve): `ask` is what each provider is asked,
    /// its `lookup` or its `resolve`; `Ok(None)` when no provider has the
    /// row.
    pub(crate) fn find(
        &self,
        spec: &str,
        ask: impl Fn(&dyn PretrainedProvider, &str) -> BunsenResult<Option<Pretrained>>,
    ) -> BunsenResult<Option<(String, Pretrained)>> {
        if let Some((head, rest)) = spec.split_once(':')
            && let Some(provider) = self.provider(head)
        {
            return Ok(ask(provider.as_ref(), rest)?.map(|row| (provider.name().to_string(), row)));
        }
        for provider in &self.providers {
            if !provider.answers_bare_names() {
                continue;
            }
            if let Some(row) = ask(provider.as_ref(), spec)? {
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
            return not_found(Some(H::KIT), "provider", head, &names);
        }
        let ids = self.ids();
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        not_found(Some(H::KIT), "pretrained", spec, &ids)
    }
}

/// Providers and a hook with the shape of things the compiled-in tables are
/// not: a hub that answers a ref but lists nothing, one that fails, and a
/// hook that builds a path.
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
    use std::path::PathBuf;

    use super::*;
    use crate::data::pretrained::{
        LoadedResources,
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

    /// A hook for kit `kit` that builds the checkpoint's path, whatever the
    /// map says.
    #[derive(Clone, Debug, Default)]
    pub struct CheckpointPath;

    impl Construct for CheckpointPath {
        type Built<B: Backend> = PathBuf;

        const KIT: &'static str = "kit";

        fn for_map(_map: &ResourceMap) -> BunsenResult<Self> {
            Ok(Self)
        }

        fn construct<B: Backend>(
            &self,
            _model: &PretrainedRef,
            loaded: &LoadedResources,
            _device: &B::Device,
        ) -> BunsenResult<Arc<PathBuf>> {
            Ok(Arc::new(loaded.expect("checkpoint")?.to_path_buf()))
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use std::path::Path;

    use super::{
        testing::{
            CheckpointPath,
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

    /// A row whose one resource is a file on disk already.
    fn local_row(
        name: &str,
        path: &Path,
    ) -> Pretrained {
        Pretrained {
            name: name.to_string(),
            aliases: Vec::new(),
            description: format!("row {name}, on disk"),
            license: None,
            origin: None,
            prefab: None,
            resources: ResourceMap::given(name, "checkpoint", path),
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

    /// An offline cache rooted under `dir`: nothing is fetched, nothing
    /// reports.
    fn offline_cache(dir: &std::path::Path) -> PretrainedCache {
        use crate::data::{
            cache::BunsenDiskCacheOptions,
            pretrained::PretrainedCacheOptions,
        };
        PretrainedCache::new(
            PretrainedCacheOptions::default()
                .with_disk(
                    BunsenDiskCacheOptions::default()
                        .with_cache_dir(Some(dir.join("cache")))
                        .without_transfer_observers(),
                )
                .with_offline(true),
        )
        .unwrap()
    }

    fn factory() -> PretrainedFactory<CheckpointPath> {
        PretrainedFactory::new()
            .with_providers([well_known(), Arc::new(ListsNothing::default()) as _])
            .unwrap()
    }

    #[test]
    fn test_register_refuses_a_duplicate_name_and_remove_frees_it() {
        let mut factory = PretrainedFactory::<CheckpointPath>::new()
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
        assert_eq!(factory.clone().providers().len(), 1);
        assert!(
            PretrainedFactory::<CheckpointPath>::default()
                .providers()
                .is_empty()
        );

        // The builder form surfaces the same error through `?`.
        assert!(
            PretrainedFactory::<CheckpointPath>::new()
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
        let factory = PretrainedFactory::<CheckpointPath>::new()
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
        let factory = PretrainedFactory::<CheckpointPath>::new()
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
        let factory = PretrainedFactory::<CheckpointPath>::new()
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

    /// A spec resolves to the row it names, qualified, bare or by alias,
    /// as a deferred model with a hook, and to nothing else: a file on disk
    /// is not the factory's business.
    #[test]
    fn test_resolve_names_and_aliases_only() {
        let factory = factory();
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());

        let model = factory.resolve("well-known:a/small", &cache).unwrap();
        assert_eq!(model.id(), "well-known:a/small");
        let model = factory.resolve("s", &cache).unwrap();
        assert_eq!(model.id(), "well-known:a/small");
        assert_eq!(
            model.model.named().map(|(p, row)| (p, row.name.as_str())),
            Some(("well-known", "a/small"))
        );
        assert_eq!(model.to_map().name, "well-known:a/small");
        assert!(format!("{:?}", model.hook).contains("CheckpointPath"));
        let model = factory.resolve("hub:x/y", &cache).unwrap();
        assert_eq!(model.id(), "hub:x/y");

        let file = dir.path().join("ckpt.pt");
        std::fs::write(&file, b"x").unwrap();
        match factory.resolve(file.to_str().unwrap(), &cache) {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert!(m.contains("well-known:a/small"), "{m}");
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            factory.resolve("well-known:a/gigantic", &cache),
            Err(BunsenError::ResourceNotFound(_))
        ));
    }

    /// `load` runs the whole pathway through the hook the map called for;
    /// a ref the caller overlays first goes through the same hook.
    #[test]
    fn test_load_goes_through_the_hook() {
        use crate::{
            data::pretrained::Provenance,
            support::testing::{
                CpuBackend,
                default_device,
            },
        };

        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());
        let file = dir.path().join("ckpt.pt");
        std::fs::write(&file, b"x").unwrap();
        let on_disk = table("disk", vec![group("l", vec![local_row("ckpt", &file)])]);
        let factory = PretrainedFactory::<CheckpointPath>::new()
            .with_providers([well_known(), on_disk])
            .unwrap();

        let loaded = factory
            .load::<CpuBackend>("disk:l/ckpt", &cache, &default_device())
            .unwrap();
        assert_eq!(*loaded.handle, file);
        assert_eq!(loaded.name, "disk:l/ckpt");
        assert_eq!(
            loaded.resources.get("checkpoint").unwrap().provenance,
            Provenance::LocalDir
        );

        let other = dir.path().join("other.pt");
        std::fs::write(&other, b"y").unwrap();
        let loaded = factory
            .resolve("ckpt", &cache)
            .unwrap()
            .with_overlay(ResourceMap::given("mine", "checkpoint", &other))
            .unwrap()
            .load::<CpuBackend>(&cache, &default_device())
            .unwrap();
        assert_eq!(*loaded.handle, other);

        assert!(
            matches!(
                factory.load::<CpuBackend>("well-known:a/small", &cache, &default_device()),
                Err(BunsenError::ResourceNotFound(_))
            ),
            "offline, and nothing local"
        );
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
