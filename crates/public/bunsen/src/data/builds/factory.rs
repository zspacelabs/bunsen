//! The pretrained factory: a kit's providers, and the dispatch of a spec.

use alloc::{
    string::{
        String,
        ToString,
    },
    sync::Arc,
    vec::Vec,
};
use core::marker::PhantomData;

use burn::tensor::Device;
use log::Level;
use serde::{
    Serialize,
    de::DeserializeOwned,
};

#[cfg(doc)]
use crate::data::resources::ResourceSet;
use crate::{
    data::{
        builds::{
            ConstructInit,
            Deferred,
            Loaded,
            LookupExpr,
            Pretrained,
            PretrainedProvider,
            PretrainedRef,
        },
        catalog::presets::PresetSelection,
        formats::LoadPolicy,
        not_found,
        pretrained::tags::KIT,
        resources::ResourceCache,
    },
    errors::{
        BunsenError,
        BunsenErrorKind,
        BunsenResult,
        LookupError,
        ResultContext,
    },
    kits::support::model_config::{
        ResolvedPresets,
        resolve_presets,
    },
};

/// A kit's providers, in search order, resolving names to [`Deferred`]
/// models built through the kit's hook `H`.
///
/// The one object a caller holds for a kit. Its interface is names and a
/// listing. [`load`](Self::load)`("[provider:]name", &cache, &device)`
/// is the whole pathway, from a name to a [`Loaded`] model.
/// [`resolve`](Self::resolve) is its index half: a [`Deferred`] model that
/// carries the hook its resource set calls for, so that a caller can overlay
/// a row ([`Deferred::with_overlay`]) before loading it, and never builds or
/// passes a hook. [`list`](Self::list), [`ids`](Self::ids) and
/// [`for_prefab`](Self::for_prefab) answer a listing. A file on disk is
/// not the factory's business: that is a given [`ResourceSet`], through
/// [`Deferred::from_resource_set`].
///
/// # Lookup expressions
///
/// A spec is a [`LookupExpr`]: `[provider:]name`, then an optional box of
/// presets and an optional YAML patch, `name[preset, …]{patch}`. Only the
/// name is dispatched. The box and the patch are checked against the row
/// found before anything is fetched: a kit whose hook does not resolve
/// presets ([`ConstructInit::RESOLVES_PRESETS`]) refuses them, and one that
/// does checks the box against the row's presets
/// ([`ResolvedPresetSet::check_selection`](crate::data::catalog::presets::ResolvedPresetSet::check_selection)).
/// The selection rides on the ref ([`PretrainedRef::selection`]) to the
/// hook, which resolves its config through it.
///
/// A table compares names ignoring ASCII case unless it is
/// [`case_sensitive`](crate::data::builds::PretrainedTable::case_sensitive):
/// `Qwen/Qwen3-0.6B` finds `qwen/qwen3-0.6b`. A hub compares as it is asked,
/// since its names are paths in URLs.
///
/// # Dispatch
///
/// - `provider:ref`, where `provider` names a registered provider, goes to that
///   provider and nowhere else.
/// - Any other spec, with no `:` or with a prefix that names no provider, is
///   offered whole to each provider that [answers bare
///   names](PretrainedProvider::answers_bare_names), in registration order, and
///   the first row wins. A hub such as
///   [`HfProvider`](crate::data::builds::HfProvider) declines bare names, so it
///   is only reached qualified.
/// - A provider's "not here" is `Ok(None)`, and the search moves on. Any error
///   a provider returns aborts the lookup: a hub that cannot be reached is not
///   the same as a hub that has no such row.
/// - A spec no provider has is a [`Lookup`](BunsenErrorKind::Lookup) error,
///   with a [`LookupError`] naming what there is.
///
/// [`resolve`](Self::resolve) asks each provider through
/// [`PretrainedProvider::resolve`], with the cache to hand, so a hub can
/// ask what a repo holds once and answer from the cache after.
/// [`lookup`](Self::lookup) asks [`PretrainedProvider::lookup`], the index
/// alone, which a table answers and a hub refuses.
///
/// # No registry
///
/// There is no process-wide registry. A kit ships a
/// `default_{kit}_factory()` over its compiled-in providers
/// ([`default_resnet_factory`](crate::kits::images::resnet::pretrained::default_resnet_factory),
/// [`default_whisper_factory`](crate::kits::speech::asr::whisper::pretrained::default_whisper_factory),
/// [`default_silero_factory`](crate::kits::speech::vad::silero_vad::pretrained::default_silero_factory)),
/// and a caller that wants more builds on it with
/// [`with_provider`](Self::with_provider), or takes one away with
/// [`remove`](Self::remove). Names are unique within a factory, since a
/// spec's `provider:` selects by name.
///
/// # Example
///
/// Whisper's factory dispatches without loading anything:
///
/// ```rust
/// # #[cfg(all(feature = "speech", feature = "store"))] {
/// use bunsen::kits::speech::asr::whisper::pretrained::default_whisper_factory;
///
/// let factory = default_whisper_factory()?;
///
/// // A bare name is offered to the providers that answer bare names, in
/// // order: the well-known table has it.
/// let (provider, row) = factory.lookup("tiny")?;
/// assert_eq!(provider, "well-known");
/// assert_eq!(row.name, "openai/whisper/tiny");
///
/// // `hf:` goes to the Hugging Face provider alone, which answers only
/// // through a cache: its error aborts the lookup.
/// assert!(factory.lookup("hf:openai/whisper-tiny").is_err());
///
/// // An unknown provider names the ones there are.
/// let err = factory.lookup("nope:tiny").unwrap_err();
/// assert!(err.to_string().contains("no provider \"nope\""), "{err}");
/// # }
/// # Ok::<(), bunsen::errors::BunsenError>(())
/// ```
#[derive(Debug)]
pub struct PretrainedFactory<H: ConstructInit> {
    providers: Vec<Arc<dyn PretrainedProvider>>,
    config: PhantomData<H>,
}

impl<H: ConstructInit> Clone for PretrainedFactory<H> {
    fn clone(&self) -> Self {
        Self {
            providers: self.providers.clone(),
            config: PhantomData,
        }
    }
}

impl<H: ConstructInit> Default for PretrainedFactory<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: ConstructInit> PretrainedFactory<H> {
    /// A factory with no providers yet.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            config: PhantomData,
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
    /// [`Illegal`](BunsenErrorKind::Illegal), with a duplicate
    /// [`LookupError`], when a provider of that name is registered already;
    /// names are what a spec's `provider:` selects, so two cannot share one.
    pub fn register(
        &mut self,
        provider: Arc<dyn PretrainedProvider>,
    ) -> BunsenResult<()> {
        if self.provider(provider.name()).is_some() {
            return Err(BunsenError::from_cause(
                BunsenErrorKind::Illegal,
                LookupError::duplicate("provider", provider.name()),
            )
            .context(H::KIT));
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
                    .map(move |row| (name.clone(), Self::with_kit(row)))
            })
            .collect()
    }

    /// `row` with this factory's kit as its [`KIT`] tag, when it has none:
    /// every row the factory hands out says which kit it is, whichever
    /// provider it came from.
    fn with_kit(mut row: Pretrained) -> Pretrained {
        if !row.tags.contains_key(KIT) {
            row.tags.insert(KIT, H::KIT);
        }
        row
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
                    .map(move |row| (name.clone(), Self::with_kit(row)))
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
    /// [`Lookup`](BunsenErrorKind::Lookup), with a [`LookupError`] naming
    /// what there is, under a frame naming where it looked: the provider's
    /// ids after a qualified miss, the providers after an unknown
    /// `provider:`, every id otherwise. A provider's own error, as it
    /// returned it.
    pub fn lookup(
        &self,
        spec: &str,
    ) -> BunsenResult<(String, Pretrained)> {
        let expr = LookupExpr::parse(spec)?;
        let (provider, row) = match self.find(&expr.name, |provider, name| provider.lookup(name))? {
            Some(found) => found,
            None => return Err(self.not_found(&expr.name)),
        };
        Self::check_selection(&row, &expr.selection)?;
        Ok((provider, row))
    }

    /// The config `spec` resolves `base` to through its row's presets, as
    /// its box and patch ask, found by [`lookup`](Self::lookup): what the
    /// kit's hook would resolve before building, with nothing fetched or
    /// built. For a caller that needs a row's policy as a value: a test
    /// feeding a recorded signal, a listing of what a preset sets.
    ///
    /// # Errors
    /// As [`lookup`](Self::lookup) and
    /// [`resolve_presets`](crate::kits::support::model_config::resolve_presets).
    pub fn resolve_presets<C>(
        &self,
        spec: &str,
        base: &C,
    ) -> BunsenResult<ResolvedPresets<C>>
    where
        C: Serialize + DeserializeOwned,
    {
        let expr = LookupExpr::parse(spec)?;
        let (_, row) = self.lookup(spec)?;
        resolve_presets(base, &row.presets, &expr.selection)
    }

    /// Checks what a lookup asks of `row`'s presets, before anything is
    /// fetched: that the kit resolves presets at all, and that the selection
    /// is one `row`'s presets allow
    /// ([`ResolvedPresetSet::check_selection`](crate::data::catalog::presets::ResolvedPresetSet::check_selection)).
    /// A row with no presets of its own, an `hf:` repo's, is left to the hook,
    /// which knows the model's family only once it reads the files
    /// ([`resolve_ref_presets`](crate::kits::support::model_config::resolve_ref_presets)).
    fn check_selection(
        row: &Pretrained,
        selection: &PresetSelection,
    ) -> BunsenResult<()> {
        if selection.is_empty() {
            return Ok(());
        }
        if !H::RESOLVES_PRESETS {
            return Err(BunsenError::unsupported(format!(
                "the {} kit takes no presets or patch, but the lookup asks for `{selection}`",
                H::KIT
            )));
        }
        if row.presets.is_empty() {
            return Ok(());
        }
        row.presets
            .check_selection(selection)
            .with_context(|| format!("the presets of {:?}", row.name))?;
        Ok(())
    }

    /// The row `spec` names, dispatched as [`lookup`](Self::lookup) is but
    /// through each provider's [`resolve`](PretrainedProvider::resolve)
    /// with `cache` to hand, as a [`Deferred`] model with the hook its resource
    /// set calls for: the index half of [`load`](Self::load), for a caller
    /// that overlays the row before loading it. A table answers from its
    /// rows; a hub asks what the repo holds, once, and keeps the answer in
    /// the cache.
    ///
    /// # Errors
    /// As [`lookup`](Self::lookup) and [`Deferred::new`]. The spec is the
    /// caller's input, a name from a command line say: an
    /// [`Illegal`](BunsenErrorKind::Illegal) error it causes, a malformed
    /// `hf:` ref say, is re-marked [`Policy`](BunsenErrorKind::Policy).
    pub fn resolve(
        &self,
        spec: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<Deferred<H>> {
        let resolve = || -> BunsenResult<Deferred<H>> {
            let expr = LookupExpr::parse(spec)?;
            let found = self.find(&expr.name, |provider, name| provider.resolve(name, cache))?;
            let (provider, pretrained) = match found {
                Some(found) => found,
                None => return Err(self.not_found(&expr.name)),
            };
            Self::check_selection(&pretrained, &expr.selection)?;
            Deferred::new(PretrainedRef::Named {
                provider,
                pretrained: Box::new(pretrained),
                selection: expr.selection,
            })
        };

        let defered = resolve().as_policy()?;

        if log::log_enabled!(Level::Info) {
            log::info!("Resolved pretrained model: {}\n{:#?}", spec, defered.model);
        }

        Ok(defered)
    }

    /// A name to what the kit builds: [`resolve`](Self::resolve), then
    /// [`Deferred::load`] under `policy`. The whole pathway.
    ///
    /// # Errors
    /// As [`resolve`](Self::resolve) and [`Deferred::load`].
    pub fn load(
        &self,
        spec: &str,
        cache: &ResourceCache,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<Loaded<H::Construct>> {
        self.resolve(spec, cache)?.load(cache, device, policy)
    }

    /// A name to the construct alone, under `policy`: [`load`](Self::load),
    /// keeping the handle.
    ///
    /// # Errors
    /// As [`load`](Self::load).
    pub fn load_construct(
        &self,
        spec: &str,
        cache: &ResourceCache,
        device: &Device,
        policy: LoadPolicy,
    ) -> BunsenResult<Arc<H::Construct>> {
        Ok(self.load(spec, cache, device, policy)?.handle)
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
            return Ok(ask(provider.as_ref(), rest)?
                .map(|row| (provider.name().to_string(), Self::with_kit(row))));
        }
        for provider in &self.providers {
            if !provider.answers_bare_names() {
                continue;
            }
            if let Some(row) = ask(provider.as_ref(), spec)? {
                return Ok(Some((provider.name().to_string(), Self::with_kit(row))));
            }
        }
        Ok(None)
    }

    /// The error for a spec no provider has.
    #[track_caller]
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
    use crate::data::resources::{
        LoadedResourceSet,
        Resource,
        ResourceSet,
        Source,
    };

    /// A provider that can look up but not list: the shape of a hub
    /// wrapper. Answers `hub:<org>/<repo>` with a two-file resource set of
    /// URLs, and counts how often it was asked, so a test can pin that a bare
    /// name never reached it.
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
                size: None,
                dtype: None,
                slot: key.to_string(),
                file: file.to_string(),
                sha256: None,
                format: Some(kind.to_string()),
                name: "hub".to_string(),
                sources: vec![Source::Url(format!("{base}/{file}"))],
            };
            let resources = ResourceSet::new(name)
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
                presets: Default::default(),
                args: Default::default(),
                tags: Default::default(),
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
            Err(BunsenError::unavailable("failing: unreachable"))
        }
    }

    /// A hook for kit `kit` that builds the checkpoint's path, whatever the
    /// resource set says.
    #[derive(Clone, Debug, Default)]
    pub struct CheckpointPath;

    impl ConstructInit for CheckpointPath {
        type Construct = PathBuf;

        const KIT: &'static str = "kit";

        fn for_resource_set(_map: &ResourceSet) -> BunsenResult<Self> {
            Ok(Self)
        }

        fn try_init(
            &self,
            _model: &PretrainedRef,
            loaded: &LoadedResourceSet,
            _device: &Device,
            _policy: LoadPolicy,
        ) -> BunsenResult<Arc<PathBuf>> {
            Ok(Arc::new(loaded.expect("checkpoint")?.to_path_buf()))
        }
    }

    /// [`CheckpointPath`], for a kit whose hook resolves presets.
    #[derive(Clone, Debug, Default)]
    pub struct PresetPath;

    impl ConstructInit for PresetPath {
        type Construct = PathBuf;

        const KIT: &'static str = "kit";
        const RESOLVES_PRESETS: bool = true;

        fn for_resource_set(_map: &ResourceSet) -> BunsenResult<Self> {
            Ok(Self)
        }

        fn try_init(
            &self,
            model: &PretrainedRef,
            loaded: &LoadedResourceSet,
            device: &Device,
            policy: LoadPolicy,
        ) -> BunsenResult<Arc<PathBuf>> {
            CheckpointPath.try_init(model, loaded, device, policy)
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
            PresetPath,
        },
        *,
    };
    use crate::{
        data::{
            builds::{
                PretrainedGroup,
                PretrainedTable,
                WELL_KNOWN,
            },
            resources::{
                Resource,
                ResourceSet,
                Source,
            },
        },
        errors::{
            LookupProblem,
            testing::{
                ErrorMatcher,
                predicate,
                text,
            },
        },
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
            resources: ResourceSet::new(name).with_resource(Resource {
                size: None,
                dtype: None,
                slot: "checkpoint".to_string(),
                file: format!("{name}.pt"),
                sha256: None,
                format: Some("pytorch".to_string()),
                name: "t".to_string(),
                sources: vec![Source::Url(format!("https://t.example/{name}.pt"))],
            }),
            presets: Default::default(),
            args: Default::default(),
            tags: Default::default(),
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
            resources: ResourceSet::given(name, "checkpoint", path),
            presets: Default::default(),
            args: Default::default(),
            tags: Default::default(),
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
            case_sensitive: false,
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
    fn offline_cache(dir: &std::path::Path) -> ResourceCache {
        use crate::data::{
            cache::BunsenDiskCacheOptions,
            resources::ResourceCacheOptions,
        };
        ResourceCache::new(
            ResourceCacheOptions::default()
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

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .display(text::eq("kit: duplicate provider \"well-known\""))
            .cause(predicate("a duplicate provider", |c: &LookupError| {
                c.problem == LookupProblem::Duplicate
            }))
            .assert_err(&factory.register(well_known()));
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

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::starts_with("well-known: no pretrained \"a/tiny\""))
            .message_contains("well-known:a/small")
            .assert_err(&factory.lookup("well-known:a/tiny"));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::starts_with("kit: no provider \"nobody\""))
            .cause(predicate("the providers there are", |c: &LookupError| {
                c.candidates == ["well-known", "hub"]
            }))
            .assert_err(&factory.lookup("nobody:a/small"));
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

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::starts_with("kit: no pretrained \"gigantic\""))
            .message_contains("well-known:a/small")
            .message_contains("second:a/only-here")
            .assert_err(&factory.lookup("gigantic"));
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

        ErrorMatcher::kind(BunsenErrorKind::Lookup).assert_err(&factory.lookup("whisper-base"));
        assert_eq!(hub.lookups(), 1, "a bare name never reached the hub");

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::eq("hub: no pretrained \"nonsense\""))
            .cause(predicate("no candidates", |c: &LookupError| {
                c.candidates.is_empty()
            }))
            .assert_err(&factory.lookup("hub:nonsense"));
        assert_eq!(hub.lookups(), 2);
    }

    /// A provider's own error is the lookup's error, qualified or bare: it
    /// is not "not here".
    #[test]
    fn test_a_providers_error_aborts_the_lookup() {
        let factory = PretrainedFactory::<CheckpointPath>::new()
            .with_providers([Arc::new(Failing) as _, well_known()])
            .unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Unavailable).assert_err(&factory.lookup("failing:x"));
        // The failing provider answers bare names, and is asked first.
        ErrorMatcher::kind(BunsenErrorKind::Unavailable).assert_err(&factory.lookup("small"));
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
        assert_eq!(model.to_resource_set().name, "well-known:a/small");
        assert!(format!("{:?}", model.config).contains("CheckpointPath"));
        let model = factory.resolve("hub:x/y", &cache).unwrap();
        assert_eq!(model.id(), "hub:x/y");

        let file = dir.path().join("ckpt.pt");
        std::fs::write(&file, b"x").unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("well-known:a/small")
            .assert_err(&factory.resolve(file.to_str().unwrap(), &cache));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .assert_err(&factory.resolve("well-known:a/gigantic", &cache));
    }

    /// `load` runs the whole pathway through the hook the resource set
    /// called for; a ref the caller overlays first goes through the same hook.
    #[test]
    fn test_load_goes_through_the_hook() {
        use crate::{
            data::resources::Provenance,
            support::testing::cpu_device,
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
            .load(
                "disk:l/ckpt",
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::as_stored(),
            )
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
            .with_overlay(ResourceSet::given("mine", "checkpoint", &other))
            .unwrap()
            .load(
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::as_stored(),
            )
            .unwrap();
        assert_eq!(*loaded.handle, other);

        // Offline, and nothing local.
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .frame_contains("loading kit \"well-known:a/small\"")
            .message_contains("the cache is offline")
            .assert_err(&factory.load(
                "well-known:a/small",
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::as_stored(),
            ));
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

    /// Every row the factory hands out says which kit it is, whichever way
    /// it was found; a row that already says keeps its own.
    #[test]
    fn test_rows_carry_their_kit() {
        let factory = factory();
        assert!(
            factory
                .list()
                .iter()
                .chain(&factory.for_prefab("small"))
                .all(|(_, row)| row.tags.text(KIT) == Some("kit"))
        );
        let (_, row) = factory.lookup("a/small").unwrap();
        assert_eq!(row.tags.text(KIT), Some("kit"));
        assert_eq!(
            factory
                .lookup("well-known:b/tiny")
                .unwrap()
                .1
                .tags
                .text(KIT),
            Some("kit")
        );

        let mut own = row;
        own.tags.insert(KIT, "other");
        let kept = PretrainedFactory::<CheckpointPath>::with_kit(own);
        assert_eq!(kept.tags.text(KIT), Some("other"));
    }

    /// A well-known table compares names ignoring case, and hands the row
    /// back named as it declares it.
    #[test]
    fn test_well_known_names_ignore_case() {
        let factory = factory();
        let (provider, row) = factory.lookup("A/SMALL").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.as_str()),
            ("well-known", "a/small")
        );
        assert_eq!(
            factory.lookup("well-known:B/Tiny").unwrap().1.name,
            "b/tiny"
        );
        assert_eq!(factory.lookup("S").unwrap().1.name, "a/small");
    }

    /// A box or a patch, to a kit whose hook does not resolve presets, is
    /// refused rather than ignored; a malformed expression is the caller's.
    #[test]
    fn test_a_selection_needs_a_kit_that_resolves_presets() {
        let factory = factory();
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_eq("the kit kit takes no presets or patch, but the lookup asks for `[fast]`")
            .assert_err(&factory.lookup("a/small[fast]"));
        ErrorMatcher::kind(BunsenErrorKind::Unsupported)
            .message_contains("{beam: 1}")
            .assert_err(&factory.lookup("a/small{beam: 1}"));
        ErrorMatcher::kind(BunsenErrorKind::Policy).assert_err(&factory.lookup("a/small[fast"));
    }

    /// For a kit that resolves presets, the box is checked against the
    /// row's presets before anything is fetched, and rides on the ref.
    #[test]
    fn test_a_selection_is_checked_against_the_row() {
        use crate::data::catalog::{
            document::{
                MergeMode,
                PresetEntry,
            },
            presets::ResolvedPresetSet,
        };

        let mut small = row("small", &[], "small");
        small.presets = ResolvedPresetSet {
            presets: vec![PresetEntry {
                name: "fast".to_string(),
                merge: MergeMode::Update,
                description: "a greedy search".to_string(),
                patch: serde_json::json!({"beam": 1}),
            }],
            ..Default::default()
        };
        let factory = PretrainedFactory::<PresetPath>::new()
            .with_providers([table(WELL_KNOWN, vec![group("a", vec![small])])])
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());

        let model = factory.resolve("a/small[FAST]{beam: 2}", &cache).unwrap();
        assert_eq!(model.id(), "well-known:a/small");
        let selection = model.model.selection().unwrap();
        assert_eq!(selection.presets, ["FAST"]);
        assert_eq!(selection.patch, Some(serde_json::json!({"beam": 2})));

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("no preset `slow`; there are: fast")
            .assert_err(&factory.resolve("a/small[slow]", &cache));
        // A row with no presets of its own leaves the box to the hook.
        let bare = PretrainedFactory::<PresetPath>::new()
            .with_providers([table(
                WELL_KNOWN,
                vec![group("a", vec![row("bare", &[], "small")])],
            )])
            .unwrap();
        let model = bare.resolve("a/bare[anything]", &cache).unwrap();
        assert_eq!(model.model.selection().unwrap().presets, ["anything"]);
        // An empty box is a malformed expression, not an empty selection.
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .assert_err(&factory.resolve("a/small[]", &cache));
    }
}
