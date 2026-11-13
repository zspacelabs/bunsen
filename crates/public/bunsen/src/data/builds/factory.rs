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
            BuildProvider,
            BuildRef,
            ConstructInit,
            DeferredBuild,
            Loaded,
            LookupExpr,
        },
        catalog::{
            ResolvedBuild,
            presets::PresetSelection,
        },
        formats::LoadPolicy,
        resources::ResourceCache,
        support::not_found,
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

/// A kit's providers, in search order, resolving names to [`DeferredBuild`]
/// models built through the kit's hook `H`.
///
/// The one object a caller holds for a kit. Its interface is names and a
/// listing. [`load`](Self::load)`("[provider:]name", &cache, &device)`
/// is the whole pathway, from a name to a [`Loaded`] model.
/// [`resolve`](Self::resolve) is its index half: a [`DeferredBuild`] model that
/// carries the hook its resource set calls for, so that a caller can overlay
/// a build ([`DeferredBuild::with_overlay`]) before loading it, and never
/// builds or passes a hook. [`list`](Self::list) and [`ids`](Self::ids) answer
/// a listing. A file on disk is
/// not the factory's business: that is a given [`ResourceSet`], through
/// [`DeferredBuild::from_resource_set`].
///
/// # Lookup expressions
///
/// A spec is a [`LookupExpr`]: `[provider:]name`, then an optional box of
/// presets and an optional YAML patch, `name[preset, …]{patch}`. Only the
/// name is dispatched. The box and the patch are checked against the build
/// found before anything is fetched: a kit whose hook does not resolve
/// presets ([`ConstructInit::RESOLVES_PRESETS`]) refuses them, and one that
/// does checks the box against the build's presets
/// ([`ResolvedPresetSet::check_selection`](crate::data::catalog::presets::ResolvedPresetSet::check_selection)).
/// The selection rides on the ref ([`BuildRef::selection`]) to the
/// hook, which resolves its config through it.
///
/// A catalog compares names ignoring ASCII case: `Qwen/Qwen3-0.6B` finds
/// `qwen/qwen3/qwen3-0.6b`. A hub compares as it is asked, since its names
/// are paths in URLs.
///
/// # Dispatch
///
/// - `provider:ref`, where `provider` names a registered provider, goes to that
///   provider and nowhere else.
/// - Any other spec, with no `:` or with a prefix that names no provider, is
///   offered whole to each provider that [answers bare
///   names](BuildProvider::answers_bare_names), in registration order, and the
///   first row wins. A hub such as
///   [`HfProvider`](crate::data::builds::HfProvider) declines bare names, so it
///   is only reached qualified.
/// - A provider's "not here" is `Ok(None)`, and the search moves on. Any error
///   a provider returns aborts the lookup: a hub that cannot be reached is not
///   the same as a hub that has no such row.
/// - A spec no provider has is a [`Lookup`](BunsenErrorKind::Lookup) error,
///   with a [`LookupError`] naming what there is.
///
/// [`resolve`](Self::resolve) asks each provider through
/// [`BuildProvider::resolve`], with the cache to hand, so a hub can
/// ask what a repo holds once and answer from the cache after.
/// [`lookup`](Self::lookup) asks [`BuildProvider::lookup`], the index
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
/// // order: the catalog provider has it.
/// let (provider, build) = factory.lookup("tiny")?;
/// assert_eq!(provider, "catalog");
/// assert_eq!(build.name.to_string(), "openai/whisper/tiny");
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
pub struct BuildFactory<H: ConstructInit> {
    providers: Vec<Arc<dyn BuildProvider>>,
    config: PhantomData<H>,
}

impl<H: ConstructInit> Clone for BuildFactory<H> {
    fn clone(&self) -> Self {
        Self {
            providers: self.providers.clone(),
            config: PhantomData,
        }
    }
}

impl<H: ConstructInit> Default for BuildFactory<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H: ConstructInit> BuildFactory<H> {
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
        provider: Arc<dyn BuildProvider>,
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
        provider: Arc<dyn BuildProvider>,
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
        providers: impl IntoIterator<Item = Arc<dyn BuildProvider>>,
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
    ) -> Option<Arc<dyn BuildProvider>> {
        let at = self.providers.iter().position(|p| p.name() == name)?;
        Some(self.providers.remove(at))
    }

    /// The providers, in search order.
    pub fn providers(&self) -> &[Arc<dyn BuildProvider>] {
        &self.providers
    }

    /// The provider called `name`.
    pub fn provider(
        &self,
        name: &str,
    ) -> Option<&Arc<dyn BuildProvider>> {
        self.providers.iter().find(|p| p.name() == name)
    }

    /// Every listed build with its provider's name, in registration then
    /// listing order.
    pub fn list(&self) -> Vec<(String, ResolvedBuild)> {
        self.providers
            .iter()
            .flat_map(|provider| {
                let name = provider.name().to_string();
                provider
                    .list()
                    .into_iter()
                    .map(move |build| (name.clone(), build))
            })
            .collect()
    }

    /// Every listed id, `provider:ref`: the "did you mean" list.
    pub fn ids(&self) -> Vec<String> {
        self.providers.iter().flat_map(|p| p.ids()).collect()
    }

    /// The build `spec` names, with its provider's name.
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
    ) -> BunsenResult<(String, ResolvedBuild)> {
        let expr = LookupExpr::parse(spec)?;
        let (provider, build) =
            match self.find(&expr.name, |provider, name| provider.lookup(name))? {
                Some(found) => found,
                None => return Err(self.not_found(&expr.name)),
            };
        Self::check_selection(&build, &expr.selection)?;
        Ok((provider, build))
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
        let (_, build) = self.lookup(spec)?;
        resolve_presets(base, &build.presets, &expr.selection)
    }

    /// Checks what a lookup asks of `row`'s presets, before anything is
    /// fetched: that the kit resolves presets at all, and that the selection
    /// is one `row`'s presets allow
    /// ([`ResolvedPresetSet::check_selection`](crate::data::catalog::presets::ResolvedPresetSet::check_selection)).
    /// A row with no presets of its own, an `hf:` repo's, is left to the hook,
    /// which knows the model's family only once it reads the files
    /// ([`resolve_ref_presets`](crate::kits::support::model_config::resolve_ref_presets)).
    fn check_selection(
        build: &ResolvedBuild,
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
        if build.presets.is_empty() {
            return Ok(());
        }
        build
            .presets
            .check_selection(selection)
            .with_context(|| format!("the presets of {:?}", build.name.to_string()))?;
        Ok(())
    }

    /// The row `spec` names, dispatched as [`lookup`](Self::lookup) is but
    /// through each provider's [`resolve`](BuildProvider::resolve)
    /// with `cache` to hand, as a [`DeferredBuild`] model with the hook its
    /// resource set calls for: the index half of [`load`](Self::load), for a
    /// caller that overlays the row before loading it. A table answers from its
    /// rows; a hub asks what the repo holds, once, and keeps the answer in
    /// the cache.
    ///
    /// # Errors
    /// As [`lookup`](Self::lookup) and [`DeferredBuild::new`]. The spec is the
    /// caller's input, a name from a command line say: an
    /// [`Illegal`](BunsenErrorKind::Illegal) error it causes, a malformed
    /// `hf:` ref say, is re-marked [`Policy`](BunsenErrorKind::Policy).
    pub fn resolve(
        &self,
        spec: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<DeferredBuild<H>> {
        let resolve = || -> BunsenResult<DeferredBuild<H>> {
            let expr = LookupExpr::parse(spec)?;
            let found = self.find(&expr.name, |provider, name| provider.resolve(name, cache))?;
            let (provider, build) = match found {
                Some(found) => found,
                None => return Err(self.not_found(&expr.name)),
            };
            Self::check_selection(&build, &expr.selection)?;
            DeferredBuild::new(BuildRef::named(provider, build, expr.selection))
        };

        let defered = resolve().as_policy()?;

        log::debug!("resolved {spec} to {}", defered.model.summary());
        if log::log_enabled!(Level::Trace) {
            log::trace!("resolved {spec}:\n{:#?}", defered.model);
        }

        Ok(defered)
    }

    /// A name to what the kit builds: [`resolve`](Self::resolve), then
    /// [`DeferredBuild::load`] under `policy`. The whole pathway.
    ///
    /// # Errors
    /// As [`resolve`](Self::resolve) and [`DeferredBuild::load`].
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
        ask: impl Fn(&dyn BuildProvider, &str) -> BunsenResult<Option<ResolvedBuild>>,
    ) -> BunsenResult<Option<(String, ResolvedBuild)>> {
        if let Some((head, rest)) = spec.split_once(':')
            && let Some(provider) = self.provider(head)
        {
            return Ok(ask(provider.as_ref(), rest)?.map(|b| (provider.name().to_string(), b)));
        }
        for provider in &self.providers {
            if !provider.answers_bare_names() {
                continue;
            }
            if let Some(build) = ask(provider.as_ref(), spec)? {
                return Ok(Some((provider.name().to_string(), build)));
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
            let described: Vec<String> = self
                .providers
                .iter()
                .map(|p| format!("{}: {}", p.name(), p.description()))
                .collect();
            return not_found(Some(H::KIT), "provider", head, &names)
                .with_details(described.join("\n"));
        }
        let ids = self.ids();
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        not_found(Some(H::KIT), "pretrained", spec, &ids)
    }
}

/// Providers and hooks for the factory's tests: a catalog of toy builds, a
/// hub that answers a ref but lists nothing, one that fails, and a hook
/// that builds a path.
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
    use crate::data::{
        builds::CatalogBuildProvider,
        catalog::{
            BunsenCatalogStack,
            CatalogName,
            document::{
                ResourceEntry,
                ResourceSource,
            },
        },
        resources::{
            LoadedResourceSet,
            ResourceSet,
        },
    };

    /// The base catalog of the toy builds: the `toy` contract, whose builds
    /// are one `checkpoint`.
    pub const TOY_PRELUDE: &str = r#"
$contracts:
  toy:
    $kind: contract
    $description: "A toy checkpoint."
    $resources:
      checkpoint: { $format: [pytorch] }
template_class_root: { $kind: template_class, $description: "The root." }
"#;

    /// The toy builds `yaml` declares, over [`TOY_PRELUDE`], as a catalog
    /// provider called `name`.
    pub fn toy_catalog(
        name: &str,
        yaml: &str,
    ) -> Arc<dyn BuildProvider> {
        let mut stack = BunsenCatalogStack::new();
        stack.push_yaml("prelude", TOY_PRELUDE).unwrap();
        stack.push_yaml("builds", yaml).unwrap();
        Arc::new(
            CatalogBuildProvider::new(name, format!("the {name} builds"), &stack, "toy").unwrap(),
        )
    }

    /// A toy build `leaf`, its checkpoint `{leaf}.pt` at `t.example`, with
    /// `extra` YAML fields, as an entry of a namespace's.
    pub fn toy_build(
        leaf: &str,
        extra: &str,
    ) -> String {
        format!(
            r#"  {leaf}:
    $kind: build
    $description: "Build {leaf}."
    $contract: toy
    $resources: {{ checkpoint: checkpoint }}
    {extra}
    checkpoint:
      $kind: resource
      $file: {leaf}.pt
      $format: pytorch
      $sources: [{{ $url: "https://t.example/{leaf}.pt" }}]
"#
        )
    }

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

    impl BuildProvider for ListsNothing {
        fn name(&self) -> &str {
            "hub"
        }

        fn description(&self) -> &str {
            "answers `hub:<org>/<repo>`; lists nothing"
        }

        fn origin(&self) -> Option<&str> {
            Some("https://hub.example")
        }

        fn list(&self) -> Vec<ResolvedBuild> {
            Vec::new()
        }

        fn lookup(
            &self,
            name: &str,
        ) -> BunsenResult<Option<ResolvedBuild>> {
            self.lookups.fetch_add(1, Ordering::SeqCst);
            let Some((org, repo)) = name.split_once('/') else {
                return Ok(None);
            };
            let base = format!("https://hub.example/{org}/{repo}/resolve/main");
            let n = |rest: &str| CatalogName::parse(&format!("hub/{org}/{repo}{rest}")).unwrap();
            let file = |file: &str, format: &str| {
                ResourceEntry::new(
                    file,
                    format,
                    vec![ResourceSource::Url(format!("{base}/{file}"))],
                )
            };
            let mut build = ResolvedBuild::new(n(""), format!("{org}/{repo}, from the hub"))
                .with_resource(
                    "checkpoint",
                    n("/checkpoint"),
                    file("model.safetensors", "safetensors"),
                )
                .with_resource("config", n("/config"), file("config.json", "json"));
            build.origin = Some(base);
            Ok(Some(build))
        }

        fn answers_bare_names(&self) -> bool {
            false
        }
    }

    /// A provider whose lookups fail: a hub that cannot be reached.
    #[derive(Debug, Default)]
    pub struct Failing;

    impl BuildProvider for Failing {
        fn name(&self) -> &str {
            "failing"
        }

        fn description(&self) -> &str {
            "cannot be reached"
        }

        fn list(&self) -> Vec<ResolvedBuild> {
            Vec::new()
        }

        fn lookup(
            &self,
            _name: &str,
        ) -> BunsenResult<Option<ResolvedBuild>> {
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
            _model: &BuildRef,
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
            model: &BuildRef,
            loaded: &LoadedResourceSet,
            device: &Device,
            policy: LoadPolicy,
        ) -> BunsenResult<Arc<PathBuf>> {
            CheckpointPath.try_init(model, loaded, device, policy)
        }

        /// A fixed selection, `[applied]`: what a hook reports it applied.
        fn applied_presets(_construct: &PathBuf) -> Option<PresetSelection> {
            Some(PresetSelection::of(["applied"]))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{
        testing::{
            CheckpointPath,
            Failing,
            ListsNothing,
            PresetPath,
            toy_build,
            toy_catalog,
        },
        *,
    };
    use crate::{
        data::{
            builds::CATALOG,
            resources::ResourceSet,
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

    /// `catalog`, the toy builds `a/small` (alias `s`), `a/large-v1`,
    /// `a/large-v2` (alias `large`), `b/small` and `b/tiny`.
    fn well_known() -> Arc<dyn BuildProvider> {
        toy_catalog(
            CATALOG,
            &format!(
                r#"
a:
  $kind: namespace
  $description: "Group a."
  s: {{ $kind: alias, $ref: {{ $path: small, $kind: build }} }}
  large: {{ $kind: alias, $ref: {{ $path: large-v2, $kind: build }} }}
{}{}{}b:
  $kind: namespace
  $description: "Group b."
{}{}"#,
                toy_build("small", ""),
                toy_build("large-v1", ""),
                toy_build("large-v2", ""),
                toy_build("small", ""),
                toy_build("tiny", ""),
            ),
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

    fn factory() -> BuildFactory<CheckpointPath> {
        BuildFactory::new()
            .with_providers([well_known(), Arc::new(ListsNothing::default()) as _])
            .unwrap()
    }

    #[test]
    fn test_register_refuses_a_duplicate_name_and_remove_frees_it() {
        let mut factory = BuildFactory::<CheckpointPath>::new()
            .with_provider(well_known())
            .unwrap();
        assert_eq!(factory.kit(), "kit");
        assert_eq!(factory.providers().len(), 1);
        assert!(factory.provider(CATALOG).is_some());

        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .display(text::eq("kit: duplicate provider \"catalog\""))
            .cause(predicate("a duplicate provider", |c: &LookupError| {
                c.problem == LookupProblem::Duplicate
            }))
            .assert_err(&factory.register(well_known()));
        assert_eq!(factory.providers().len(), 1);

        assert!(factory.remove("nobody").is_none());
        let removed = factory.remove(CATALOG).unwrap();
        assert_eq!(removed.name(), CATALOG);
        assert!(factory.providers().is_empty());
        factory.register(well_known()).unwrap();
        assert!(format!("{factory:?}").contains("CatalogBuildProvider"));
        assert_eq!(factory.clone().providers().len(), 1);
        assert!(
            BuildFactory::<CheckpointPath>::default()
                .providers()
                .is_empty()
        );

        // The builder form surfaces the same error through `?`.
        assert!(
            BuildFactory::<CheckpointPath>::new()
                .with_providers([well_known(), well_known()])
                .is_err()
        );
    }

    /// `provider:ref` asks that provider and nothing else: a ref the
    /// provider has, a ref it does not, a provider that is not there.
    #[test]
    fn test_a_qualified_spec_goes_to_one_provider() {
        let factory = factory();

        let (provider, row) = factory.lookup("catalog:b/tiny").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.to_string().as_str()),
            ("catalog", "b/tiny")
        );
        let (provider, row) = factory.lookup("catalog:large").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.to_string().as_str()),
            ("catalog", "a/large-v2")
        );

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::starts_with("catalog: no pretrained \"a/tiny\""))
            .message_contains("catalog:a/small")
            .assert_err(&factory.lookup("catalog:a/tiny"));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::starts_with("kit: no provider \"nobody\""))
            .cause(predicate("the providers there are", |c: &LookupError| {
                c.candidates == ["catalog", "hub"]
            }))
            .details_contains("catalog: the catalog builds")
            .details_contains("hub: answers `hub:<org>/<repo>`; lists nothing")
            .assert_err(&factory.lookup("nobody:a/small"));
    }

    /// A bare spec is offered to the providers in order; the first row
    /// wins, and a provider that declines bare names is skipped.
    #[test]
    fn test_a_bare_spec_searches_in_order_and_skips_decliners() {
        let second = toy_catalog(
            "second",
            &format!(
                "a:\n  $kind: namespace\n  $description: \"Group a.\"\n{}{}",
                toy_build("small", ""),
                toy_build("only-here", "")
            ),
        );
        let factory = BuildFactory::<CheckpointPath>::new()
            .with_providers([Arc::new(ListsNothing::default()) as _, well_known(), second])
            .unwrap();

        let (provider, row) = factory.lookup("a/small").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.to_string().as_str()),
            ("catalog", "a/small")
        );
        // A spelling two builds of the first catalog match is its error.
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("`small` is ambiguous: a/small, b/small")
            .assert_err(&factory.lookup("small"));
        let (provider, row) = factory.lookup("only-here").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.to_string().as_str()),
            ("second", "a/only-here")
        );
        let (provider, row) = factory.lookup("second:a/small").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.to_string().as_str()),
            ("second", "a/small")
        );
        let (_, row) = factory.lookup("tiny").unwrap();
        assert_eq!(row.name.to_string(), "b/tiny");
        let (_, row) = factory.lookup("a/s").unwrap();
        assert_eq!(row.name.to_string(), "a/small");

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .display(text::starts_with("kit: no pretrained \"gigantic\""))
            .message_contains("catalog:a/small")
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
        let factory = BuildFactory::<CheckpointPath>::new()
            .with_providers([well_known(), hub.clone() as _])
            .unwrap();

        assert_eq!(factory.list().len(), 5, "the hub lists nothing");
        assert_eq!(factory.ids().len(), 5);
        assert!(factory.ids().iter().all(|id| id.starts_with("catalog:")));

        let (provider, row) = factory.lookup("hub:openai/whisper-base").unwrap();
        assert_eq!(provider, "hub");
        assert_eq!(row.name.to_string(), "hub/openai/whisper-base");
        let set = row.resource_set();
        assert_eq!(set.keys(), ["checkpoint", "config"]);
        assert_eq!(
            set.get("checkpoint").unwrap().urls(),
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
        let factory = BuildFactory::<CheckpointPath>::new()
            .with_providers([Arc::new(Failing) as _, well_known()])
            .unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Unavailable).assert_err(&factory.lookup("failing:x"));
        // The failing provider answers bare names, and is asked first.
        ErrorMatcher::kind(BunsenErrorKind::Unavailable).assert_err(&factory.lookup("small"));
        assert_eq!(
            factory.lookup("catalog:tiny").unwrap().1.name.to_string(),
            "b/tiny"
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

        let model = factory.resolve("catalog:a/small", &cache).unwrap();
        assert_eq!(model.id(), "catalog:a/small");
        let model = factory.resolve("s", &cache).unwrap();
        assert_eq!(model.id(), "catalog:a/small");
        assert_eq!(
            model
                .model
                .build()
                .map(|(p, build)| (p, build.name.to_string())),
            Some(("catalog", "a/small".to_string()))
        );
        assert_eq!(model.to_resource_set().name, "catalog:a/small");
        assert!(format!("{:?}", model.config).contains("CheckpointPath"));
        let model = factory.resolve("hub:x/z", &cache).unwrap();
        assert_eq!(model.id(), "hub:x/z");

        let file = dir.path().join("ckpt.pt");
        std::fs::write(&file, b"x").unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("catalog:a/small")
            .assert_err(&factory.resolve(file.to_str().unwrap(), &cache));
        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .assert_err(&factory.resolve("catalog:a/gigantic", &cache));
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
        let on_disk = toy_catalog(
            "disk",
            &format!(
                r#"
l:
  $kind: namespace
  $description: "On disk."
  ckpt:
    $kind: build
    $description: "A checkpoint on disk."
    $contract: toy
    $resources: {{ checkpoint: checkpoint }}
    checkpoint:
      $kind: resource
      $file: ckpt.pt
      $format: pytorch
      $sources: [{{ $local_dir: {{ $name: disk, $default: ["{}"] }} }}]
"#,
                dir.path().display()
            ),
        );
        let factory = BuildFactory::<CheckpointPath>::new()
            .with_providers([well_known(), on_disk.clone()])
            .unwrap();

        let loaded = factory
            .load(
                "disk:l/ckpt",
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::load_as_stored(),
            )
            .unwrap();
        assert_eq!(*loaded.handle, file);
        assert_eq!(loaded.name, "disk:l/ckpt");
        assert_eq!(
            loaded.resources.get("checkpoint").unwrap().provenance,
            Provenance::LocalDir
        );
        // A kit that resolves no presets reports none applied.
        assert_eq!(loaded.summary.id, "disk:l/ckpt");
        assert_eq!(loaded.summary.requested, Some(PresetSelection::default()));
        assert_eq!(loaded.summary.applied, None);

        // The summary carries what the hook reports it applied.
        let loaded = BuildFactory::<PresetPath>::new()
            .with_providers([on_disk])
            .unwrap()
            .load(
                "disk:l/ckpt",
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::load_as_stored(),
            )
            .unwrap();
        assert_eq!(
            loaded.summary.applied,
            Some(PresetSelection::of(["applied"]))
        );
        assert_eq!(
            loaded.summary.to_string().split(' ').nth(1),
            Some("[applied]")
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
                crate::data::formats::LoadPolicy::load_as_stored(),
            )
            .unwrap();
        assert_eq!(*loaded.handle, other);

        // Offline, and nothing local.
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .frame_contains("loading kit \"catalog:a/small\"")
            .message_contains("the cache is offline")
            .assert_err(&factory.load(
                "catalog:a/small",
                &cache,
                &cpu_device(),
                crate::data::formats::LoadPolicy::load_as_stored(),
            ));
    }

    #[test]
    fn test_list_and_ids_cover_every_provider() {
        let factory = factory();
        let listed: Vec<String> = factory
            .list()
            .iter()
            .map(|(p, build)| format!("{p}:{}", build.name))
            .collect();
        assert_eq!(listed, factory.ids());
        assert_eq!(
            factory.ids(),
            [
                "catalog:a/small",
                "catalog:a/large-v1",
                "catalog:a/large-v2",
                "catalog:b/small",
                "catalog:b/tiny",
            ]
        );
    }

    /// A catalog compares names ignoring case, and hands the build back
    /// named as it declares it.
    #[test]
    fn test_well_known_names_ignore_case() {
        let factory = factory();
        let (provider, row) = factory.lookup("A/SMALL").unwrap();
        assert_eq!(
            (provider.as_str(), row.name.to_string().as_str()),
            ("catalog", "a/small")
        );
        assert_eq!(
            factory.lookup("catalog:B/Tiny").unwrap().1.name.to_string(),
            "b/tiny"
        );
        assert_eq!(factory.lookup("S").unwrap().1.name.to_string(), "a/small");
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
        let catalog = |extra: &str| {
            toy_catalog(
                CATALOG,
                &format!(
                    "a:\n  $kind: namespace\n  $description: \"Group a.\"\n{}{}",
                    toy_build("small", extra),
                    toy_build("bare", "")
                ),
            )
        };
        let factory = BuildFactory::<PresetPath>::new()
            .with_providers([catalog(
                r#"$presets: { fast: { $description: "a greedy search", $patch: { beam: 1 } } }"#,
            )])
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let cache = offline_cache(dir.path());

        let model = factory.resolve("a/small[FAST]{beam: 2}", &cache).unwrap();
        assert_eq!(model.id(), "catalog:a/small");
        let selection = model.model.selection().unwrap();
        assert_eq!(selection.presets, ["FAST"]);
        assert_eq!(selection.patch, Some(serde_json::json!({"beam": 2})));
        // Before a load, the summary shows what was asked.
        let summary = model.model.summary();
        assert_eq!(summary.requested.as_ref(), Some(selection));
        assert!(
            summary
                .to_string()
                .starts_with("catalog:a/small [FAST]{beam: 2}"),
            "{summary}"
        );

        ErrorMatcher::kind(BunsenErrorKind::Lookup)
            .message_contains("no preset `slow`; there are: fast")
            .assert_err(&factory.resolve("a/small[slow]", &cache));
        // A build with no presets of its own leaves the box to the hook.
        let model = factory.resolve("a/bare[anything]", &cache).unwrap();
        assert_eq!(model.model.selection().unwrap().presets, ["anything"]);
        // An empty box is a malformed expression, not an empty selection.
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .assert_err(&factory.resolve("a/small[]", &cache));
    }
}
