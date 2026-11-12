//! Build providers: namespaces over builds.

use alloc::{
    format,
    string::String,
    vec::Vec,
};
use core::fmt;

#[cfg(feature = "cache")]
use crate::data::resources::ResourceCache;
use crate::{
    data::catalog::ResolvedBuild,
    errors::BunsenResult,
};

/// The name of the provider the catalog's builds sit behind:
/// `catalog:openai/whisper/tiny`.
pub const CATALOG: &str = "catalog";

/// A namespace of builds: the `provider` of `provider:ref`.
///
/// A provider names and looks up, and lists when it can, answering with the
/// catalog's [`ResolvedBuild`]: what the build is, its files in the
/// catalog's form, its presets and its arguments. It knows no hook, since
/// the function that loads a kit's models is the kit's and supplies one, and
/// it does not know which resource of a build is the model.
///
/// A [`BuildFactory`](crate::data::builds::BuildFactory)
/// holds a kit's providers in search order and offers each spec to them; its
/// [dispatch rules] are where `Ok(None)` and an error part ways. The
/// implementations in bunsen are
/// [`CatalogBuildProvider`](super::CatalogBuildProvider), for a catalog's
/// builds ([`CATALOG`]), and [`HfProvider`](super::HfProvider), for
/// `hf:org/repo`.
///
/// [dispatch rules]: crate::data::builds::BuildFactory#dispatch
///
/// Object-safe by construction, so a factory holds any mix of these behind
/// `Arc<dyn BuildProvider>`. [`list`](Self::list) and
/// [`lookup`](Self::lookup) are separate so that a provider which cannot
/// enumerate what it has (a hub) can still answer a ref, with an empty
/// listing. Such a provider also says it does not
/// [answer bare names](Self::answers_bare_names), so a spec with no
/// `provider:` never reaches it.
pub trait BuildProvider: Send + Sync + fmt::Debug {
    /// The namespace: the `provider` of `provider:ref`.
    fn name(&self) -> &str;

    /// A line for a listing.
    fn description(&self) -> &str;

    /// The license the rows are distributed under, when one covers them
    /// all.
    fn license(&self) -> Option<&str> {
        None
    }

    /// Where the builds come from.
    fn origin(&self) -> Option<&str> {
        None
    }

    /// The builds this provider can enumerate, in listing order. Empty for a
    /// provider that can look up but not list.
    fn list(&self) -> Vec<ResolvedBuild>;

    /// The build `name` refers to, by its name or an alias
    /// ([`ResolvedBuild::answers_to`]).
    ///
    /// `Ok(None)` is "not here", and a factory moves on to the next
    /// provider. The default scans [`list`](Self::list).
    ///
    /// # Errors
    /// The provider's own: a hub that cannot be reached. Any error aborts
    /// the lookup; it is not "not here".
    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<Option<ResolvedBuild>> {
        Ok(self.list().into_iter().find(|b| b.answers_to(name)))
    }

    /// The build `name` refers to, with a cache to hand: what a factory
    /// asks when it resolves a spec to load. The default is
    /// [`lookup`](Self::lookup). A hub answers here instead, asking the
    /// hub what the repo holds once and keeping the answer in the cache, so
    /// the same ref is answered offline after.
    ///
    /// # Errors
    /// The provider's own, as [`lookup`](Self::lookup).
    #[cfg(feature = "cache")]
    fn resolve(
        &self,
        name: &str,
        cache: &ResourceCache,
    ) -> BunsenResult<Option<ResolvedBuild>> {
        let _ = cache;
        self.lookup(name)
    }

    /// Whether a spec with no `provider:` is offered to this provider.
    ///
    /// `true` by default. A network provider says `false`, so that a bare
    /// name never reaches it; its refs are then only reachable qualified.
    fn answers_bare_names(&self) -> bool {
        true
    }

    /// The qualified id of a build: `provider:ref`, the ref its full name,
    /// or the rest of it under a root named for the provider
    /// (`hf:openai/whisper-tiny` for `hf/openai/whisper-tiny`).
    fn id(
        &self,
        build: &ResolvedBuild,
    ) -> String {
        qualified_id(self.name(), build)
    }

    /// Every qualified id, in listing order.
    fn ids(&self) -> Vec<String> {
        self.list().iter().map(|b| self.id(b)).collect()
    }
}

/// `provider:ref` for `build` in `provider`: the ref the build's full name,
/// or the rest of it under a root named for the provider.
pub fn qualified_id(
    provider: &str,
    build: &ResolvedBuild,
) -> String {
    let name = build.name.to_string();
    let reference = name
        .strip_prefix(provider)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or(&name);
    format!("{provider}:{reference}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::catalog::CatalogName;

    fn build(
        name: &str,
        aliases: &[&str],
    ) -> ResolvedBuild {
        let mut build = ResolvedBuild::new(CatalogName::parse(name).unwrap(), "a build");
        build.aliases = aliases
            .iter()
            .map(|a| CatalogName::parse(a).unwrap())
            .collect();
        build
    }

    /// A provider that only lists gets `lookup` and `ids` for free: a build
    /// answers to its full name, an alias's, or an alias's leaf, in any case.
    #[test]
    fn test_the_provided_lookup_scans_the_listing() {
        #[derive(Debug)]
        struct Flat;

        impl BuildProvider for Flat {
            fn name(&self) -> &str {
                "flat"
            }

            fn description(&self) -> &str {
                "two builds"
            }

            fn origin(&self) -> Option<&str> {
                Some("https://flat.example")
            }

            fn list(&self) -> Vec<ResolvedBuild> {
                alloc::vec![build("a/small", &["a/s"]), build("b/tiny", &[])]
            }
        }

        let flat = Flat;
        assert_eq!(flat.origin(), Some("https://flat.example"));
        assert_eq!(flat.ids(), ["flat:a/small", "flat:b/tiny"]);
        let found = |name: &str| flat.lookup(name).unwrap().map(|b| b.name.to_string());
        assert_eq!(found("A/Small").as_deref(), Some("a/small"));
        assert_eq!(found("s").as_deref(), Some("a/small"));
        assert_eq!(found("a/s").as_deref(), Some("a/small"));
        assert_eq!(found("small"), None, "a leaf of the name is no alias");
    }

    /// A build under a root named for its provider is qualified by the rest.
    #[test]
    fn test_the_qualified_id() {
        assert_eq!(
            qualified_id("hf", &build("hf/openai/whisper-tiny", &[])),
            "hf:openai/whisper-tiny"
        );
        assert_eq!(
            qualified_id("catalog", &build("openai/whisper/tiny", &[])),
            "catalog:openai/whisper/tiny"
        );
        assert_eq!(qualified_id("hf", &build("hfx/a", &[])), "hf:hfx/a");
    }
}
