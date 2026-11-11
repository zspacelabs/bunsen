//! Pretrained providers: namespaces over rows, and the table of groups.

use alloc::{
    format,
    string::String,
    vec::Vec,
};
use core::fmt;

use super::Pretrained;
#[cfg(feature = "cache")]
use super::PretrainedCache;
use crate::errors::BunsenResult;

/// The name of the provider the compiled-in rows sit behind:
/// `well-known:openai/tiny`.
pub const WELL_KNOWN: &str = "well-known";

/// The name of the provider whose rows a build ships with:
/// `bundled:openai/base`. A caller registers one after [`WELL_KNOWN`], so
/// that a bare name means the same row with it or without. A bundled row is
/// served from the bundle rather than fetched: in place from files the build
/// laid out (`bundled:openai/base`, the same files as
/// `well-known:openai/base`), or from bytes linked into the binary
/// ([`Source::Bundled`](super::Source::Bundled)).
pub const BUNDLED: &str = "bundled";

/// A namespace of pretrained rows: the `provider` of `provider:ref`.
///
/// A provider names and looks up, and lists when it can. It knows no hook,
/// since the function that loads a kit's models is the kit's and supplies
/// one, and it does not know which resource of a row is the model. A row
/// may name the prefab it instantiates, so "what shape is this" is
/// answered by the row, and "what rows exist for this shape" is derived by
/// scanning the listing ([`for_prefab`](Self::for_prefab)). One prefab,
/// many rows, many providers.
///
/// A [`PretrainedFactory`](crate::kits::support::construct::PretrainedFactory)
/// holds a kit's providers in search order and offers each spec to them; its
/// [dispatch rules] are where `Ok(None)` and an error part ways. The
/// implementations in bunsen are [`PretrainedTable`], for compiled-in rows
/// ([`WELL_KNOWN`], [`BUNDLED`]) and for any table read from a manifest, and
/// [`HfProvider`](super::HfProvider), for `hf:org/repo`.
///
/// [dispatch rules]: crate::kits::support::construct::PretrainedFactory#dispatch
///
/// Object-safe by construction, so a factory holds any mix of these behind
/// `Arc<dyn PretrainedProvider>`. [`list`](Self::list) and
/// [`lookup`](Self::lookup) are separate so that a provider which cannot
/// enumerate what it has (a hub) can still answer a ref, with an empty
/// listing. Such a provider also says it does not
/// [answer bare names](Self::answers_bare_names), so a spec with no
/// `provider:` never reaches it.
pub trait PretrainedProvider: Send + Sync + fmt::Debug {
    /// The namespace: the `provider` of `provider:ref`.
    fn name(&self) -> &str;

    /// A line for a listing.
    fn description(&self) -> &str;

    /// The license the rows are distributed under, when one covers them
    /// all.
    fn license(&self) -> Option<&str> {
        None
    }

    /// Where the table came from.
    fn origin(&self) -> Option<&str> {
        None
    }

    /// The rows this provider can enumerate, in listing order; each row's
    /// `name` is the ref it answers to. Empty for a provider that can look
    /// up but not list.
    fn list(&self) -> Vec<Pretrained>;

    /// The row `name` refers to, by its name or an alias.
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
    ) -> BunsenResult<Option<Pretrained>> {
        Ok(self.list().into_iter().find(|p| p.matches(name)))
    }

    /// The row `name` refers to, with a cache to hand: what a factory
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
        cache: &PretrainedCache,
    ) -> BunsenResult<Option<Pretrained>> {
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

    /// The qualified id of a row: `provider:ref`.
    fn id(
        &self,
        pretrained: &Pretrained,
    ) -> String {
        format!("{}:{}", self.name(), pretrained.name)
    }

    /// Every qualified id, in listing order.
    fn ids(&self) -> Vec<String> {
        self.list().iter().map(|p| self.id(p)).collect()
    }

    /// The listed rows that instantiate `prefab`, in listing order.
    fn for_prefab(
        &self,
        prefab: &str,
    ) -> Vec<Pretrained> {
        self.list()
            .into_iter()
            .filter(|p| p.prefab.as_deref() == Some(prefab))
            .collect()
    }
}

/// A labelled set of pretrained rows, with one license and origin: the
/// `openai` of `openai/tiny`.
///
/// A group sits in a [`PretrainedTable`], which names its rows
/// `group/name`.
#[derive(Clone, Debug, PartialEq)]
pub struct PretrainedGroup {
    /// The label: the `group` in `group/name`.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The license the rows are distributed under.
    pub license: Option<String>,

    /// Where the table came from.
    pub origin: Option<String>,

    /// The rows, in listing order.
    pub items: Vec<Pretrained>,
}

impl PretrainedGroup {
    /// The row `name` names, by its name or an alias.
    pub fn lookup(
        &self,
        name: &str,
    ) -> Option<&Pretrained> {
        self.lookup_with(name, true)
    }

    /// The row `name` names, by its name or an alias, ignoring ASCII case
    /// unless `case_sensitive`.
    pub fn lookup_with(
        &self,
        name: &str,
        case_sensitive: bool,
    ) -> Option<&Pretrained> {
        self.items
            .iter()
            .find(|p| p.matches_with(name, case_sensitive))
    }

    /// The ref of a row within a table: `group/name`.
    pub fn id(
        &self,
        pretrained: &Pretrained,
    ) -> String {
        format!("{}/{}", self.name, pretrained.name)
    }

    /// Every ref, in listing order.
    pub fn ids(&self) -> Vec<String> {
        self.items.iter().map(|p| self.id(p)).collect()
    }

    /// The rows that instantiate `prefab`, in listing order.
    pub fn for_prefab(
        &self,
        prefab: &str,
    ) -> Vec<&Pretrained> {
        self.items
            .iter()
            .filter(|p| p.prefab.as_deref() == Some(prefab))
            .collect()
    }

    /// A row as the table lists it: named by its ref, `group/name`, with
    /// its aliases as authored.
    fn qualified(
        &self,
        pretrained: &Pretrained,
    ) -> Pretrained {
        let mut row = pretrained.clone();
        row.name = self.id(pretrained);
        row
    }
}

/// A provider over groups: the compiled-in rows, or rows built in code. Its
/// refs are `{group}/{name}`; a bare name or alias
/// searches the groups in order, first hit wins.
///
/// Names compare ignoring ASCII case unless the table is
/// [`case_sensitive`](Self::case_sensitive): `Qwen/Qwen3-0.6B` finds
/// `qwen/qwen3-0.6b`. A listing gives names as the table declares them.
#[derive(Clone, Debug, PartialEq)]
pub struct PretrainedTable {
    /// The provider's name: [`WELL_KNOWN`] for the compiled-in rows.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The groups, in search order.
    pub groups: Vec<PretrainedGroup>,

    /// Whether names compare exactly; by default they ignore ASCII case.
    pub case_sensitive: bool,
}

impl PretrainedTable {
    /// The group called `name`, by the table's case rule.
    pub fn group(
        &self,
        name: &str,
    ) -> Option<&PretrainedGroup> {
        self.groups.iter().find(|g| {
            if self.case_sensitive {
                g.name == name
            } else {
                g.name.eq_ignore_ascii_case(name)
            }
        })
    }

    /// The row a ref names, with its group: `group/name` looks in that
    /// group, by name or alias; anything else searches the groups in order.
    /// Names compare by the table's case rule.
    pub fn get(
        &self,
        name: &str,
    ) -> Option<(&PretrainedGroup, &Pretrained)> {
        let exact = self.case_sensitive;
        if let Some((group, rest)) = name.split_once('/')
            && let Some(group) = self.group(group)
        {
            return group.lookup_with(rest, exact).map(|p| (group, p));
        }
        self.groups
            .iter()
            .find_map(|group| group.lookup_with(name, exact).map(|p| (group, p)))
    }
}

impl PretrainedProvider for PretrainedTable {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn list(&self) -> Vec<Pretrained> {
        self.groups
            .iter()
            .flat_map(|group| group.items.iter().map(|p| group.qualified(p)))
            .collect()
    }

    fn lookup(
        &self,
        name: &str,
    ) -> BunsenResult<Option<Pretrained>> {
        Ok(self.get(name).map(|(group, p)| group.qualified(p)))
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        string::ToString,
        sync::Arc,
    };

    use super::*;
    use crate::data::pretrained::{
        Resource,
        ResourceBundle,
        Source,
    };

    /// A row of one `weights` file at `{base}/{file}`.
    fn entry(
        name: &str,
        aliases: &[&str],
        prefab: &str,
        file: &str,
        base: &str,
    ) -> Pretrained {
        Pretrained {
            name: name.to_string(),
            aliases: aliases.iter().map(|a| a.to_string()).collect(),
            description: "an entry".to_string(),
            license: None,
            origin: None,
            prefab: Some(prefab.to_string()),
            resources: ResourceBundle::new(name).with_resource(Resource {
                size: None,
                dtype: None,
                key: "weights".to_string(),
                file: file.to_string(),
                sha256: None,
                kind: None,
                namespace: "t".to_string(),
                sources: alloc::vec![Source::Url(format!("{base}/{file}"))],
            }),
            presets: Default::default(),
            args: Default::default(),
            tags: Default::default(),
        }
    }

    fn a_small() -> Pretrained {
        entry("small", &["s"], "small", "small.pt", "https://a.example")
    }

    fn b_tiny() -> Pretrained {
        entry("tiny", &[], "tiny", "tiny.pt", "https://b.example")
    }

    fn group_a() -> PretrainedGroup {
        PretrainedGroup {
            name: "a".to_string(),
            description: "group a".to_string(),
            license: Some("MIT".to_string()),
            origin: Some("https://a.example".to_string()),
            items: alloc::vec![
                a_small(),
                entry("large-v1", &[], "large", "large-v1.pt", "https://a.example"),
                entry(
                    "large-v2",
                    &["large"],
                    "large",
                    "large-v2.pt",
                    "https://a.example"
                ),
            ],
        }
    }

    /// The two groups as the well-known table.
    fn table() -> PretrainedTable {
        PretrainedTable {
            name: WELL_KNOWN.to_string(),
            description: "the test rows".to_string(),
            groups: alloc::vec![
                group_a(),
                PretrainedGroup {
                    name: "b".to_string(),
                    description: "group b".to_string(),
                    license: None,
                    origin: None,
                    items: alloc::vec![
                        entry("small", &[], "small", "small.pt", "https://b.example"),
                        b_tiny(),
                    ],
                },
            ],
            case_sensitive: false,
        }
    }

    /// A group answers a row by name or alias, names its rows `group/name`,
    /// finds the rows of a prefab, and survives a round trip through JSON.
    #[test]
    fn test_group_lookups_and_ids() {
        let a = group_a();
        assert_eq!(a.name, "a");
        assert_eq!(a.license.as_deref(), Some("MIT"));
        assert_eq!(a.lookup("s").map(|p| p.name.as_str()), Some("small"));
        assert_eq!(a.lookup("large").map(|p| p.name.as_str()), Some("large-v2"));
        assert!(a.lookup("tiny").is_none());
        assert_eq!(a.id(&a.items[0]), "a/small");
        assert_eq!(a.ids(), ["a/small", "a/large-v1", "a/large-v2"]);
        assert_eq!(
            a.for_prefab("large")
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["large-v1", "large-v2"]
        );
        assert!(a.for_prefab("tiny").is_empty());
        assert_eq!(a.items[0].resources.keys(), ["weights"]);
    }

    /// A table lists its rows by ref, `group/name`, and answers a ref in one
    /// group or a bare name across them, first group first.
    #[test]
    fn test_the_table_lists_refs_and_looks_up() {
        let table = table();
        assert_eq!(table.name(), WELL_KNOWN);
        assert_eq!(table.description(), "the test rows");
        assert_eq!(table.license(), None, "a table's license is its groups'");
        assert!(table.answers_bare_names());

        let names: Vec<String> = table.list().into_iter().map(|p| p.name).collect();
        assert_eq!(
            names,
            ["a/small", "a/large-v1", "a/large-v2", "b/small", "b/tiny"]
        );
        assert_eq!(
            table.ids(),
            [
                "well-known:a/small",
                "well-known:a/large-v1",
                "well-known:a/large-v2",
                "well-known:b/small",
                "well-known:b/tiny",
            ]
        );

        let row = |name: &str| table.lookup(name).unwrap().map(|p| p.name);
        assert_eq!(row("a/small").as_deref(), Some("a/small"));
        assert_eq!(
            row("a/s").as_deref(),
            Some("a/small"),
            "an alias, in a group"
        );
        assert_eq!(row("a/large").as_deref(), Some("a/large-v2"));
        assert_eq!(row("b/small").as_deref(), Some("b/small"));
        assert_eq!(row("small").as_deref(), Some("a/small"), "first group wins");
        assert_eq!(row("tiny").as_deref(), Some("b/tiny"));
        assert_eq!(row("large").as_deref(), Some("a/large-v2"), "a bare alias");
        assert_eq!(row("c/small"), None, "no such group, and no such bare name");
        assert_eq!(row("a/tiny"), None, "the right group, no such row");
        assert_eq!(row("gigantic"), None);

        let listed = table.lookup("s").unwrap().unwrap();
        assert_eq!(listed.aliases, ["s"], "aliases stay as authored");
        assert_eq!(listed.resources.keys(), ["weights"]);
        assert_eq!(table.id(&listed), "well-known:a/small");

        assert_eq!(
            table
                .for_prefab("small")
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["a/small", "b/small"]
        );
        assert_eq!(table.group("b").map(|g| g.items.len()), Some(2));
        assert!(table.get("nope").is_none());
    }

    /// A table is held as a trait object, which is how a factory holds
    /// every provider.
    #[test]
    fn test_a_table_behind_a_trait_object() {
        let provider: Arc<dyn PretrainedProvider> = Arc::new(table());
        assert_eq!(provider.name(), "well-known");
        assert_eq!(provider.list().len(), 5);
        assert_eq!(
            provider
                .lookup("b/tiny")
                .unwrap()
                .map(|p| p.name)
                .as_deref(),
            Some("b/tiny")
        );
        assert!(format!("{provider:?}").contains("PretrainedTable"));
    }

    /// A provider that only lists gets `lookup`, `ids` and `for_prefab`
    /// for free.
    #[test]
    fn test_the_provided_lookup_scans_the_listing() {
        #[derive(Debug)]
        struct Flat;

        impl PretrainedProvider for Flat {
            fn name(&self) -> &str {
                "flat"
            }

            fn description(&self) -> &str {
                "two rows, no groups"
            }

            fn origin(&self) -> Option<&str> {
                Some("https://flat.example")
            }

            fn list(&self) -> Vec<Pretrained> {
                alloc::vec![a_small(), b_tiny()]
            }
        }

        let flat = Flat;
        assert_eq!(flat.origin(), Some("https://flat.example"));
        assert_eq!(flat.ids(), ["flat:small", "flat:tiny"]);
        assert_eq!(
            flat.lookup("s").unwrap().map(|p| p.name).as_deref(),
            Some("small")
        );
        assert_eq!(flat.lookup("large").unwrap(), None);
        assert_eq!(flat.for_prefab("tiny").len(), 1);
    }
}
