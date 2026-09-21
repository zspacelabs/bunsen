//! # Pretrained providers
//!
//! A provider is a namespace over pretrained rows: the `provider` of
//! `provider:ref`. It names and looks up, and lists when it can. It knows
//! no hook, since the function that loads a kit's models is the kit's and
//! supplies one, and it does not know which resource of a row is the model.
//! A row may name the prefab it instantiates, so "what shape is this" is
//! answered by the row, and "what rows exist for this shape" is derived by
//! scanning the listing. One prefab, many rows, many providers.
//!
//! [`PretrainedProvider`] is the trait; a
//! [`PretrainedFactory`](super::PretrainedFactory) holds providers behind it
//! and dispatches a spec. The compiled-in rows sit behind one provider named
//! [`WELL_KNOWN`], a [`PretrainedTable`] of [`PretrainedGroup`]s: the `openai`
//! of `well-known:openai/tiny`. A group is a labelled set of rows with one
//! license and origin; a table's ref is `{group}/{name}`, and a bare name or
//! alias searches its groups in order. A hub that can answer a ref but not
//! enumerate what it has implements the trait with an empty listing and
//! declines bare names.

use alloc::{
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};
use core::fmt;

use serde::{
    Deserialize,
    Serialize,
};

use super::{
    Pretrained,
    StaticPretrained,
};
use crate::errors::BunsenResult;

/// The name of the provider the compiled-in rows sit behind:
/// `well-known:openai/tiny`.
pub const WELL_KNOWN: &str = "well-known";

/// The name of the provider whose rows a build ships with:
/// `bundled:openai/base`, `bundled:silero/vad`. A kit registers one under
/// its `*-weights` feature, after [`WELL_KNOWN`], so a bare name's identity
/// does not change with the feature; the rows are the same files, served
/// from the bundle rather than fetched.
pub const BUNDLED: &str = "bundled";

/// A namespace of pretrained rows: the `provider` of `provider:ref`.
///
/// Object-safe by construction, so a factory holds any mix of these behind
/// `Arc<dyn PretrainedProvider>`. [`list`](Self::list) and
/// [`lookup`](Self::lookup) are separate so that a provider which cannot
/// enumerate what it has (a hub) can still answer a ref; such a provider
/// also says it does not [answer bare names](Self::answers_bare_names), so
/// a spec with no `provider:` never reaches it.
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

/// A labelled set of pretrained rows, as a compiled-in table spells it: the
/// `openai` of `openai/tiny`.
#[derive(Debug)]
pub struct StaticPretrainedGroup<'a> {
    /// The label: the `group` in `group/name`.
    pub name: &'a str,

    /// A line for a listing.
    pub description: &'a str,

    /// The license the rows are distributed under.
    pub license: Option<&'a str>,

    /// Where the table came from.
    pub origin: Option<&'a str>,

    /// The rows, in listing order.
    pub items: &'a [&'a StaticPretrained<'a>],
}

impl<'a> StaticPretrainedGroup<'a> {
    /// The row `name` names, by its name or an alias.
    pub fn lookup(
        &self,
        name: &str,
    ) -> Option<&'a StaticPretrained<'a>> {
        self.items.iter().copied().find(|p| p.matches(name))
    }

    /// The ref of a row within a table: `group/name`.
    pub fn id(
        &self,
        pretrained: &StaticPretrained<'_>,
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
    ) -> Vec<&'a StaticPretrained<'a>> {
        self.items
            .iter()
            .copied()
            .filter(|p| p.prefab == Some(prefab))
            .collect()
    }

    /// The owned twin.
    ///
    /// # Panics
    /// If a row's maps share a key; a kit's tests pin that none do.
    pub fn to_group(&self) -> PretrainedGroup {
        PretrainedGroup {
            name: self.name.to_string(),
            description: self.description.to_string(),
            license: self.license.map(str::to_string),
            origin: self.origin.map(str::to_string),
            items: self.items.iter().map(|p| p.to_pretrained()).collect(),
        }
    }
}

impl From<&StaticPretrainedGroup<'_>> for PretrainedGroup {
    fn from(group: &StaticPretrainedGroup<'_>) -> Self {
        group.to_group()
    }
}

/// A labelled set of pretrained rows: the `openai` of `openai/tiny`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
        self.items.iter().find(|p| p.matches(name))
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

/// A provider over groups, as a compiled-in table spells it.
#[derive(Debug)]
pub struct StaticPretrainedTable<'a> {
    /// The provider's name: [`WELL_KNOWN`] for the compiled-in rows.
    pub name: &'a str,

    /// A line for a listing.
    pub description: &'a str,

    /// The groups, in search order.
    pub groups: &'a [&'a StaticPretrainedGroup<'a>],
}

impl<'a> StaticPretrainedTable<'a> {
    /// The group called `name`.
    pub fn group(
        &self,
        name: &str,
    ) -> Option<&'a StaticPretrainedGroup<'a>> {
        self.groups.iter().copied().find(|g| g.name == name)
    }

    /// The owned twin: the provider.
    ///
    /// # Panics
    /// If a row's maps share a key; a kit's tests pin that none do.
    pub fn to_table(&self) -> PretrainedTable {
        PretrainedTable {
            name: self.name.to_string(),
            description: self.description.to_string(),
            groups: self.groups.iter().map(|g| g.to_group()).collect(),
        }
    }
}

impl From<&StaticPretrainedTable<'_>> for PretrainedTable {
    fn from(table: &StaticPretrainedTable<'_>) -> Self {
        table.to_table()
    }
}

/// A provider over groups: the compiled-in rows, and any table read from
/// a manifest. Its refs are `{group}/{name}`; a bare name or alias
/// searches the groups in order, first hit wins.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PretrainedTable {
    /// The provider's name: [`WELL_KNOWN`] for the compiled-in rows.
    pub name: String,

    /// A line for a listing.
    pub description: String,

    /// The groups, in search order.
    pub groups: Vec<PretrainedGroup>,
}

impl PretrainedTable {
    /// The group called `name`.
    pub fn group(
        &self,
        name: &str,
    ) -> Option<&PretrainedGroup> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// The row a ref names, with its group: `group/name` looks in that
    /// group, by name or alias; anything else searches the groups in order.
    pub fn get(
        &self,
        name: &str,
    ) -> Option<(&PretrainedGroup, &Pretrained)> {
        if let Some((group, rest)) = name.split_once('/')
            && let Some(group) = self.group(group)
        {
            return group.lookup(rest).map(|p| (group, p));
        }
        self.groups
            .iter()
            .find_map(|group| group.lookup(name).map(|p| (group, p)))
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
    use alloc::sync::Arc;

    use super::*;
    use crate::data::pretrained::{
        StaticBase,
        StaticResource,
        StaticResourceMap,
    };

    macro_rules! one_file {
        ($name:ident, $map:literal, $file:literal, $base:literal) => {
            static $name: StaticResourceMap<'static> = StaticResourceMap {
                name: $map,
                description: "one file",
                license: None,
                origin: None,
                namespace: "t",
                bases: &[StaticBase::Url($base)],
                resources: &[StaticResource {
                    key: "weights",
                    file: $file,
                    sha256: None,
                    kind: None,
                    sources: &[],
                }],
            };
        };
    }

    one_file!(A_SMALL_MAP, "a/small.pt", "small.pt", "https://a.example");
    one_file!(
        A_LARGE_V1_MAP,
        "a/large-v1.pt",
        "large-v1.pt",
        "https://a.example"
    );
    one_file!(
        A_LARGE_V2_MAP,
        "a/large-v2.pt",
        "large-v2.pt",
        "https://a.example"
    );
    one_file!(B_SMALL_MAP, "b/small.pt", "small.pt", "https://b.example");
    one_file!(B_TINY_MAP, "b/tiny.pt", "tiny.pt", "https://b.example");

    const fn entry(
        name: &'static str,
        aliases: &'static [&'static str],
        prefab: &'static str,
        maps: &'static [&'static StaticResourceMap<'static>],
    ) -> StaticPretrained<'static> {
        StaticPretrained {
            name,
            aliases,
            description: "an entry",
            license: None,
            origin: None,
            prefab: Some(prefab),
            maps,
        }
    }

    static A_SMALL: StaticPretrained<'static> = entry("small", &["s"], "small", &[&A_SMALL_MAP]);
    static A_LARGE_V1: StaticPretrained<'static> =
        entry("large-v1", &[], "large", &[&A_LARGE_V1_MAP]);
    static A_LARGE_V2: StaticPretrained<'static> =
        entry("large-v2", &["large"], "large", &[&A_LARGE_V2_MAP]);
    static B_SMALL: StaticPretrained<'static> = entry("small", &[], "small", &[&B_SMALL_MAP]);
    static B_TINY: StaticPretrained<'static> = entry("tiny", &[], "tiny", &[&B_TINY_MAP]);

    static A: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
        name: "a",
        description: "group a",
        license: Some("MIT"),
        origin: Some("https://a.example"),
        items: &[&A_SMALL, &A_LARGE_V1, &A_LARGE_V2],
    };
    static B: StaticPretrainedGroup<'static> = StaticPretrainedGroup {
        name: "b",
        description: "group b",
        license: None,
        origin: None,
        items: &[&B_SMALL, &B_TINY],
    };
    /// The two groups as the well-known table.
    pub(crate) static TABLE: StaticPretrainedTable<'static> = StaticPretrainedTable {
        name: WELL_KNOWN,
        description: "the test rows",
        groups: &[&A, &B],
    };

    #[test]
    fn test_group_lookups_and_ids() {
        assert_eq!(A.lookup("s").map(|p| p.name), Some("small"));
        assert_eq!(A.lookup("large").map(|p| p.name), Some("large-v2"));
        assert!(A.lookup("tiny").is_none());
        assert_eq!(A.id(&A_SMALL), "a/small");
        assert_eq!(A.ids(), vec!["a/small", "a/large-v1", "a/large-v2"]);
        assert_eq!(
            A.for_prefab("large")
                .iter()
                .map(|p| p.name)
                .collect::<Vec<_>>(),
            vec!["large-v1", "large-v2"]
        );
        assert!(A.for_prefab("tiny").is_empty());
    }

    #[test]
    fn test_owned_group_matches_the_static_one() {
        let owned = A.to_group();
        assert_eq!(owned, PretrainedGroup::from(&A));
        assert_eq!(owned.name, "a");
        assert_eq!(owned.license.as_deref(), Some("MIT"));
        assert_eq!(owned.items.len(), 3);
        assert_eq!(
            owned.lookup("large").map(|p| p.name.as_str()),
            Some("large-v2")
        );
        assert_eq!(owned.ids(), A.ids());
        assert_eq!(owned.for_prefab("large").len(), 2);
        assert_eq!(owned.id(&owned.items[0]), "a/small");
        assert_eq!(owned.items[0].resources.keys(), ["weights"]);
        let json = serde_json::to_string(&owned).unwrap();
        assert_eq!(
            serde_json::from_str::<PretrainedGroup>(&json).unwrap(),
            owned
        );
    }

    /// A table lists its rows by ref, `group/name`, and answers a ref in one
    /// group or a bare name across them, first group first.
    #[test]
    fn test_the_table_lists_refs_and_looks_up() {
        let table = TABLE.to_table();
        assert_eq!(table, PretrainedTable::from(&TABLE));
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
        assert_eq!(TABLE.group("b").map(|g| g.items.len()), Some(2));
        assert!(table.get("nope").is_none());

        let json = serde_json::to_string(&table).unwrap();
        assert_eq!(
            serde_json::from_str::<PretrainedTable>(&json).unwrap(),
            table
        );
    }

    /// A table is held as a trait object, which is how a factory holds
    /// every provider.
    #[test]
    fn test_a_table_behind_a_trait_object() {
        let provider: Arc<dyn PretrainedProvider> = Arc::new(TABLE.to_table());
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
                alloc::vec![A_SMALL.to_pretrained(), B_TINY.to_pretrained()]
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
