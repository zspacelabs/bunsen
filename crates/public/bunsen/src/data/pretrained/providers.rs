//! # Pretrained providers
//!
//! A provider is a namespace over pretrained rows: the `openai` in
//! `openai/tiny.en`. It names and lists. It knows no hook, since the
//! function that loads a kit's models is the kit's and supplies one, and it
//! does not know which resource of a row is the model. A row may name the
//! prefab it instantiates, so "what shape is this" is answered by the row,
//! and "what rows exist for this shape" is derived by scanning the
//! providers ([`pretrained_for_prefab`]). One prefab, many rows, many
//! providers.

use alloc::{
    format,
    string::{
        String,
        ToString,
    },
    vec::Vec,
};

use serde::{
    Deserialize,
    Serialize,
};

use super::{
    Pretrained,
    StaticPretrained,
};

/// A namespace of pretrained rows, as a compiled-in table spells it.
#[derive(Debug)]
pub struct StaticPretrainedProvider<'a> {
    /// The namespace: the `provider` in `provider/name`.
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

impl<'a> StaticPretrainedProvider<'a> {
    /// The row `name` names, by its name or an alias.
    pub fn lookup(
        &self,
        name: &str,
    ) -> Option<&'a StaticPretrained<'a>> {
        self.items.iter().copied().find(|p| p.matches(name))
    }

    /// The qualified id of a row: `provider/name`.
    pub fn id(
        &self,
        pretrained: &StaticPretrained<'_>,
    ) -> String {
        format!("{}/{}", self.name, pretrained.name)
    }

    /// Every qualified id, in listing order.
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
    pub fn to_provider(&self) -> PretrainedProvider {
        PretrainedProvider {
            name: self.name.to_string(),
            description: self.description.to_string(),
            license: self.license.map(str::to_string),
            origin: self.origin.map(str::to_string),
            items: self.items.iter().map(|p| p.to_pretrained()).collect(),
        }
    }
}

impl From<&StaticPretrainedProvider<'_>> for PretrainedProvider {
    fn from(provider: &StaticPretrainedProvider<'_>) -> Self {
        provider.to_provider()
    }
}

/// A namespace of pretrained rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PretrainedProvider {
    /// The namespace: the `provider` in `provider/name`.
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

impl PretrainedProvider {
    /// The row `name` names, by its name or an alias.
    pub fn lookup(
        &self,
        name: &str,
    ) -> Option<&Pretrained> {
        self.items.iter().find(|p| p.matches(name))
    }

    /// The qualified id of a row: `provider/name`.
    pub fn id(
        &self,
        pretrained: &Pretrained,
    ) -> String {
        format!("{}/{}", self.name, pretrained.name)
    }

    /// Every qualified id, in listing order.
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
}

/// The provider called `name`.
pub fn provider<'a>(
    providers: &[&'a StaticPretrainedProvider<'a>],
    name: &str,
) -> Option<&'a StaticPretrainedProvider<'a>> {
    providers.iter().copied().find(|p| p.name == name)
}

/// The row `name` names under `provider`, or under any provider when none
/// is given and exactly one has it.
///
/// A bare name two providers both answer to is ambiguous, and `None`: the
/// caller has to qualify it.
pub fn lookup_pretrained<'a>(
    providers: &[&'a StaticPretrainedProvider<'a>],
    provider: Option<&str>,
    name: &str,
) -> Option<(&'a StaticPretrainedProvider<'a>, &'a StaticPretrained<'a>)> {
    match provider {
        Some(provider) => {
            let provider = self::provider(providers, provider)?;
            provider.lookup(name).map(|p| (provider, p))
        }
        None => {
            let mut hits = providers
                .iter()
                .copied()
                .filter_map(|provider| provider.lookup(name).map(|p| (provider, p)));
            let first = hits.next()?;
            match hits.next() {
                Some(_) => None,
                None => Some(first),
            }
        }
    }
}

/// Every qualified id across `providers`, for a "did you mean" listing.
pub fn available_ids(providers: &[&StaticPretrainedProvider<'_>]) -> Vec<String> {
    providers.iter().flat_map(|p| p.ids()).collect()
}

/// Every row across `providers` that instantiates `prefab`, with its
/// provider: the derived "what rows exist for this shape".
pub fn pretrained_for_prefab<'a>(
    providers: &[&'a StaticPretrainedProvider<'a>],
    prefab: &str,
) -> Vec<(&'a StaticPretrainedProvider<'a>, &'a StaticPretrained<'a>)> {
    providers
        .iter()
        .copied()
        .flat_map(|p| p.for_prefab(prefab).into_iter().map(move |d| (p, d)))
        .collect()
}

#[cfg(test)]
mod tests {
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

    static A: StaticPretrainedProvider<'static> = StaticPretrainedProvider {
        name: "a",
        description: "provider a",
        license: Some("MIT"),
        origin: Some("https://a.example"),
        items: &[&A_SMALL, &A_LARGE_V1, &A_LARGE_V2],
    };
    static B: StaticPretrainedProvider<'static> = StaticPretrainedProvider {
        name: "b",
        description: "provider b",
        license: None,
        origin: None,
        items: &[&B_SMALL, &B_TINY],
    };
    static PROVIDERS: &[&StaticPretrainedProvider<'static>] = &[&A, &B];

    #[test]
    fn test_provider_lookups_and_ids() {
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

    /// A qualified name looks in one provider; a bare name looks across all
    /// and is `None` when two answer.
    #[test]
    fn test_lookup_pretrained_across_providers() {
        assert!(provider(PROVIDERS, "b").is_some());
        assert!(provider(PROVIDERS, "c").is_none());

        let (p, d) = lookup_pretrained(PROVIDERS, Some("b"), "small").unwrap();
        assert_eq!((p.name, d.name), ("b", "small"));
        let (p, d) = lookup_pretrained(PROVIDERS, None, "tiny").unwrap();
        assert_eq!((p.name, d.name), ("b", "tiny"));
        let (p, d) = lookup_pretrained(PROVIDERS, None, "large").unwrap();
        assert_eq!((p.name, d.name), ("a", "large-v2"));
        assert!(
            lookup_pretrained(PROVIDERS, None, "small").is_none(),
            "both providers have a `small`"
        );
        assert!(lookup_pretrained(PROVIDERS, Some("c"), "small").is_none());
        assert!(lookup_pretrained(PROVIDERS, None, "gigantic").is_none());

        assert_eq!(
            available_ids(PROVIDERS),
            vec!["a/small", "a/large-v1", "a/large-v2", "b/small", "b/tiny"]
        );
        assert_eq!(
            pretrained_for_prefab(PROVIDERS, "small")
                .iter()
                .map(|(p, d)| format!("{}/{}", p.name, d.name))
                .collect::<Vec<_>>(),
            vec!["a/small", "b/small"]
        );
    }

    #[test]
    fn test_owned_provider_matches_the_static_one() {
        let owned = A.to_provider();
        assert_eq!(owned, PretrainedProvider::from(&A));
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
            serde_json::from_str::<PretrainedProvider>(&json).unwrap(),
            owned
        );
    }
}
