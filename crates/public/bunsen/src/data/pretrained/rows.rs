//! Pretrained rows: what a name in a provider is.

use alloc::{
    string::String,
    vec::Vec,
};

use serde::{
    Deserialize,
    Serialize,
};

use super::{
    PresetSet,
    ResourceBundle,
};

/// A pretrained row: a name in a provider, over a resource bundle.
///
/// What a name in a provider is: listing fields, aliases, the prefab it
/// instantiates when the kit keeps prefabs, and the resource bundles it is
/// made of, fused strictly into one. A row is plain data. It knows no
/// hook, since the function that loads a kit's models is the kit's and
/// supplies one, and it does not know which of its resources is the model.
///
/// A kit's [`PretrainedCatalog`](super::PretrainedCatalog) spells its rows
/// as data, or one comes from a manifest through serde. A
/// [`PretrainedProvider`](super::PretrainedProvider) hands rows out from
/// its `list`, `lookup` and `resolve`; a
/// [`PretrainedTable`](super::PretrainedTable) keeps them in groups and
/// names them `group/name`. A factory wraps the row it found in a
/// [`PretrainedRef::Named`](super::PretrainedRef::Named), the next step
/// toward a model. The [`prefab`](Self::prefab) field joins the row to the
/// kit's [`StaticPreFabMap`](super::StaticPreFabMap), through
/// [`PretrainedRef::prefab`](super::PretrainedRef::prefab).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pretrained {
    /// Its name, unique within its provider.
    pub name: String,

    /// Other names it answers to.
    pub aliases: Vec<String>,

    /// A line for a listing.
    pub description: String,

    /// The license it is distributed under.
    pub license: Option<String>,

    /// Where it is published.
    pub origin: Option<String>,

    /// The prefab it instantiates, by name in the kit's prefab resource bundle,
    /// when the kit keeps one.
    pub prefab: Option<String>,

    /// What it is made of, fused.
    pub resources: ResourceBundle,

    /// Its presets: named deltas over the config it is built from, and its
    /// default selection.
    #[serde(default, skip_serializing_if = "PresetSet::is_empty")]
    pub presets: PresetSet,
}

impl Pretrained {
    /// Does `name` name this pretrained, by its name or an alias?
    pub fn matches(
        &self,
        name: &str,
    ) -> bool {
        self.matches_with(name, true)
    }

    /// Does `name` name this pretrained, by its name or an alias, comparing
    /// exactly when `case_sensitive` and ignoring ASCII case otherwise?
    pub fn matches_with(
        &self,
        name: &str,
        case_sensitive: bool,
    ) -> bool {
        let eq = |a: &str| {
            if case_sensitive {
                a == name
            } else {
                a.eq_ignore_ascii_case(name)
            }
        };
        eq(&self.name) || self.aliases.iter().any(|a| eq(a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::pretrained::Resource;

    /// A row answers to its name and aliases, and survives a round trip
    /// through JSON.
    #[test]
    fn test_matches_and_round_trip() {
        let row = Pretrained {
            name: "small".to_string(),
            aliases: vec!["s".to_string()],
            description: "a small model".to_string(),
            license: Some("MIT".to_string()),
            origin: Some("https://t.example".to_string()),
            prefab: Some("small".to_string()),
            resources: ResourceBundle::new("small")
                .with_resource(Resource::given("checkpoint", "/t/small.pt")),
            presets: PresetSet::default(),
        };
        assert!(row.matches("small"));
        assert!(row.matches("s"));
        assert!(!row.matches("large"));
        assert!(!row.matches("Small"));
        assert!(row.matches_with("SMALL", false));
        assert!(row.matches_with("S", false));
        assert!(!row.matches_with("S", true));
        let json = serde_json::to_string(&row).unwrap();
        assert_eq!(serde_json::from_str::<Pretrained>(&json).unwrap(), row);
    }
}
