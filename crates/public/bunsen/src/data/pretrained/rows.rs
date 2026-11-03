//! Pretrained rows: what a name in a provider is.

use alloc::{
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
    Fuse,
    ResourceBundle,
    StaticResourceBundle,
};
use crate::errors::BunsenResult;

/// A pretrained, as a compiled-in table spells it.
///
/// Its [`resource_bundles`](Self::resource_bundles) are fused strictly into one
/// resource bundle when it becomes the owned twin, a [`Pretrained`]
/// ([`to_pretrained`](Self::to_pretrained)). A file many rows share is a
/// resource bundle of its own that each row references; a one-file model is a
/// row of one resource bundle.
#[derive(Debug)]
pub struct StaticPretrained<'a> {
    /// Its name, unique within its provider.
    pub name: &'a str,

    /// Other names it answers to.
    pub aliases: &'a [&'a str],

    /// A line for a listing.
    pub description: &'a str,

    /// The license it is distributed under.
    pub license: Option<&'a str>,

    /// Where it is published.
    pub origin: Option<&'a str>,

    /// The prefab it instantiates, by name in the kit's prefab resource bundle,
    /// when the kit keeps one. The generic layer never reads it.
    pub prefab: Option<&'a str>,

    /// The resource bundles it is made of, fused strictly: a repeated key
    /// across them is a table's mistake, which a kit's tests pin.
    pub resource_bundles: &'a [&'a StaticResourceBundle<'a>],
}

impl StaticPretrained<'_> {
    /// Does `name` name this pretrained, by its name or an alias?
    pub fn matches(
        &self,
        name: &str,
    ) -> bool {
        self.name == name || self.aliases.contains(&name)
    }

    /// The resource bundles fused strictly into one, named and described as the
    /// row is.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), as
    /// [`ResourceBundle::fuse`] under [`Fuse::Strict`], naming a key two of the
    /// resource bundles share.
    pub fn try_to_resource_bundle(&self) -> BunsenResult<ResourceBundle> {
        let mut fused = ResourceBundle::new(self.name);
        fused.description = self.description.to_string();
        fused.license = self.license.map(str::to_string);
        fused.origin = self.origin.map(str::to_string);
        for map in self.resource_bundles {
            fused = fused.fuse(map.to_resource_bundle(), Fuse::Strict)?;
        }
        Ok(fused)
    }

    /// The resource bundles fused strictly into one.
    ///
    /// # Panics
    /// If two of the resource bundles share a key; a kit's tests pin that none
    /// do.
    pub fn to_resource_bundle(&self) -> ResourceBundle {
        self.try_to_resource_bundle()
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// The owned twin.
    ///
    /// # Errors
    /// As [`try_to_resource_bundle`](Self::try_to_resource_bundle).
    pub fn try_to_pretrained(&self) -> BunsenResult<Pretrained> {
        Ok(Pretrained {
            name: self.name.to_string(),
            aliases: self.aliases.iter().map(|s| s.to_string()).collect(),
            description: self.description.to_string(),
            license: self.license.map(str::to_string),
            origin: self.origin.map(str::to_string),
            prefab: self.prefab.map(str::to_string),
            resources: self.try_to_resource_bundle()?,
        })
    }

    /// The owned twin.
    ///
    /// # Panics
    /// As [`to_resource_bundle`](Self::to_resource_bundle).
    pub fn to_pretrained(&self) -> Pretrained {
        self.try_to_pretrained().unwrap_or_else(|e| panic!("{e}"))
    }
}

impl From<&StaticPretrained<'_>> for Pretrained {
    fn from(pretrained: &StaticPretrained<'_>) -> Self {
        pretrained.to_pretrained()
    }
}

/// A pretrained row: a name in a provider, over a resource bundle.
///
/// What a name in a provider is: listing fields, aliases, the prefab it
/// instantiates when the kit keeps prefabs, and the resource bundles it is
/// made of, fused strictly into one. A row is plain data. It knows no
/// hook, since the function that loads a kit's models is the kit's and
/// supplies one, and it does not know which of its resources is the model.
///
/// A compiled-in table spells a row as a [`StaticPretrained`], or it comes
/// from a manifest through serde. A
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
}

impl Pretrained {
    /// Does `name` name this pretrained, by its name or an alias?
    pub fn matches(
        &self,
        name: &str,
    ) -> bool {
        self.name == name || self.aliases.iter().any(|a| a == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data::pretrained::{
            StaticBase,
            StaticResource,
        },
        errors::{
            BunsenErrorKind,
            LookupError,
            testing::ErrorMatcher,
        },
    };

    macro_rules! one_file {
        ($name:ident, $map:literal, $key:literal, $file:literal) => {
            static $name: StaticResourceBundle<'static> = StaticResourceBundle {
                name: $map,
                description: "one file",
                license: None,
                origin: None,
                namespace: "t",
                bases: &[StaticBase::Url("https://t.example")],
                resources: &[StaticResource {
                    key: $key,
                    file: $file,
                    sha256: None,
                    kind: None,
                    sources: &[],
                }],
            };
        };
    }

    one_file!(CHECKPOINT, "t/small.pt", "checkpoint", "small.pt");
    one_file!(VOCABULARY, "t/vocab.txt", "vocabulary", "vocab.txt");
    one_file!(ANOTHER_CHECKPOINT, "t/other.pt", "checkpoint", "other.pt");

    static SMALL: StaticPretrained<'static> = StaticPretrained {
        name: "small",
        aliases: &["s"],
        description: "a small model",
        license: Some("MIT"),
        origin: Some("https://t.example"),
        prefab: Some("small"),
        resource_bundles: &[&CHECKPOINT, &VOCABULARY],
    };

    static CLASH: StaticPretrained<'static> = StaticPretrained {
        name: "clash",
        aliases: &[],
        description: "two checkpoints",
        license: None,
        origin: None,
        prefab: None,
        resource_bundles: &[&CHECKPOINT, &ANOTHER_CHECKPOINT],
    };

    /// A row fuses its resource bundles under its own name and listing fields,
    /// and answers to its aliases.
    #[test]
    fn test_to_map_fuses_under_the_rows_name() {
        assert!(SMALL.matches("small"));
        assert!(SMALL.matches("s"));
        assert!(!SMALL.matches("large"));

        let map = SMALL.to_resource_bundle();
        assert_eq!(map.name, "small");
        assert_eq!(map.description, "a small model");
        assert_eq!(map.license.as_deref(), Some("MIT"));
        assert_eq!(map.keys(), ["checkpoint", "vocabulary"]);
        assert_eq!(map.get("checkpoint").unwrap().file, "small.pt");
        assert_eq!(
            map.get("vocabulary").unwrap().urls(),
            ["https://t.example/vocab.txt"]
        );
        map.validate().unwrap();

        let owned = SMALL.to_pretrained();
        assert_eq!(owned, Pretrained::from(&SMALL));
        assert_eq!(owned.prefab.as_deref(), Some("small"));
        assert!(owned.matches("s"));
        assert_eq!(owned.resources, map);
        let json = serde_json::to_string(&owned).unwrap();
        assert_eq!(serde_json::from_str::<Pretrained>(&json).unwrap(), owned);
    }

    /// Two resource bundles that share a key are a table's mistake: named by
    /// the fallible form, a panic in the other.
    #[test]
    fn test_a_repeated_key_is_refused() {
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_eq("duplicate resource \"checkpoint\"")
            .has_cause::<LookupError>()
            .assert_err(&CLASH.try_to_resource_bundle());
        assert!(CLASH.try_to_pretrained().is_err());
    }

    #[test]
    #[should_panic(expected = "duplicate resource \"checkpoint\"")]
    fn test_to_map_panics_on_a_repeated_key() {
        CLASH.to_resource_bundle();
    }
}
