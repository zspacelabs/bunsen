//! # Shard-set maps
//!
//! Named collections of shard sets: a static table, and its owned twin.

use std::collections::BTreeMap;

use super::{
    ShardSetDescriptor,
    StaticShardSetDescriptor,
};
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A compiled-in table of shard sets.
#[derive(Debug)]
pub struct StaticShardSetMap {
    /// The table's name.
    pub name: &'static str,

    /// What the table collects.
    pub description: &'static str,

    /// The sets, by reference to their statics.
    pub items: &'static [&'static StaticShardSetDescriptor<'static>],
}

impl StaticShardSetMap {
    /// The owned twin.
    pub fn to_map(&self) -> ShardSetMap {
        ShardSetMap {
            name: self.name.to_string(),
            description: self.description.to_string(),
            items: self
                .items
                .iter()
                .map(|d| (d.name.to_string(), d.to_descriptor()))
                .collect(),
        }
    }

    /// The sets' names, in table order.
    pub fn names(&self) -> Vec<&'static str> {
        self.items.iter().map(|d| d.name).collect()
    }

    /// The set called `name`, as an owned descriptor.
    pub fn lookup(
        &self,
        name: &str,
    ) -> Option<ShardSetDescriptor> {
        self.items
            .iter()
            .find(|d| d.name == name)
            .map(|d| d.to_descriptor())
    }

    /// The set called `name`.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`] naming the sets there are.
    pub fn try_lookup(
        &self,
        name: &str,
    ) -> BunsenResult<ShardSetDescriptor> {
        self.lookup(name)
            .ok_or_else(|| not_found(self.name, name, &self.names()))
    }

    /// The set called `name`.
    ///
    /// # Panics
    /// If there is no such set.
    pub fn expect_lookup(
        &self,
        name: &str,
    ) -> ShardSetDescriptor {
        self.try_lookup(name).unwrap_or_else(|e| panic!("{e}"))
    }
}

impl From<&StaticShardSetMap> for ShardSetMap {
    fn from(map: &StaticShardSetMap) -> Self {
        map.to_map()
    }
}

/// A collection of shard sets, by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardSetMap {
    /// The collection's name.
    pub name: String,

    /// What the collection collects.
    pub description: String,

    /// The sets, by name.
    pub items: BTreeMap<String, ShardSetDescriptor>,
}

impl ShardSetMap {
    /// The sets' names, in name order.
    pub fn names(&self) -> Vec<&str> {
        self.items.keys().map(String::as_str).collect()
    }

    /// The set called `name`.
    pub fn lookup(
        &self,
        name: &str,
    ) -> Option<ShardSetDescriptor> {
        self.items.get(name).cloned()
    }

    /// The set called `name`.
    ///
    /// # Errors
    /// [`BunsenError::ResourceNotFound`] naming the sets there are.
    pub fn try_lookup(
        &self,
        name: &str,
    ) -> BunsenResult<ShardSetDescriptor> {
        self.lookup(name)
            .ok_or_else(|| not_found(&self.name, name, &self.names()))
    }

    /// The set called `name`.
    ///
    /// # Panics
    /// If there is no such set.
    pub fn expect_lookup(
        &self,
        name: &str,
    ) -> ShardSetDescriptor {
        self.try_lookup(name).unwrap_or_else(|e| panic!("{e}"))
    }
}

fn not_found(
    map: &str,
    name: &str,
    names: &[&str],
) -> BunsenError {
    BunsenError::ResourceNotFound(format!(
        "{map}: no shard set {name:?}; there are: {}",
        names.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::shards::StaticShardDigests;

    static TINY: StaticShardSetDescriptor<'static> = StaticShardSetDescriptor {
        name: "tiny",
        description: "a tiny set",
        license: None,
        origin: None,
        base_urls: &["https://a.example/tiny"],
        template: "shard_{index}.bin",
        index_width: 3,
        count: 12,
        format: "bin",
        digests: StaticShardDigests::Unpinned,
    };

    static SMALL: StaticShardSetDescriptor<'static> = StaticShardSetDescriptor {
        name: "small",
        description: "a small set",
        license: None,
        origin: None,
        base_urls: &["https://a.example/small"],
        template: "{index}.bin",
        index_width: 2,
        count: 3,
        format: "bin",
        digests: StaticShardDigests::Unpinned,
    };

    static SETS: StaticShardSetMap = StaticShardSetMap {
        name: "test-sets",
        description: "two sets",
        items: &[&TINY, &SMALL],
    };

    #[test]
    fn test_static_map_lookups() {
        assert_eq!(SETS.names(), vec!["tiny", "small"]);
        assert_eq!(SETS.lookup("small").unwrap().count, 3);
        assert_eq!(SETS.expect_lookup("tiny"), TINY.to_descriptor());
        match SETS.try_lookup("large") {
            Err(BunsenError::ResourceNotFound(m)) => {
                assert_eq!(
                    m,
                    "test-sets: no shard set \"large\"; there are: tiny, small"
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_owned_map_matches_the_static_one() {
        let map = SETS.to_map();
        assert_eq!(map, ShardSetMap::from(&SETS));
        assert_eq!(map.name, "test-sets");
        assert_eq!(map.names(), vec!["small", "tiny"], "owned names are sorted");
        assert_eq!(map.lookup("tiny"), SETS.lookup("tiny"));
        assert_eq!(map.expect_lookup("small").template, "{index}.bin");
        assert!(matches!(
            map.try_lookup("large"),
            Err(BunsenError::ResourceNotFound(_))
        ));
        for d in map.items.values() {
            d.validate().unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "no shard set")]
    fn test_expect_lookup_panics() {
        SETS.expect_lookup("large");
    }
}
