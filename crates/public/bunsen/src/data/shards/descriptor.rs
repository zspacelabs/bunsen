//! # Shard-set descriptors
//!
//! What a shard set is: its name, where its shards come from, how they are
//! named and numbered, and whether they are pinned.

use std::{
    collections::BTreeSet,
    fmt,
};

use burn::tensor::Slice;
use serde::{
    Deserialize,
    Serialize,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The placeholder a shard template carries for the zero-padded index.
pub const INDEX_PLACEHOLDER: &str = "{index}";

/// A shard's number within its set: `0..count`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShardId(pub usize);

impl From<usize> for ShardId {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl fmt::Display for ShardId {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// How a set's shards are pinned.
///
/// Only [`Unpinned`](Self::Unpinned) exists today: a fetched shard is checked
/// against its `Content-Length` alone. A fetched manifest of digests is the
/// planned addition, which is why this is an enum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ShardDigests {
    /// No digests: a shard is trusted on its length.
    #[default]
    Unpinned,
}

/// A shard set, as a compiled-in table spells it.
///
/// [`to_descriptor`](Self::to_descriptor) gives the owned twin,
/// [`ShardSetDescriptor`], which is where the behavior lives.
#[derive(Debug)]
pub struct StaticShardSetDescriptor<'a> {
    /// The set's name, unique within its map.
    pub name: &'a str,

    /// What the set is.
    pub description: &'a str,

    /// The data's license.
    pub license: Option<&'a str>,

    /// Where the set is published.
    pub origin: Option<&'a str>,

    /// Base URLs the shard file name is appended to: mirrors, tried in order.
    pub base_urls: &'a [&'a str],

    /// The shard file name, with [`INDEX_PLACEHOLDER`] where the index goes.
    pub template: &'a str,

    /// The zero-padded width of the index in a file name.
    pub index_width: usize,

    /// How many shards there are: ids are `0..count`.
    pub count: usize,

    /// The shards' format, as a label (`"parquet"`).
    pub format: &'a str,

    /// How the shards are pinned.
    pub digests: ShardDigests,
}

impl StaticShardSetDescriptor<'_> {
    /// The owned twin.
    pub fn to_descriptor(&self) -> ShardSetDescriptor {
        ShardSetDescriptor {
            name: self.name.to_string(),
            description: self.description.to_string(),
            license: self.license.map(str::to_string),
            origin: self.origin.map(str::to_string),
            base_urls: self.base_urls.iter().map(|s| s.to_string()).collect(),
            template: self.template.to_string(),
            index_width: self.index_width,
            count: self.count,
            format: self.format.to_string(),
            digests: self.digests,
        }
    }
}

impl From<&StaticShardSetDescriptor<'_>> for ShardSetDescriptor {
    fn from(descriptor: &StaticShardSetDescriptor<'_>) -> Self {
        descriptor.to_descriptor()
    }
}

/// A shard set: what it is, and how its shards are named and reached.
///
/// Built from a [`StaticShardSetDescriptor`], deserialized, or assembled at
/// runtime. [`validate`](Self::validate) checks a hand-built one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardSetDescriptor {
    /// The set's name, unique within its map.
    pub name: String,

    /// What the set is.
    pub description: String,

    /// The data's license.
    pub license: Option<String>,

    /// Where the set is published.
    pub origin: Option<String>,

    /// Base URLs the shard file name is appended to: mirrors, tried in order.
    pub base_urls: Vec<String>,

    /// The shard file name, with [`INDEX_PLACEHOLDER`] where the index goes.
    pub template: String,

    /// The zero-padded width of the index in a file name.
    pub index_width: usize,

    /// How many shards there are: ids are `0..count`.
    pub count: usize,

    /// The shards' format, as a label (`"parquet"`).
    pub format: String,

    /// How the shards are pinned.
    pub digests: ShardDigests,
}

impl ShardSetDescriptor {
    /// Checks the descriptor hangs together: a template with exactly one
    /// [`INDEX_PLACEHOLDER`], at least one base URL, at least one shard.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] naming the first problem.
    pub fn validate(&self) -> BunsenResult<()> {
        let placeholders = self.template.matches(INDEX_PLACEHOLDER).count();
        if placeholders != 1 {
            return Err(BunsenError::Invalid(format!(
                "{}: template {:?} must contain {INDEX_PLACEHOLDER} exactly once, not {placeholders} times",
                self.name, self.template
            )));
        }
        if self.base_urls.is_empty() {
            return Err(BunsenError::Invalid(format!("{}: no base URL", self.name)));
        }
        if self.count == 0 {
            return Err(BunsenError::Invalid(format!("{}: no shards", self.name)));
        }
        Ok(())
    }

    /// Every id, in order.
    pub fn ids(&self) -> impl Iterator<Item = ShardId> {
        (0..self.count).map(ShardId)
    }

    /// `true` when `id` is within the set.
    pub fn contains(
        &self,
        id: ShardId,
    ) -> bool {
        id.0 < self.count
    }

    /// The file name of shard `id`: the template with the zero-padded index.
    pub fn file_name(
        &self,
        id: ShardId,
    ) -> String {
        let index = format!("{:0width$}", id.0, width = self.index_width);
        self.template.replace(INDEX_PLACEHOLDER, &index)
    }

    /// The id a file name spells, when it is one of this set's.
    pub fn parse_file_name(
        &self,
        file_name: &str,
    ) -> Option<ShardId> {
        let (prefix, suffix) = self.template.split_once(INDEX_PLACEHOLDER)?;
        let digits = file_name.strip_prefix(prefix)?.strip_suffix(suffix)?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let index: usize = digits.parse().ok()?;
        (index < self.count).then_some(ShardId(index))
    }

    /// Where shard `id` can be fetched from: one URL per base URL, in order.
    pub fn urls(
        &self,
        id: ShardId,
    ) -> Vec<String> {
        let file_name = self.file_name(id);
        self.base_urls
            .iter()
            .map(|base| format!("{}/{file_name}", base.trim_end_matches('/')))
            .collect()
    }

    /// The lowercase hex SHA-256 shard `id` must have, when the set is pinned.
    pub fn digest(
        &self,
        _id: ShardId,
    ) -> Option<&str> {
        match self.digests {
            ShardDigests::Unpinned => None,
        }
    }

    /// The ids `slices` name, sorted and without repeats.
    ///
    /// Each slice is resolved against [`count`](Self::count): a negative
    /// bound counts from the end, an open end is the end. Steps must be
    /// positive.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] for a bound outside `0..=count` or a
    /// reversed slice.
    pub fn select(
        &self,
        slices: &[Slice],
    ) -> BunsenResult<Vec<ShardId>> {
        let count = self.count;
        let resolve = |bound: isize| -> BunsenResult<usize> {
            let resolved = if bound < 0 {
                count as isize + bound
            } else {
                bound
            };
            if resolved < 0 || resolved > count as isize {
                return Err(BunsenError::InvalidArgument {
                    msg: format!(
                        "shard index {bound} is out of range for {}, which has {count} shards",
                        self.name
                    ),
                });
            }
            Ok(resolved as usize)
        };

        let mut ids = BTreeSet::new();
        for slice in slices {
            if slice.is_reversed() {
                return Err(BunsenError::InvalidArgument {
                    msg: format!("shard slice {slice} is reversed; shards are selected in order"),
                });
            }
            let start = resolve(slice.start)?;
            let end = match slice.end {
                Some(end) => resolve(end)?,
                None => count,
            };
            let step = slice.step().unsigned_abs().max(1);
            ids.extend((start..end).step_by(step).map(ShardId));
        }
        Ok(ids.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn tiny() -> StaticShardSetDescriptor<'static> {
        StaticShardSetDescriptor {
            name: "tiny",
            description: "a tiny set",
            license: Some("CC0"),
            origin: Some("https://example.invalid/tiny"),
            base_urls: &["https://a.example/tiny/", "https://b.example/mirror"],
            template: "shard_{index}.bin",
            index_width: 3,
            count: 12,
            format: "bin",
            digests: ShardDigests::Unpinned,
        }
    }

    #[test]
    fn test_static_to_owned_round_trip() {
        let s = tiny();
        let d = s.to_descriptor();
        assert_eq!(d.name, "tiny");
        assert_eq!(d.license.as_deref(), Some("CC0"));
        assert_eq!(d.base_urls.len(), 2);
        assert_eq!(d.template, "shard_{index}.bin");
        assert_eq!((d.index_width, d.count), (3, 12));
        assert_eq!(d.digests, ShardDigests::Unpinned);
        assert_eq!(ShardSetDescriptor::from(&s), d);
        d.validate().unwrap();

        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(
            serde_json::from_str::<ShardSetDescriptor>(&json).unwrap(),
            d
        );
    }

    #[test]
    fn test_validate_names_the_problem() {
        let mut d = tiny().to_descriptor();
        d.template = "shard.bin".to_string();
        assert!(matches!(d.validate(), Err(BunsenError::Invalid(m)) if m.contains("exactly once")));
        d.template = "{index}-{index}".to_string();
        assert!(matches!(d.validate(), Err(BunsenError::Invalid(_))));

        let mut d = tiny().to_descriptor();
        d.base_urls.clear();
        assert!(matches!(d.validate(), Err(BunsenError::Invalid(m)) if m.contains("no base URL")));

        let mut d = tiny().to_descriptor();
        d.count = 0;
        assert!(matches!(d.validate(), Err(BunsenError::Invalid(m)) if m.contains("no shards")));
    }

    #[test]
    fn test_names_and_urls() {
        let d = tiny().to_descriptor();
        assert_eq!(d.file_name(ShardId(0)), "shard_000.bin");
        assert_eq!(d.file_name(ShardId(11)), "shard_011.bin");
        assert_eq!(
            d.urls(ShardId(7)),
            vec![
                "https://a.example/tiny/shard_007.bin",
                "https://b.example/mirror/shard_007.bin",
            ]
        );
        assert_eq!(d.digest(ShardId(7)), None);
        assert_eq!(ShardId(7).to_string(), "7");
        assert_eq!(ShardId::from(3), ShardId(3));

        assert_eq!(d.parse_file_name("shard_007.bin"), Some(ShardId(7)));
        assert_eq!(d.parse_file_name("shard_7.bin"), Some(ShardId(7)));
        assert_eq!(d.parse_file_name("shard_012.bin"), None, "past count");
        assert_eq!(d.parse_file_name("shard_.bin"), None);
        assert_eq!(d.parse_file_name("shard_0x7.bin"), None);
        assert_eq!(d.parse_file_name("other_007.bin"), None);
        assert_eq!(d.parse_file_name("shard_007.bin.partial"), None);

        assert!(d.contains(ShardId(11)));
        assert!(!d.contains(ShardId(12)));
        assert_eq!(d.ids().count(), 12);
        assert_eq!(d.ids().last(), Some(ShardId(11)));
    }

    #[test]
    fn test_select_resolves_slices_in_order_without_repeats() {
        let d = tiny().to_descriptor();
        let ids = |v: &[usize]| v.iter().copied().map(ShardId).collect::<Vec<_>>();

        assert_eq!(d.select(&[]).unwrap(), ids(&[]));
        assert_eq!(d.select(&[Slice::index(3)]).unwrap(), ids(&[3]));
        assert_eq!(
            d.select(&[Slice::from(..3), Slice::index(1)]).unwrap(),
            ids(&[0, 1, 2])
        );
        assert_eq!(d.select(&[Slice::from(9..)]).unwrap(), ids(&[9, 10, 11]));
        assert_eq!(d.select(&[Slice::from(-2..)]).unwrap(), ids(&[10, 11]));
        assert_eq!(
            d.select(&[Slice::from_range_stepped(0..12, 5)]).unwrap(),
            ids(&[0, 5, 10])
        );
        assert_eq!(d.select(&[Slice::from(..)]).unwrap().len(), 12);
        assert_eq!(
            d.select(&[Slice::from(12..)]).unwrap(),
            ids(&[]),
            "empty at the end"
        );

        assert!(matches!(
            d.select(&[Slice::from(..13)]),
            Err(BunsenError::InvalidArgument { .. })
        ));
        assert!(matches!(
            d.select(&[Slice::from(-13..)]),
            Err(BunsenError::InvalidArgument { .. })
        ));
        assert!(matches!(
            d.select(&[Slice::with_step(5, Some(0), -1)]),
            Err(BunsenError::InvalidArgument { .. })
        ));
    }
}
