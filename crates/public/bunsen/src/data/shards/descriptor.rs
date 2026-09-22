//! # Shard-set descriptors
//!
//! What a shard set is: its name, where its shards come from, how they are
//! named and numbered, and whether they are pinned. A selection of shards
//! is also a [`ResourceMap`], for a pretrained that fuses a shard set in.

use std::{
    collections::BTreeSet,
    fmt,
};

use burn::tensor::Slice;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    data::pretrained::{
        Resource,
        ResourceMap,
        Source,
    },
    errors::{
        BunsenError,
        BunsenResult,
    },
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

/// How a set's shards are pinned, as a compiled-in table spells it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StaticShardDigests<'a> {
    /// No digests: a shard is trusted on its length.
    #[default]
    Unpinned,

    /// One lowercase hex SHA-256 per shard, index-ordered.
    Table(&'a [&'a str]),
}

impl StaticShardDigests<'_> {
    /// The owned twin.
    pub fn to_digests(&self) -> ShardDigests {
        match self {
            Self::Unpinned => ShardDigests::Unpinned,
            Self::Table(table) => {
                ShardDigests::Table(table.iter().map(|s| s.to_string()).collect())
            }
        }
    }
}

/// How a set's shards are pinned.
///
/// [`Unpinned`](Self::Unpinned): a fetched shard is checked against its
/// `Content-Length` alone. [`Table`](Self::Table): one SHA-256 per shard,
/// which the fetch verifies as the bytes land. A listing fetched from the
/// source is the planned addition, which is why this is `#[non_exhaustive]`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ShardDigests {
    /// No digests: a shard is trusted on its length.
    #[default]
    Unpinned,

    /// One lowercase hex SHA-256 per shard, index-ordered.
    Table(Vec<String>),
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
    pub digests: StaticShardDigests<'a>,
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
            digests: self.digests.to_digests(),
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
        if let ShardDigests::Table(table) = &self.digests {
            if table.len() != self.count {
                return Err(BunsenError::Invalid(format!(
                    "{}: {} digests for {} shards",
                    self.name,
                    table.len(),
                    self.count
                )));
            }
            let is_hex = |d: &str| {
                d.len() == 64 && d.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
            };
            if let Some((i, bad)) = table.iter().enumerate().find(|(_, d)| !is_hex(d)) {
                return Err(BunsenError::Invalid(format!(
                    "{}: shard {i}'s digest {bad:?} is not 64 lowercase hex digits",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// `true` when fetched shards are checked against a digest.
    pub fn is_pinned(&self) -> bool {
        !matches!(self.digests, ShardDigests::Unpinned)
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
        id: ShardId,
    ) -> Option<&str> {
        match &self.digests {
            ShardDigests::Unpinned => None,
            ShardDigests::Table(table) => table.get(id.0).map(String::as_str),
        }
    }

    /// The shards `ids` name as a resource map: one resource per shard,
    /// keyed by its file name, pinned by the table when the set is, labeled
    /// by the set's format, under the set's name as its namespace, and
    /// reachable from every base URL in order. What a pretrained row fuses a
    /// shard set in as, and what the pretrained cache's `load` brings in
    /// together.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] for an id outside the set.
    pub fn to_resource_map(
        &self,
        ids: &[ShardId],
    ) -> BunsenResult<ResourceMap> {
        let mut map = ResourceMap::new(&self.name);
        map.description = self.description.clone();
        map.license = self.license.clone();
        map.origin = self.origin.clone();
        for &id in ids {
            if !self.contains(id) {
                return Err(BunsenError::Invalid(format!(
                    "{}: shard {id} is out of range; the set has {} shards",
                    self.name, self.count
                )));
            }
            let file = self.file_name(id);
            map.insert(Resource {
                key: file.clone(),
                file,
                sha256: self.digest(id).map(str::to_string),
                kind: Some(self.format.clone()),
                namespace: self.name.clone(),
                sources: self.urls(id).into_iter().map(Source::Url).collect(),
            });
        }
        Ok(map)
    }

    /// The ids `slices` name, sorted and without repeats.
    ///
    /// Each slice is resolved against [`count`](Self::count): a negative
    /// bound counts from the end, an open end is the end. Steps must be
    /// positive.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] for a bound outside `0..=count` or a
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
                return Err(BunsenError::Invalid(format!(
                    "shard index {bound} is out of range for {}, which has {count} shards",
                    self.name
                )));
            }
            Ok(resolved as usize)
        };

        let mut ids = BTreeSet::new();
        for slice in slices {
            if slice.is_reversed() {
                return Err(BunsenError::Invalid(format!(
                    "shard slice {slice} is reversed; shards are selected in order"
                )));
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
            digests: StaticShardDigests::Unpinned,
        }
    }

    /// Twelve digests for `tiny`, all the same and all well formed.
    static TINY_SHA256: [&str; 12] =
        ["ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"; 12];

    fn tiny_pinned() -> StaticShardSetDescriptor<'static> {
        StaticShardSetDescriptor {
            digests: StaticShardDigests::Table(&TINY_SHA256),
            ..tiny()
        }
    }

    /// A table pins every shard: it converts, validates, serializes, and
    /// answers `digest` by index; a short or malformed table is refused.
    #[test]
    fn test_table_digests() {
        let d = tiny_pinned().to_descriptor();
        assert!(d.is_pinned());
        assert!(!tiny().to_descriptor().is_pinned());
        assert_eq!(
            d.digests,
            ShardDigests::Table(vec![TINY_SHA256[0].to_string(); 12])
        );
        assert_eq!(d.digest(ShardId(0)), Some(TINY_SHA256[0]));
        assert_eq!(d.digest(ShardId(11)), Some(TINY_SHA256[11]));
        assert_eq!(d.digest(ShardId(12)), None);
        d.validate().unwrap();
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(
            serde_json::from_str::<ShardSetDescriptor>(&json).unwrap(),
            d
        );

        let mut short = d.clone();
        short.digests = ShardDigests::Table(vec![TINY_SHA256[0].to_string(); 11]);
        assert!(matches!(
            short.validate(),
            Err(BunsenError::Invalid(m)) if m.contains("11 digests for 12 shards")
        ));

        let mut bad = d.clone();
        let mut table = vec![TINY_SHA256[0].to_string(); 12];
        table[3] = TINY_SHA256[0].to_uppercase();
        bad.digests = ShardDigests::Table(table);
        assert!(matches!(
            bad.validate(),
            Err(BunsenError::Invalid(m)) if m.contains("shard 3's digest")
        ));
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
            Err(BunsenError::Invalid(_))
        ));
        assert!(matches!(
            d.select(&[Slice::from(-13..)]),
            Err(BunsenError::Invalid(_))
        ));
        assert!(matches!(
            d.select(&[Slice::with_step(5, Some(0), -1)]),
            Err(BunsenError::Invalid(_))
        ));
    }

    /// The shards a slice names, as a map: keyed by file name, pinned when
    /// the set is, labeled by the format, one URL per base in order.
    #[test]
    fn test_to_resource_map() {
        let d = tiny_pinned().to_descriptor();
        let ids = d.select(&[Slice::from(..2)]).unwrap();
        let map = d.to_resource_map(&ids).unwrap();
        assert_eq!(map.name, "tiny");
        assert_eq!(map.description, "a tiny set");
        assert_eq!(map.license.as_deref(), Some("CC0"));
        assert_eq!(map.keys(), ["shard_000.bin", "shard_001.bin"]);
        let r = map.get("shard_001.bin").unwrap();
        assert_eq!(r.file, "shard_001.bin");
        assert_eq!(r.sha256.as_deref(), Some(TINY_SHA256[1]));
        assert_eq!(r.kind.as_deref(), Some("bin"));
        assert_eq!(r.namespace, "tiny");
        assert_eq!(
            r.urls(),
            [
                "https://a.example/tiny/shard_001.bin",
                "https://b.example/mirror/shard_001.bin",
            ]
        );
        map.validate().unwrap();

        let unpinned = tiny()
            .to_descriptor()
            .to_resource_map(&[ShardId(3)])
            .unwrap();
        assert!(unpinned.get("shard_003.bin").unwrap().sha256.is_none());
        assert_eq!(unpinned.len(), 1);

        assert!(matches!(
            d.to_resource_map(&[ShardId(12)]),
            Err(BunsenError::Invalid(_))
        ));
    }

    #[test]
    fn test_select_is_ordered_without_repeats_twice_over() {
        let d = tiny().to_descriptor();
        let ids = |v: &[usize]| v.iter().copied().map(ShardId).collect::<Vec<_>>();
        assert_eq!(
            d.select(&[Slice::index(1), Slice::index(1)]).unwrap(),
            ids(&[1])
        );
    }
}
