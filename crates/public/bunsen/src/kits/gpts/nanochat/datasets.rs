//! # `NanoChat` datasets
//!
//! The shard sets `NanoChat` trains on, as
//! [`data::shards`](crate::data::shards) tables. The machinery is there; this
//! is the data.

use crate::data::shards::{
    ShardDigests,
    StaticShardSetDescriptor,
    StaticShardSetMap,
};

/// karpathy's shuffled 100B-token sample of fineweb-edu, as nanochat trains
/// on it: 1822 parquet shards of about 90 MB each.
pub static FINEWEB_EDU_100B_SHUFFLE: StaticShardSetDescriptor<'static> = StaticShardSetDescriptor {
    name: "fineweb-edu-100b-shuffle",
    description: "karpathy's shuffled 100B-token sample of fineweb-edu: 1822 parquet shards of ~90 MB, \
                  the nanochat training corpus",
    license: Some("ODC-By 1.0"),
    origin: Some("https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle"),
    base_urls: &["https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"],
    template: "shard_{index}.parquet",
    index_width: 5,
    count: 1822,
    format: "parquet",
    digests: ShardDigests::Unpinned,
};

/// The shard sets `NanoChat` trains on.
pub static NANOCHAT_SHARD_SETS: StaticShardSetMap = StaticShardSetMap {
    name: "nanochat",
    description: "The datasets NanoChat trains on.",
    items: &[&FINEWEB_EDU_100B_SHUFFLE],
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::shards::ShardId;

    /// The table spells the set as upstream names it.
    #[test]
    fn test_fineweb_edu_table() {
        let d = NANOCHAT_SHARD_SETS.expect_lookup("fineweb-edu-100b-shuffle");
        d.validate().unwrap();
        assert_eq!(d.count, 1822);
        assert_eq!(d.file_name(ShardId(0)), "shard_00000.parquet");
        assert_eq!(d.file_name(ShardId(1821)), "shard_01821.parquet");
        assert_eq!(
            d.urls(ShardId(312)),
            vec![
                "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main/shard_00312.parquet"
            ]
        );
        assert_eq!(d.parse_file_name("shard_00312.parquet"), Some(ShardId(312)));
        assert_eq!(d.parse_file_name("shard_01822.parquet"), None);
        assert_eq!(d.digest(ShardId(0)), None);
        assert_eq!(
            NANOCHAT_SHARD_SETS.names(),
            vec!["fineweb-edu-100b-shuffle"]
        );
    }
}
