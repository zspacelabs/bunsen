//! # `NanoChat` datasets
//!
//! The shard sets `NanoChat` trains on, as
//! [`data::shards`](crate::data::shards) tables. The machinery is there; this
//! is the data.
//!
//! Each set is pinned to one revision of its Hugging Face repository: the
//! base URL resolves that revision, and the digest table was read from its
//! listing by `tools/gen_shard_digests.py`, so the files and their digests
//! cannot drift apart.

mod fineweb_edu_100b_shuffle_digests;

pub use fineweb_edu_100b_shuffle_digests::FINEWEB_EDU_100B_SHUFFLE_SHA256;

use crate::data::shards::{
    StaticShardDigests,
    StaticShardSetDescriptor,
    StaticShardSetMap,
};

/// The revision of `karpathy/fineweb-edu-100b-shuffle` the table pins.
pub const FINEWEB_EDU_100B_SHUFFLE_REVISION: &str = "4c8f30d6756da75362432a4d5569e1b229263b71";

/// karpathy's shuffled 100B-token sample of fineweb-edu, as nanochat trains
/// on it: 1823 parquet shards of about 90 MB each, digest-pinned to
/// [`FINEWEB_EDU_100B_SHUFFLE_REVISION`].
pub static FINEWEB_EDU_100B_SHUFFLE: StaticShardSetDescriptor<'static> = StaticShardSetDescriptor {
    name: "fineweb-edu-100b-shuffle",
    description: "karpathy's shuffled 100B-token sample of fineweb-edu: 1823 parquet shards of ~90 MB, \
                  the nanochat training corpus",
    license: Some("ODC-By 1.0"),
    origin: Some("https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle"),
    base_urls: &[
        "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/4c8f30d6756da75362432a4d5569e1b229263b71",
    ],
    template: "shard_{index}.parquet",
    index_width: 5,
    count: 1823,
    format: "parquet",
    digests: StaticShardDigests::Table(FINEWEB_EDU_100B_SHUFFLE_SHA256),
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

    /// The table spells the set as upstream names it, pinned to one revision,
    /// with a digest for every shard.
    #[test]
    fn test_fineweb_edu_table() {
        let d = NANOCHAT_SHARD_SETS.expect_lookup("fineweb-edu-100b-shuffle");
        d.validate().unwrap();
        assert_eq!(d.count, 1823);
        assert_eq!(FINEWEB_EDU_100B_SHUFFLE_SHA256.len(), 1823);
        assert!(d.is_pinned());
        assert_eq!(d.file_name(ShardId(0)), "shard_00000.parquet");
        assert_eq!(d.file_name(ShardId(1822)), "shard_01822.parquet");
        assert_eq!(
            d.urls(ShardId(312)),
            vec![format!(
                "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/{FINEWEB_EDU_100B_SHUFFLE_REVISION}/shard_00312.parquet"
            )]
        );
        assert!(d.base_urls[0].ends_with(FINEWEB_EDU_100B_SHUFFLE_REVISION));
        assert_eq!(d.parse_file_name("shard_00312.parquet"), Some(ShardId(312)));
        assert_eq!(d.parse_file_name("shard_01823.parquet"), None);
        assert_eq!(
            NANOCHAT_SHARD_SETS.names(),
            vec!["fineweb-edu-100b-shuffle"]
        );
    }

    /// Two digests spot-checked against the hub's listing, first and last.
    #[test]
    fn test_fineweb_edu_digests_match_the_listing() {
        let d = FINEWEB_EDU_100B_SHUFFLE.to_descriptor();
        assert_eq!(
            d.digest(ShardId(0)),
            Some("6a3450ea745ee6b6830c09819c0f2c72682eba443022de595bcb50ceb3092c22")
        );
        assert_eq!(
            d.digest(ShardId(1822)),
            Some("a97e7f2f4680561c1fdf0461295b0cfc38d85f497ba03851ab267ac1b8165990")
        );
        assert_eq!(d.digest(ShardId(1823)), None);
    }
}
