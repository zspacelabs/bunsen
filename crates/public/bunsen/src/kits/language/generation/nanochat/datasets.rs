//! # `NanoChat` datasets
//!
//! The shard sets `NanoChat` trains on, by their builds' names in the
//! built-in catalog: each a build of the `parquet-shards` contract, its
//! shards pinned to one revision of its Hugging Face repository by `cargo
//! catalog`, so the files and their digests cannot drift apart.
//! [`ShardSetDescriptor::from_builtin`](crate::data::datasets::ShardSetDescriptor::from_builtin)
//! gives the set.

/// karpathy's shuffled 100B-token sample of fineweb-edu, as nanochat trains
/// on it: 1823 parquet shards of about 90 MB each.
pub const FINEWEB_EDU_100B_SHUFFLE: &str = "karpathy/fineweb-edu-100b-shuffle";

/// The revision of `karpathy/fineweb-edu-100b-shuffle` the catalog pins.
pub const FINEWEB_EDU_100B_SHUFFLE_REVISION: &str = "4c8f30d6756da75362432a4d5569e1b229263b71";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::datasets::{
        ShardId,
        ShardSetDescriptor,
    };

    /// The set as upstream names it, pinned to one revision, with a digest
    /// for every shard; two spot-checked against the Hub's listing, first and
    /// last.
    #[test]
    fn test_fineweb_edu() {
        let d = ShardSetDescriptor::from_builtin(FINEWEB_EDU_100B_SHUFFLE).unwrap();
        assert_eq!(d.count, 1823);
        assert!(d.is_pinned());
        assert_eq!(d.file_name(ShardId(0)), "shard_00000.parquet");
        assert_eq!(d.file_name(ShardId(1822)), "shard_01822.parquet");
        assert_eq!(
            d.urls(ShardId(312)),
            vec![format!(
                "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/{FINEWEB_EDU_100B_SHUFFLE_REVISION}/shard_00312.parquet"
            )]
        );
        assert_eq!(d.parse_file_name("shard_00312.parquet"), Some(ShardId(312)));
        assert_eq!(d.parse_file_name("shard_01823.parquet"), None);
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
