//! `ImageNet`'s data, by the names the built-in catalog gives it.

/// `ImageNet`-1k's class synsets, in class order: a catalog resource.
pub const IMAGENET_1K_SYNSETS: &str = "imagenet/synsets-1k";

/// Every `ImageNet` subset's synsets' lemmas: a catalog resource.
pub const IMAGENET_LEMMAS: &str = "imagenet/lemmas";

/// `ImageNet`-1k's 50,000 validation images, as timm's webdataset mirror
/// shards them: a catalog build of the `webdataset-shards` contract, 64 tars
/// of `<key>.jpg` and `<key>.cls`.
///
/// The mirror is gated: the Hub serves it to an account that has accepted
/// `ImageNet`'s terms, with its token (`bunsen_fetch::hub_token`). bunsen
/// fetches it on use, into the cache's data directory, and never ships it.
pub const IMAGENET_1K_VALIDATION: &str = "timm/imagenet-1k-validation";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::datasets::ShardSetDescriptor;

    /// The validation set resolves in the built-in catalog: 64 shards, each
    /// pinned, nothing fetched.
    #[test]
    fn test_the_validation_set_is_pinned() {
        let desc = ShardSetDescriptor::from_builtin(IMAGENET_1K_VALIDATION).unwrap();
        assert_eq!(desc.ids().count(), 64);
        assert!(desc.is_pinned());
        assert_eq!(
            desc.file_name(crate::data::datasets::ShardId(0)),
            "imagenet1k-validation-00.tar"
        );
    }
}
