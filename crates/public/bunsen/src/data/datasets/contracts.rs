//! The shard sets' contracts and slot.

/// The name the datasets' contracts are gathered under, beside the kits'.
pub const DATASETS_KIT: &str = "datasets";

/// The contract a shard set's builds are of, `parquet-shards`: one slot,
/// [`SHARDS`], a templated resource of parquet files.
pub const PARQUET_SHARDS_CONTRACT: &str = include_str!("contract.yml");

/// The contract of a parquet shard set, by its name in the built-in
/// catalog's `$contracts`.
pub const PARQUET_SHARDS_CATALOG_CONTRACT: &str = "parquet-shards";

/// The contract a webdataset shard set's builds are of, `webdataset-shards`:
/// one slot, [`SHARDS`], a templated resource of tar files whose members are
/// each sample's files (`<key>.jpg`, `<key>.cls`), read by
/// `WebDatasetSamples` (feature `store`).
pub const WEBDATASET_SHARDS_CONTRACT: &str = include_str!("webdataset_contract.yml");

/// The contract of a webdataset shard set, by its name in the built-in
/// catalog's `$contracts`.
pub const WEBDATASET_SHARDS_CATALOG_CONTRACT: &str = "webdataset-shards";

/// The slot a shard set's files fill.
pub const SHARDS: &str = "shards";
