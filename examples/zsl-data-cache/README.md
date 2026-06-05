# zsl-data-cache example

A reusable on-disk dataset cache for the karpathy/nanochat
(`fineweb-edu-100b-shuffle`) training corpus. It models the upstream dataset as
a family of numbered parquet shards, downloads requested shards on demand,
caches them locally, and exposes a parquet `RecordBatch` reader for training
loops. The `train-chat` example consumes this crate to feed its data loader.

The bundled `pull_shards` example binary is a small CLI that pre-fetches a range
of shards into a cache directory.

## Bunsen features exercised

This crate is a support library rather than a bunsen API showcase: it provides
the shard-cache plumbing (`DatasetCacheConfig`, `DatasetSource`) that backs
bunsen-based training examples. It builds on Burn's `Config` derive for its
configuration types and on `parquet`/`arrow` for shard reading, and demonstrates
the download-and-disk-cache pattern that bunsen aims to fold into a first-class
component (see the "Future Components" section of the top-level README).

## Running the Example

Pull a range of shards (each shard is ~90MB):

```bash
$ cargo run --release -p pull_shards -- \
  --dataset-dir /media/data/nanochat/dataset --shards ..8
```
