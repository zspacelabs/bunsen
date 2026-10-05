# Shard sets and training data

A language model trains on a corpus too large to fetch at once, published as
many numbered files. This chapter covers how bunsen names, pins and fetches
those files, and where they go next. The reference is
[`bunsen::data::shards`], whose
[lifecycle](bunsen::data::shards#lifecycle) links every step.

## The pathway

```text
a kit's shard set table      names its sets, compiled in
  │  look up, select a range of shards
  ▼
ShardSet                     the set, bound to a directory
  │  locate / fetch / fetch_many
  ▼
shard paths
  │
  ▼
a data loader                bunsen-arrow-dataloaders, in train-chat
```

A [`ShardSetDescriptor`](bunsen::data::shards::ShardSetDescriptor) says
what a set is: its name, the base URLs its files come from, how they are
named and numbered, and whether they are pinned. It is plain data, and it
follows the same split as [pretrained models](./pretrained.md): a static
twin for tables compiled into a kit, such as
[NanoChat's corpus](bunsen::kits::gpts::nanochat::datasets), and an owned
twin for a set built at run time. A
[`ShardSet`](bunsen::data::shards::ShardSet) binds a descriptor to a
directory, the disk cache's or one the caller gives, and brings shards in.
Binding needs the `cache` feature; downloading needs `fetch`. Without
`fetch`, a set works from shards that are already on disk.

## Pinned shards

A pinned set carries one SHA-256 per shard, and a fetch verifies each file
as its bytes land. The tables are generated, not typed:
[`tools/gen_shard_digests.py`](https://github.com/zspacelabs/bunsen/blob/main/tools/gen_shard_digests.py)
reads the dataset's file listing at a pinned revision of its Hugging Face
repository, and the descriptor's base URL resolves that same revision. The
files and their digests therefore can't drift apart. Like the other
fixture generators, the script is run once by hand and its output committed
([Validation](../development/validation.md#fixtures-generated-once)).

## bunsen stops at paths

bunsen's part ends with a list of local paths. Reading them is a data
loader's job, and the loader that reads NanoChat's Parquet shards is a
separate crate,
[`bunsen_arrow_dataloaders`], a preview that does
not depend on bunsen. Its
[`ChatDataLoader`](bunsen_arrow_dataloaders::dataloaders::chat::ChatDataLoader)
streams the shards through text selection, tokenization and packing into
fixed-shape blocks of token ids, with an optional shuffle
([pipeline overview](bunsen_arrow_dataloaders::dataloaders::chat#pipeline-overview)).

A selection of shards can also become a
[`ResourceMap`](bunsen::data::pretrained::ResourceMap), through
[`to_resource_map`](bunsen::data::shards::ShardSetDescriptor::to_resource_map),
for a pretrained row that includes a shard set.

## End to end: train-chat

[`examples/train-chat`](https://github.com/zspacelabs/bunsen/tree/main/examples/train-chat)
joins the two halves. Its command line selects shards of NanoChat's corpus
and a fetch policy, it fetches the selection through a `ShardSet`, splits the
paths into training and validation, and hands each part to a
`ChatDataLoader`. The shard-selection arguments come from the `clap-common`
dev crate, and the loader crate's own `dl-test` example uses the same ones.
