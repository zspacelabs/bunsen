# train-chat example

This example trains a NanoChat-style GPT language model from scratch on the
karpathy/nanochat (`fineweb-edu-100b-shuffle`) corpus. It loads a pretrained
tokenizer vocabulary, lazily downloads and caches dataset shards, streams packed
token blocks through a chat data loader, and trains with the Burn `Learner`. Its
defining feature is a per-parameter-group optimizer setup that mirrors nanochat:
the model's 2D matrix weights are optimized with Muon while embeddings, the LM
head, and remaining scalar parameters use separately-tuned AdamW groups, with a
linear learning-rate warmup.

## Bunsen features exercised

- `bunsen::kits::gpts::nanochat` — the `NanoChatGpt` model with
  `NanoChatGptConfig` / `NanoChatGptMeta`.
- `bunsen::burner::module::reflection::XmlModuleTree` — reflect over the module
  tree and select parameters by XPath-like queries (`select_params`,
  `to_param_ids`) to build optimizer groups.
- `bunsen::burner::optim` — `GroupOptimizerAdaptor2` and `OptimizerGroup` for
  composing multiple optimizers (Muon + AdamW) with per-group learning-rate
  selectors over disjoint parameter sets.
- `bunsen::public::hashbrown` — re-exported `HashSet` / `HashMap`.
- `bunsen-preview-chat-dataloader` — the `ChatDataLoader` plus dense
  token-block batching options (`DenseTokenBlocksOptions`,
  `TokenBatchIteratorOptions`).
- `zsl-data-cache` — the nanochat shard download/disk cache (see the
  [`zsl-data-cache`](../zsl-data-cache) example).

It demonstrates bunsen's reflection-driven optimizer-group machinery, the most
advanced training-configuration feature in the example set.

## Running the Example

First fetch some dataset shards (see the `zsl-data-cache` example), then train:

```bash
$ cargo run --release -p train-chat -- \
  --dataset-dir ~/Data/nanochat/dataset/ --shards 0..10 --weight-decay 0.02
```
