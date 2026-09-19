# train-chat example

This example trains a NanoChat-style GPT language model from scratch on the karpathy/nanochat
(`fineweb-edu-100b-shuffle`) corpus. It loads a pretrained tokenizer vocabulary, lazily downloads and caches dataset
shards, streams packed token blocks through a chat data loader, and trains with the Burn `Learner`. Its defining feature
is a per-parameter-group optimizer setup that mirrors nanochat:
the model's 2D matrix weights are optimized with Muon while embeddings, the LM head, and remaining scalar parameters use
separately-tuned AdamW groups, with a linear learning-rate warmup.

## Bunsen features exercised

- `bunsen::kits::gpts::nanochat` — the `NanoChatGpt` model with
  `NanoChatGptConfig` / `NanoChatGptMeta`.
- `bunsen::burner::module::reflection::XmlModuleTree` — reflect over the module tree and select parameters by XPath-like
  queries (`select_params`,
  `to_param_ids`) to build optimizer groups.
- `bunsen::burner::optim` — `GroupOptimizerAdaptor2` and `OptimizerGroup` for composing multiple optimizers (Muon +
  AdamW) with per-group learning-rate selectors over disjoint parameter sets.
- `bunsen::public::hashbrown` — re-exported `HashSet` / `HashMap`.
- `bunsen-arrow-dataloaders` — the `ChatDataLoader` plus dense token-block batching options (`DenseTokenBlocksOptions`,
  `TokenBatchIteratorOptions`).
- `bunsen::data::shards` with `kits::gpts::nanochat::datasets::NANOCHAT_SHARD_SETS` — the fineweb-edu shard set,
  fetched into the cache's data directory or a `--dataset-dir` of existing shards, under a `--parallel`/`--retries`/
  `--keep-going` policy with a summary report.

It demonstrates bunsen's reflection-driven optimizer-group machinery, the most advanced training-configuration feature
in the example set.

## Running the Example

Train; the shards named by `--shards` are fetched first if they are not there (each is ~90 MB):

```bash
$ cargo run --release -p train-chat -- \
  --dataset-dir ~/Data/nanochat/dataset/ --shards 0..10 --weight-decay 0.02
```
