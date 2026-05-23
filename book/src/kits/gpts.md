# `bunsen::kits::gpts`

Full GPT / LLM variants. Where [`bunsen::blocks`](../blocks/overview.md)
provides reusable transformer sub-modules, `gpts` is for whole
language-model architectures: end-to-end models, tokenizer wiring, and
the training/inference surface around them.

API: <https://docs.rs/bunsen/latest/bunsen/kits/gpts/>

## Current models

### `nanochat`

A compact GPT in the spirit of the "nano" GPT lineage &mdash; small
enough to train on modest hardware, opinionated enough to be a useful
reference implementation.

The model lives in
[`bunsen::kits::gpts::nanochat`](https://docs.rs/bunsen/latest/bunsen/kits/gpts/nanochat/index.html)
and is split into:

- the per-layer **MLP**,
- the transformer **block** (attention + MLP + norms),
- the full **model** wrapper that stacks the blocks and adds embedding
  and head layers.

`gpts` is a work in progress; further GPT/LLM variants will land here as
the port from upstream reference implementations progresses.
