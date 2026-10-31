# Asset crates

Validation assets that are not ONNX references: each crate fetches, at build time and only under its features, the
checkpoints a model's validation crate compares against, digest-pinned and laid out as a bunsen pretrained cache.
None is published, and none fetches anything unless a feature asks.

| Crate | What it is |
|---|---|
| `fetch-nanochat` | The community nanochat checkpoints (`nanochat-students`' d20 base, mid and RL; `nitishpandey04`'s d18 base and SFT), upstream's own `.pt` files and tokenizers, for `nanochat-model-validation`, which holds bunsen to upstream's `gpt.py` on them. |
