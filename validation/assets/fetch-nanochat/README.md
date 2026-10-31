# fetch-nanochat

nanochat validation assets for [`bunsen`](https://crates.io/crates/bunsen), fetched at build time. Not published:
only `nanochat-model-validation` depends on it.

**This crate hosts nothing.** Its build script downloads each asset into the build directory and checks it against a
pinned SHA-256, laid out as a bunsen pretrained cache. Each row is behind a feature, and none is on by default, so a
plain build fetches nothing:

- `base_d20`, `mid_d20`, `rl_d20`: `nanochat-students`' d20 checkpoints (base, midtrained, RL), each a
  `model_*.pt`, its `meta_*.json`, and the tokenizer (`tokenizer.pkl`, `token_bytes.pt`).
- `d18_base`, `d18_sft`: `nitishpandey04`'s d18 checkpoints (base, SFT), on nanochat's 2026 architecture, with
  their tokenizer.

These are upstream nanochat's own checkpoint files, unconverted: `nanochat-model-validation` holds bunsen to
upstream's `gpt.py` on the same files. A build without the network can point the per-file environment variables at
local copies; the API docs list them.

## License

Distributed under the terms of both the MIT license and the Apache License (Version 2.0). The assets it fetches are
under their publishers' licenses: `nanochat-students`' checkpoints are Apache-2.0, `nitishpandey04`'s MIT.
