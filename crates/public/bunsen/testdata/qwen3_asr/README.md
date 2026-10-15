# Qwen3-ASR fixtures

A tiny seeded `Qwen3ASRForConditionalGeneration` and its answers, used by
`kits::speech::qwen3_asr`'s toy tests: synthetic, random weights and random
inputs, written by `transformers`' own implementation, so carrying no licence
beyond the repository's. And two `config.json` files, copied from Qwen's
repos (Apache-2.0), which the `hf:` config reader is tested on.

| file | |
|---|---|
| `toy.safetensors` | the toy's weights (`model.*`, `transformers` names) and inputs and outputs (`io.*`), float32 |
| `toy.json` | the toy's config, and its input lengths |
| `qwen3-asr-0.6b-hf.config.json` | `Qwen/Qwen3-ASR-0.6B-hf`'s `config.json` at `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` (Apache-2.0) |
| `qwen3-asr-1.7b-hf.config.json` | `Qwen/Qwen3-ASR-1.7B-hf`'s `config.json` at `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` (Apache-2.0) |

Regenerate with `tools/gen_qwen3_asr_toy_fixtures.py`; its docstring pins the
versions used.
