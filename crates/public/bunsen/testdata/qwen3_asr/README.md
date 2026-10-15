# Qwen3-ASR fixtures

A tiny seeded `Qwen3ASRForConditionalGeneration` and its answers, used by
`kits::speech::qwen3_asr`'s toy tests. Everything here is synthetic: random
weights and random inputs, written by `transformers`' own implementation. No
model, recording or text from elsewhere is included, so the files carry no
licence beyond the repository's.

| file | |
|---|---|
| `toy.safetensors` | the toy's weights (`model.*`, `transformers` names) and inputs and outputs (`io.*`), float32 |
| `toy.json` | the toy's config, and its input lengths |

Regenerate with `tools/gen_qwen3_asr_toy_fixtures.py`; its docstring pins the
versions used.
