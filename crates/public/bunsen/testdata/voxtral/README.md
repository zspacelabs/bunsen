# Voxtral Realtime fixtures

A tiny seeded `VoxtralRealtimeForConditionalGeneration` and its answers, used by
`kits::speech::voxtral`'s toy tests: synthetic, random weights and random
inputs, written by `transformers`' own implementation, so carrying no licence
beyond the repository's. And one `config.json`, copied from Mistral's repo
(Apache-2.0), which the `config.json` reader is tested on.

| file | |
|---|---|
| `toy.safetensors` | the toy's weights (under the published `model.safetensors`'s names) and inputs and outputs (`io.*`), float32 |
| `toy.json` | the toy's config, and the delays its logits were recorded at |
| `voxtral-mini-4b-realtime-2602.config.json` | `mistralai/Voxtral-Mini-4B-Realtime-2602`'s `config.json` at `2769294da9567371363522aac9bbcfdd19447add` (Apache-2.0) |

Regenerate with `tools/gen_voxtral_toy_fixtures.py`; its docstring pins the
versions used.
