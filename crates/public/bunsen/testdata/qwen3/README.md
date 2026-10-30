# Qwen3 fixtures

A tiny seeded `Qwen3ForCausalLM` and its answers, used by `kits::gpts::qwen3`'s
toy tests: synthetic, random weights and random inputs, written by
`transformers`' own implementation, so carrying no licence beyond the
repository's. And the `config.json` of each checkpoint bunsen pins, copied
from Qwen's repos (Apache-2.0), which the `hf:` config reader is tested on.

| file | |
|---|---|
| `toy.safetensors` | the toy's weights (`model.*`, `transformers` names), a stale `lm_head.weight` a tied loader must not read, and inputs and outputs (`io.*`), float32 |
| `toy.config.json` | the toy's `config.json` |
| `qwen3-0.6b.config.json` | `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca` |
| `qwen3-0.6b-base.config.json` | `Qwen/Qwen3-0.6B-Base` at `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| `qwen3-1.7b.config.json` | `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| `qwen3-1.7b-base.config.json` | `Qwen/Qwen3-1.7B-Base` at `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| `qwen3-4b.config.json` | `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c` |
| `qwen3-4b-base.config.json` | `Qwen/Qwen3-4B-Base` at `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| `qwen3-4b-instruct-2507.config.json` | `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554` |
| `qwen3-4b-thinking-2507.config.json` | `Qwen/Qwen3-4B-Thinking-2507` at `768f209d9ea81521153ed38c47d515654e938aea` |

Regenerate the toy with `tools/gen_qwen3_toy_fixtures.py`; its docstring pins
the versions used.
