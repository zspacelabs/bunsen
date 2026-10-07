# The workspace

bunsen is one Cargo workspace with four kinds of member. Only the crates
under `crates/public` are published; everything else exists to build,
check, validate or demonstrate them.

```text
crates/public/      the published crates (released together, one version)
crates/dev/         development tooling: upstream bug repros, shared CLI code, benches
crates/validation/  cross-checks of bunsen models against reference implementations
examples/           runnable programs, one per kit or system
tools/              one-shot fixture generators and repo checks (Python, stdlib-first)
```

## Published crates

All published crates share the workspace version and are released together
by release-plz ([Releasing](../development/release.md)).

| Crate | What it is |
|---|---|
| [`bunsen`] | The library: contracts, ops, blocks, kits, the pretrained-model machinery, audits. Most users depend on this alone. See [the bunsen crate](./crate.md). |
| [`bunsen-contracts-macros`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-contracts-macros) | The proc macro behind [`shape_contract!`](bunsen::contracts#macros-and-methods). Never a direct dependency; `bunsen` re-exports it. |
| [`bunsen-bundled-silero`](bunsen_bundled_silero) | Silero VAD's pretrained weights, generated at build time from a committed ONNX graph. Pulled in by `bunsen`'s `silero-weights` feature. |
| [`bunsen-bundled-whisper`](bunsen_bundled_whisper) | Whisper's `base` checkpoint and tokenizer vocabularies, fetched and digest-pinned at build time and laid out as a pretrained cache. Pulled in by `whisper-weights`. Hosts no files itself. |
| [`bunsen-firehose`](bunsen_firehose) | A column-oriented data-processing pipeline that feeds burn's `Batcher`. Independent of `bunsen`. See [Firehose](../systems/firehose.md). |
| [`bunsen-firehose-image`](bunsen_firehose_image) | Image operators for firehose: loading, augmentation, conversion to tensor data. |
| [`bunsen-arrow-dataloaders`](bunsen_arrow_dataloaders) | *Preview.* Parquet shards to tokenized training batches, for language-model training. See [Shard sets](../systems/shards.md). |

The two `bundled` crates are how bunsen ships model assets without making
every build download them; [Assets, bundling and the network](../development/assets.md)
explains the scheme.

## Development crates

| Crate | Role |
|---|---|
| [`burn_bug_repro`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/burn_bug_repro) | Reproductions of defects in burn and its ecosystem, each paired with a pin on today's behaviour. See [Upstream bug reproductions](../development/repros.md). |
| [`bunsen-app`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/bunsen-app) | Shared command-line plumbing (logging, device selection, shard selection, fetch policy) for the examples. |
| [`silero-bench`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/silero-bench) | A benchmark CLI for Silero VAD. See [Benchmarks](../development/benchmarks.md). |

## Validation crates

| Crate | Validates |
|---|---|
| [`silero-model-validation`](https://github.com/zspacelabs/bunsen/tree/main/crates/validation/silero-model-validation) | Silero VAD against the ONNX graph it was transliterated from, and against a committed golden trace. |
| [`whisper-model-validation`](https://github.com/zspacelabs/bunsen/tree/main/crates/validation/whisper-model-validation) | Whisper against an ONNX export, stage by stage, and against `openai-whisper`'s own decode of real speech. |

These are not part of a normal build's test run; they are how a kit earns
trust. See [Validation against reference implementations](../development/validation.md).

## Examples and tools

The [examples](./examples.md) are runnable programs, one or more per kit or
system. `tools/` holds the Python scripts that generate committed test
fixtures from reference implementations, run once by hand, never by CI
([Validation](../development/validation.md)), and the repository checks CI
runs.
