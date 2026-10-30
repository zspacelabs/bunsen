# The workspace

bunsen is one Cargo workspace with four kinds of member. Only the crates
under `crates/public` are published; everything else exists to build,
check, validate or demonstrate them.

```text
crates/public/      the published crates (released together, one version)
crates/dev/         development tooling: upstream bug repros, benches, test audio
crates/validation/  cross-checks of bunsen models against reference implementations
examples/           runnable programs, one per kit or system
tools/              one-shot fixture generators and repo checks (Python, stdlib-first)
```

## Published crates

All published crates share the workspace version and are released together
by release-plz ([Releasing](../development/release.md)), except
`bunsen-bundled-parakeet`, `bunsen-bundled-qwen3`, `bunsen-bundled-qwen3-asr`
and `bunsen-bundled-sortformer`, which fetch the checkpoints the validation crates
check against and are not published.

| Crate | What it is |
|---|---|
| [`bunsen`] | The library: contracts, ops, blocks, kits, the pretrained-model machinery, audits. Most users depend on this alone. See [the bunsen crate](./crate.md). |
| [`bunsen-contracts-macros`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-contracts-macros) | The proc macro behind [`shape_contract!`](bunsen::contracts#macros-and-methods). Never a direct dependency; `bunsen` re-exports it. |
| [`bunsen-bundled-silero`](bunsen_bundled_silero) | Silero VAD's 16 kHz checkpoint and, for validation, its ONNX graph and the reference model generated from it, fetched and digest-pinned at build time, for `silero-model-validation`. Hosts no files itself. |
| [`bunsen-bundled-whisper`](bunsen_bundled_whisper) | Whisper's `base` checkpoint and tokenizer vocabularies, fetched and digest-pinned at build time and laid out as a pretrained cache, for `whisper-model-validation`. Hosts no files itself. |
| [`bunsen-firehose`](bunsen_firehose) | A column-oriented data-processing pipeline that feeds burn's `Batcher`. Independent of `bunsen`. See [Firehose](../systems/firehose.md). |
| [`bunsen-firehose-image`](bunsen_firehose_image) | Image operators for firehose: loading, augmentation, conversion to tensor data. |
| [`bunsen-arrow-dataloaders`](bunsen_arrow_dataloaders) | *Preview.* Parquet shards to tokenized training batches, for language-model training. See [Shard sets](../systems/shards.md). |
| [`bunsen-app`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-app) | Shared command-line plumbing for bunsen's binaries: device and precision flags, logging, shard selection, and the weights cache's flags and `cache` commands. The examples and `bunsen-asr` are built on it. |
| [`bunsen-asr`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-asr) | A binary, `cargo install bunsen-asr`: speech to text by profile (`en`, `multi`, `translate-en`) or with Parakeet, Qwen3-ASR or Whisper directly, each through its [stream driver](../systems/streaming.md); speaker diarization with Sortformer; and `bunsen-asr cache` over the weights cache. |

The two `bundled` crates are how bunsen ships model assets without making
every build download them; [Assets, bundling and the network](../development/assets.md)
explains the scheme.

## Development crates

| Crate | Role |
|---|---|
| [`burn_bug_repro`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/burn_bug_repro) | Reproductions of defects in burn and its ecosystem, each paired with a pin on today's behaviour. See [Upstream bug reproductions](../development/repros.md). |
| [`silero-bench`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/silero-bench) | A benchmark CLI for Silero VAD. See [Benchmarks](../development/benchmarks.md). |
| [`bunsen-test-audio`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/bunsen-test-audio) | The audio clips the tests, the validation suites and the fixture scripts read, with their provenance and ground truth. A leaf crate, so `bunsen` takes it as a dev-dependency. |

## Validation crates

| Crate | Validates |
|---|---|
| [`sortformer-model-validation`](https://github.com/zspacelabs/bunsen/tree/main/crates/validation/sortformer-model-validation) | Sortformer diarization against ONNX exports of the same checkpoints (`Nemotron-3-Diarization`, Sortformer v2.1), each held itself to onnxruntime's run of the published graph, and against `NeMo`'s own diarization of an AMI excerpt by each model. |
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
