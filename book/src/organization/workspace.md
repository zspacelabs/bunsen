# The workspace

bunsen is two Cargo workspaces. The main one, at the repository root, has
three kinds of member; only the crates under `crates/public` are published,
and everything else exists to build, check or demonstrate them. The
validation workspace, under `validation/`, holds bunsen's models to
reference implementations; nothing the main workspace runs builds it.

```text
crates/public/      the published crates (released together, one version)
crates/dev/         development tooling: upstream bug repros, benches, test audio
examples/           runnable programs, one per kit or system
tools/              one-shot fixture generators and repo checks (Python, stdlib-first)

validation/         a workspace of its own (validation/Cargo.toml, validation/Cargo.lock)
  onnx/             the ONNX references and checkpoints, fetched at build time
  models.yml        the catalog of the models the validation crates check, which a profile chooses from
  models/           cross-checks of bunsen models against those references
  tools/            the fixture generators whose output lands in models/
```

## Published crates

All published crates share the workspace version and are released together
by release-plz ([Releasing](../development/release.md)).

| Crate | What it is |
|---|---|
| [`bunsen`] | The library: contracts, ops, blocks, kits, the pretrained-model machinery, audits. Most users depend on this alone. See [the bunsen crate](./crate.md). |
| [`bunsen-contracts-macros`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-contracts-macros) | The proc macro behind [`shape_contract!`](bunsen::contracts#macros-and-methods). Never a direct dependency; `bunsen` re-exports it. |
| [`bunsen-firehose`](bunsen_firehose) | A column-oriented data-processing pipeline that feeds burn's `Batcher`. Independent of `bunsen`. See [Firehose](../systems/firehose.md). |
| [`bunsen-firehose-image`](bunsen_firehose_image) | Image operators for firehose: loading, augmentation, conversion to tensor data. |
| [`bunsen-arrow-dataloaders`](bunsen_arrow_dataloaders) | *Preview.* Parquet shards to tokenized training batches, for language-model training. See [Shard sets](../systems/shards.md). |
| [`bunsen-app`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-app) | Shared command-line plumbing for bunsen's binaries: device and precision flags, logging, shard selection, and the weights cache's flags and `cache` commands. The examples and `bunsen-tool` are built on it. |
| [`bunsen-tool`](https://github.com/zspacelabs/bunsen/tree/main/crates/public/bunsen-tool) | A binary, `cargo install bunsen-tool`: `bunsen-tool asr`, speech to text by profile (`en`, `multi`, `translate-en`) or with Parakeet, Qwen3-ASR, Voxtral Realtime or Whisper directly, each through its [stream driver](../systems/streaming.md); speaker diarization with Sortformer; `bunsen-tool chat`, chat with Qwen3 or nanochat, whose driver is also a library (`bunsen_tool::chat`); and `bunsen-tool cache` over the weights cache. |

## Development crates

| Crate | Role |
|---|---|
| [`burn_bug_repro`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/burn_bug_repro) | Reproductions of defects in burn and its ecosystem, each paired with a pin on today's behaviour. See [Upstream bug reproductions](../development/repros.md). |
| [`silero-bench`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/silero-bench) | A benchmark CLI for Silero VAD. See [Benchmarks](../development/benchmarks.md). |
| [`bunsen-test-audio`](https://github.com/zspacelabs/bunsen/tree/main/crates/dev/bunsen-test-audio) | The audio clips the tests, the validation suites and the fixture scripts read, with their provenance and ground truth. A leaf crate, so `bunsen` takes it as a dev-dependency. |

## The validation workspace

`validation/` depends on the main workspace's crates by path, and has a
lockfile of its own; every package bunsen's crates reach in it is the
version `Cargo.lock` has (`tools/check_validation_lock.py`, run in CI).
Building it with its default features fetches nothing; `weights` and
`download` fetch, and only the validation runs turn them on
([Validation](../development/validation.md)). No CI job builds it;
`cargo make check-validation` does, by hand.

### ONNX binding crates

| Crate | Role |
|---|---|
| [`rewrite-genai-onnx`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/rewrite-genai-onnx) | Rewrites an onnxruntime-genai ONNX export's `com.microsoft` ops to standard ONNX, so `burn-onnx` imports it; the `bind-{model}-onnx` build scripts call it. |
| `bind-{model}-onnx` ([`whisper`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/bind-whisper-onnx), [`silero`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/bind-silero-onnx), [`parakeet`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/bind-parakeet-onnx), [`qwen3`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/bind-qwen3-onnx), [`qwen3-asr`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/bind-qwen3-asr-onnx), [`sortformer`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/bind-sortformer-onnx), [`voxtral`](https://github.com/zspacelabs/bunsen/tree/main/validation/onnx/bind-voxtral-onnx)) | Each validation crate's assets: the checkpoints it compares, fetched and digest-pinned at build time and laid out as a pretrained cache, and the reference model `burn-onnx` generates from an upstream ONNX export. Host no files. See [Assets, bundling and the network](../development/assets.md#the-bundled-crates). |

### Validation crates

| Crate | Validates |
|---|---|
| [`nanochat-model-validation`](https://github.com/zspacelabs/bunsen/tree/main/validation/models/nanochat-model-validation) | nanochat against upstream's own code at the pinned commit of each model, on the community d20 and d18 checkpoints: prefill logits and greedy decodes, in float32 and, on CUDA, in bf16. |
| [`sortformer-model-validation`](https://github.com/zspacelabs/bunsen/tree/main/validation/models/sortformer-model-validation) | Sortformer diarization against ONNX exports of the same checkpoints (`Nemotron-3-Diarization`, Sortformer v2.1), each held itself to onnxruntime's run of the published graph, and against `NeMo`'s own diarization of an AMI excerpt by each model. |
| [`silero-model-validation`](https://github.com/zspacelabs/bunsen/tree/main/validation/models/silero-model-validation) | Silero VAD against the ONNX graph it was transliterated from, and against a committed golden trace. |
| [`whisper-model-validation`](https://github.com/zspacelabs/bunsen/tree/main/validation/models/whisper-model-validation) | Whisper against an ONNX export, stage by stage, and against `openai-whisper`'s own decode of real speech. |

Their reference checks are not part of a normal build's test run; they are
how a kit earns trust. See [Validation against reference implementations](../development/validation.md).

## Examples and tools

The [examples](./examples.md) are runnable programs, one or more per kit or
system. `tools/` holds the Python scripts that generate bunsen's committed
test fixtures from reference implementations, and the repository checks CI
runs; `validation/tools/` holds those that generate the validation crates'
fixtures. Each fixture script is run once by hand, never by CI
([Validation](../development/validation.md)).
