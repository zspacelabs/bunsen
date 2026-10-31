# validation

A Cargo workspace of its own, apart from the bunsen workspace at the
repository root, of expensive per-model validation crates: each holds one of
bunsen's kits to an independent reference, a Rust model `burn-onnx` generates
from upstream's ONNX export, and to upstream's own checkpoints. Every asset
comes through a `bind-*-onnx` crate, fetched at build time and digest-pinned.

```text
onnx/bind-{model}-onnx/            each model's checkpoints and ONNX export, and the reference generated from it
onnx/rewrite-genai-onnx/           the onnxruntime-genai graph rewrite the bind crates' build scripts share
assets/fetch-{model}/              a model's checkpoints when its reference is not an ONNX export
models/{model}-model-validation/   each model's cross-checks
tools/                             the fixture generators whose output lands in models/*/testdata
```

No command in the bunsen workspace builds anything here. Run a crate from
this directory, or from the root with `--manifest-path validation/Cargo.toml`
(the `cargo make` tasks below do). The crates depend on bunsen's by path;
`Cargo.lock` here is this workspace's own, and must build bunsen's crates at
the versions the root `Cargo.lock` has (`tools/check_validation_lock.py`,
which prints the refresh when it fails).

Nothing here fetches by default: `cargo test` here, with the default
features, runs the fixture checks and fetches nothing. CI builds nothing
here; `cargo make check-validation` runs those checks, with clippy and the
formatting check, by hand. Two features opt in, on each crate:

| feature | fetches | compiles |
|---|---|---|
| `weights` | the checkpoints | the checks that need no reference graph (Silero, Whisper) |
| `download` | `weights`, and the reference graphs | everything, with `gpu-tests` and a backend for the tests that load a model |

| command | what it runs |
|---|---|
| `cargo make validate-cpu` | the `weights` checks, on the CPU (~150 MB) |
| `cargo make validate-ci` | Silero's and Whisper's whole suites, on the CPU (~730 MB); `validation.yml` runs this when asked: from the Actions tab, or for a PR labelled `validate` |
| `cargo make validate-<model>` | one model's whole suite, in release mode, on `VALIDATION_BACKEND` (default `wgpu`) |
| `cargo make validate-all` | every model's whole suite (~60 GB on a cold cache) |

`<model>` is `silero`, `whisper`, `parakeet`, `qwen3-asr`, `qwen3`, `sortformer`, `voxtral` or `nanochat`.

bunsen's own tests read no fetched asset. A kit test that needs what a real
model does replays a committed recording of it (for example
`RecordedVoiceActivity::silero_jfk_moon_4s`), and the model's validation
crate checks the recording against the model.
