# validation

Expensive per-model validation crates: each holds one of bunsen's kits to an
independent reference, a Rust model `burn-onnx` generates from upstream's
ONNX export, and to upstream's own checkpoints. Every asset comes through a
`bind-*-onnx` crate, fetched at build time and digest-pinned.

Nothing here runs by default: a plain `cargo build --workspace` or
`cargo test` fetches nothing. Two features opt in, on each crate:

| feature | fetches | compiles |
|---|---|---|
| `weights` | the checkpoints | the checks that need no reference graph (Silero, Whisper) |
| `download` | `weights`, and the reference graphs | everything, with `gpu-tests` and a backend for the tests that load a model |

| command | what it runs |
|---|---|
| `cargo make validate-cpu` | the `weights` checks, on the CPU (~150 MB) |
| `cargo make validate-ci` | Silero's and Whisper's whole suites, on the CPU (~730 MB); `validation.yml` runs this when asked: from the Actions tab, or for a PR labelled `validate` |
| `cargo make validate-<model>` | one model's whole suite, in release mode, on `VALIDATION_BACKEND` (default `wgpu`) |
| `cargo make validate-all` | every model's whole suite (~50 GB on a cold cache) |

`<model>` is `silero`, `whisper`, `parakeet`, `qwen3-asr`, `qwen3`, `sortformer` or `voxtral`.

bunsen's own tests read no fetched asset. A kit test that needs what a real
model does replays a committed recording of it (for example
`RecordedVoiceActivity::silero_jfk_moon_4s`), and the model's validation
crate checks the recording against the model.
