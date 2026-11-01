# validation

A Cargo workspace of its own, apart from the bunsen workspace at the
repository root, of expensive per-model validation crates: each holds one of
bunsen's kits to an independent reference, a Rust model `burn-onnx` generates
from upstream's ONNX export, and to upstream's own checkpoints. Every asset
comes through a `bind-*-onnx` crate, fetched at build time and digest-pinned.

```text
onnx/bind-{model}-onnx/            each model's checkpoints and ONNX export, and the reference generated from it
onnx/rewrite-genai-onnx/           the onnxruntime-genai graph rewrite the bind crates' build scripts share
models.yml, profiles/               the catalog of models and the profiles that choose from it
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
formatting check, by hand. Each crate's features opt in:

| feature | fetches | compiles |
|---|---|---|
| `checkpoints` | the checkpoints, at run time, by the tests that read them, through bunsen's pinned rows into the weights cache | the tests against the checkpoints |
| `reference` | the ONNX reference and the bundle the conversion is cross-checked with, at build time | the reference checks, and the checks of the bundle |
| `gpu-tests` | nothing | the tests that run a model on a backend |

| command | what it runs |
|---|---|
| `cargo make validate` | the models `VALIDATION_PROFILE` chooses (default `small-core`), every crate ("Choosing what to validate", below) |
| `cargo make validate-cpu` | Silero's and Whisper's checks on the checkpoints alone, on the CPU (~150 MB) |
| `cargo make validate-ci` | the stream drivers on the smallest models, on the CPU: the `ci` profile (~0.6 GB); `validation.yml` runs this when asked: from the Actions tab, or for a PR labelled `validate` |
| `cargo make validate-<model>` | one crate's every model (`large-all`), in release mode, on `VALIDATION_BACKEND` (default `wgpu`) |
| `cargo make validate-all` | every model's whole suite (~95 GB on a cold cache) |

`<model>` is `silero`, `whisper`, `parakeet`, `qwen3-asr`, `qwen3`, `sortformer`, `voxtral` or `nanochat`.

## Choosing what to validate

A machine validates what it can hold, chosen by a **profile** from a
**catalog** of the models, and run through
[`cargo nextest`](https://nexte.st) (this workspace alone; the bunsen
workspace runs `cargo test`). Every crate's tests fetch their checkpoints at run time, through bunsen's own
pinned rows into the weights cache (`BUNSEN_CACHE_DIR`, or the user's cache
directory), so a model is fetched once, when its first test runs, and shared
with everything else that loads it through bunsen. Only a crate's ONNX
reference is a build-time asset, with the bundle of checkpoints the
conversion is cross-checked with: bundling is for verification alone, so it
comes in only under `reference`. `BUNSEN_VALIDATE_OFFLINE=1` keeps the tests
off the network.

```sh
cargo make validate                                   # VALIDATION_PROFILE, else small-core
VALIDATION_PROFILE=medium-core VALIDATION_BACKEND=cuda cargo make validate
python3 validation/tools/validate.py list --profile medium-core   # what it chooses, and the GB
python3 validation/tools/validate.py run  --profile my-box.yml --crate qwen3
python3 validation/tools/validate.py check                        # the catalog against the tests
```

`validate.py` needs PyYAML (`pip install pyyaml`).

### The catalog: `models.yml`

One entry per model a crate checks, under the crate, the model's id being
bunsen's row name:

| key | meaning |
|---|---|
| `size` | `small` (fetches up to ~1 GB, its reference included), `medium` (up to ~10 GB) or `large` |
| `gb` | what it fetches on a cold cache |
| `core` | whether it is a representative: the one model for an architecture variation the kit has; the others repeat one |
| `reference` | whether its tests need the crate's ONNX reference, the crate's `reference` feature |
| `dtypes` | what it is checked in: `f32`, `bf16` (on `cuda` alone) |
| `varies` | the variation it stands for (`core`), or repeats |

A crate's entry names its `package`, its `runner` (`nextest`), the `features`
always on when it runs, and its `reference` feature.

### Profiles: `profiles/*.yml`

A profile chooses models from the catalog by **selectors**, each matching a
model when every key it has does: `crate`, `model`, `size` (one or a list) and
`core`. What it chooses is everything its `include` selectors match, less
everything its `deny` selectors match, added to what each profile it
`extends` chooses (a deny removes from those too).

```yaml
extends: medium-core                   # a profile, or a list of them
include:
  - {crate: qwen3, model: qwen3-1.7b}
deny:
  - {crate: voxtral}
dtypes: [f32, bf16]                    # default [f32]; bf16 runs on cuda alone
checks: [model, driver]                # the default; `driver` alone builds no ONNX reference
backend: cuda                          # the default for --backend
```

The shipped profiles are cumulative: `small-core`, `small-all`,
`medium-core`, `medium-all`, `large-core` and `large-all`, each `-core` the
representatives up to that size and each `-all` every model up to it.
`ci` is the stream drivers on the smallest models, on the CPU: what CI runs.
`profiles/example-machine.yml` is one machine's. A profile is named by
`--profile` (a name here, or a path), else `BUNSEN_VALIDATE_PROFILE`.

A **driver** check is a test with a `driver` segment in its path
(`speech::driver::phonon_2`): a stream driver held to the same model's
one-shot decode. Every other test of a model is a **model** check: the model
against its reference. `checks` chooses between them.

### Tests carry their model

Every test of a model has the model's id in its path, as
a whole segment: `qwen3-0.6b` (or `org/qwen3-0.6b`) is `qwen3_0_6b`, as in
`greedy::qwen3_0_6b` or `cross::qwen3_0_6b::tracks_the_reference`. A profile's
run drops the tests of every model it does not choose, and, without `bf16`,
the `cuda_bf16` test binaries; a test of no model (a fixture check) always
runs. `validate.py check` builds the crates and checks that every catalog
model has a test, and lists the tests of no model.

A crate's features:

| feature | what it compiles or fetches |
|---|---|
| `checkpoints` | the tests against the checkpoints, which fetch them at run time |
| `reference` | the ONNX reference, fetched and generated at build time, and its tests |
| `gpu-tests` | the `checkpoints` and `reference` tests that run a model on a backend |
| `wgpu`, `vulkan`, `cuda`, `metal` | the backend |

`validate.py` turns them on: `features` always, `reference` when a chosen
model needs it, and the backend's.

bunsen's own tests read no fetched asset. A kit test that needs what a real
model does replays a committed recording of it (for example
`RecordedVoiceActivity::silero_jfk_moon_4s`), and the model's validation
crate checks the recording against the model.
