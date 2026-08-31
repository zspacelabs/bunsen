# whisper-model-validation

Validates bunsen's Whisper against two independent references — a Rust model
generated from a pretrained ONNX export, and `openai-whisper`'s own decode —
so a disagreement points at bunsen rather than at a framework difference.

**This crate hosts nothing and generates nothing.** Both sides of the
comparison come from [`bunsen-bundled-whisper`](../../crates/bunsen-bundled-whisper):

| what | how it arrives | feature |
|---|---|---|
| `base.pt`, the checkpoint bunsen loads | `bunsen/whisper-weights` → `Whisper::load_pretrained` | `checkpoint` |
| the ONNX-generated encoder/decoder | `bunsen-bundled-whisper/onnx_gen` | `onnx_gen` |

Routing the checkpoint through `bunsen` rather than reading it directly is
deliberate: the path under test is then the one users take.

## Running it

```sh
# ~435 MB of assets on a cold cache, fetched by the bundle's build script.
cargo test --release -p whisper-model-validation --features download,wgpu
```

**A backend feature is required**, and the crate refuses to build without one
(`wgpu`, `cuda`, or `metal`). `PerformanceBackend` falls back to CPU silently
when none is selected, and a cross-check that quietly runs a live model on the
CPU backend is not worth the wall clock — it would still pass, just too slowly
to ever be run.

**`--release` matters for the same reason.** The work is inside `burn`'s
kernels rather than in this crate: about half a minute optimized, minutes not.

Without `--features download` the crate builds to almost nothing and touches no
network, so `cargo build --workspace` is unaffected. The fixture-integrity
checks still run in that mode — they need no model.

Override any asset with a local file; these are read by the bundle's build
script, not this crate's:

| variable | effect |
|---|---|
| `WHISPER_ONNX_ENCODER` | use a local encoder `.onnx` (skips the digest pin) |
| `WHISPER_ONNX_DECODER` | use a local decoder `.onnx` (skips the digest pin) |
| `WHISPER_BASE_PT` | use a local checkpoint (skips the digest pin) |

The decoder export is the KV-cache-free `decoder_model.onnx`, which consumes a
whole token sequence at once — the shape `TextDecoder::forward` has. Its
`forward` returns logits followed by 24 present-key/value tensors, which this
crate ignores. (`decoder_with_past_model.onnx` is the incremental variant.)

The ONNX export and the checkpoint must be the same model: the export is a
conversion of that checkpoint, and the two agree to 1.9e-4 under
`onnxruntime`, which is what makes the comparison meaningful.

## What is covered

`staged` feeds each stage synthetic input, so a disagreement cannot be
inherited from upstream:

| test | checks |
|---|---|
| `test_reference_encoder_runs` | the fetch-and-generate path is healthy |
| `test_bunsen_encoder_matches_reference` | encoder, on identical weights |
| `test_bunsen_decoder_matches_reference` | decoder logits, on identical weights |
| `test_bunsen_decoder_argmax_matches_reference` | the predicted token at each position |

The argmax test is separate on purpose: logits span a wide range over 51865
classes, so an elementwise tolerance can pass while the argmax differs — and
the argmax is the only thing a decoder is actually judged on.

`audio` runs the composition over `testdata/`, and judges it the way a
transcription is judged — word error rate against a ground-truth transcript,
with a per-fixture ceiling. See [`testdata/README.md`](testdata/README.md) for
the fixtures, the measured rates, and how to regenerate them.

This is the layer that catches what the staged tests let through: each stage can
agree inside tolerance while the composition diverges, because a greedy argmax
turns a small numerical difference into a different word.

## Tolerance

The binding constraint is the backend, not the implementations. wgpu agrees
with the reference inside 1e-3 absolute; CUDA drifts to ~1.1e-2, the signature
of a reduced-precision matmul. The tolerance is set from measurement with
headroom over CUDA — still far tighter than any real defect, since the ones
this crate exists to catch were each wrong by 100% or more.

## Why this crate exists

bunsen's Whisper is a by-inspection transliteration. Its unit tests check it
against itself, which cannot catch a shared misreading of the reference — and
did not. Stepping it against the real implementation turned up defects that a
green suite had hidden:

- `PytorchStore` ignores PyTorch tensor strides, so every `Linear` weight in an
  OpenAI checkpoint loaded mangled (see `bunsen::burner::repro`).
- The MLP projection biases were dropped at load.
- The MLP used ReLU where Whisper uses GELU.
- Cross-attention's shape contract required both sequences to be the same
  length, so the decoder could not run against a real encoder output at all.

Each was invisible to a test suite that only compared bunsen to itself.
