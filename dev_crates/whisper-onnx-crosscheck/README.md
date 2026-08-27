# whisper-onnx-crosscheck

Cross-checks bunsen's Whisper against an independent implementation generated
from a pretrained ONNX export, so a disagreement points at bunsen rather than at
a framework difference.

**This crate hosts no model assets.** It fetches them, pins them to a SHA-256,
and caches them under `.cache/` (gitignored).

## Running it

```sh
# Fetches ~82 MB of ONNX on a cold cache, then generates and runs the reference.
cargo test --release -p whisper-onnx-crosscheck --features download,wgpu

# Full comparison: also needs OpenAI's multilingual base.pt, which is the
# checkpoint onnx-community/whisper-base was converted from.
WHISPER_BASE_PT=/path/to/base.pt \
  cargo test --release -p whisper-onnx-crosscheck --features download,wgpu
```

**A backend feature is required**, and the crate refuses to build without one
(`wgpu`, `cuda`, or `metal`). `PerformanceBackend` falls back to CPU silently
when none is selected, and a cross-check that quietly runs a live model on the
CPU backend is not worth the wall clock — it would still pass, just too slowly
to ever be run.

**`--release` matters for the same reason.** The work is inside `burn`'s
kernels rather than in this crate: about a second optimized, minutes not.

Without `--features download` the crate builds to nothing and touches no
network, so `cargo build --workspace` is unaffected.

Both assets are fetched by `build.rs`, so nothing skips. Override either with a
local file:

| variable | effect |
|---|---|
| `WHISPER_ONNX_ENCODER` | use a local encoder `.onnx` (skips the digest pin) |
| `WHISPER_ONNX_DECODER` | use a local decoder `.onnx` (skips the digest pin) |
| `WHISPER_BASE_PT` | use a local checkpoint (skips the digest pin) |

Assets, if you'd rather fetch them by hand (~430 MB total, cached):

- encoder — [`encoder_model.onnx`](https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/encoder_model.onnx)
- decoder — [`decoder_model.onnx`](https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/decoder_model.onnx)
- checkpoint — [`base.pt`](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)

The decoder export is the KV-cache-free `decoder_model.onnx`, which consumes a
whole token sequence at once — the shape `TextDecoder::forward` has. Its
`forward` returns logits followed by 24 present-key/value tensors, which this
crate ignores. (`decoder_with_past_model.onnx` is the incremental variant, for
whenever bunsen grows a KV cache.)

Generated weights are loaded from `OUT_DIR` at run time rather than embedded:
together they are ~290 MB, and `include_bytes!` of that would dominate compile
time and binary size.

They must be the same model: the ONNX export is a conversion of that
checkpoint, and the two agree to 1.9e-4 under `onnxruntime`, which is what
makes the comparison meaningful.

## What is covered

| test | checks |
|---|---|
| `test_reference_encoder_runs` | the fetch-and-generate path is healthy; no other assets |
| `test_bunsen_encoder_matches_reference` | encoder, on identical weights |
| `test_bunsen_decoder_matches_reference` | decoder logits, on identical weights |
| `test_bunsen_decoder_argmax_matches_reference` | the predicted token at each position |

The decoder is fed a **synthetic** encoder output rather than either encoder's
real one, so a decoder disagreement cannot be inherited from upstream.

The argmax test is separate on purpose: logits span a wide range over 51865
classes, so an elementwise tolerance can pass while the argmax differs — and
the argmax is the only thing a decoder is actually judged on.

## Tolerance

The binding constraint is the backend, not the implementations. wgpu agrees
with the reference inside 1e-3 absolute; CUDA drifts to ~1.1e-2, the signature
of a reduced-precision matmul. The tolerance is set from measurement with
headroom over CUDA — still far tighter than any real defect, since the ones
this crate exists to catch were each wrong by 100% or more.

## Why fetch at build time

`burn_onnx::ModelGen` generates Rust source, so it has to run in `build.rs` —
which means the `.onnx` must exist at build time. Cargo places no restriction on
network access in a build script, but a build that silently downloads breaks
`--offline`, breaks air-gapped CI, and makes builds non-reproducible. So the
fetch is constrained:

- **Off by default.** Gated behind the `download` feature; nothing happens
  without it.
- **Cached outside `OUT_DIR`.** `cargo clean` does not force an 82 MB refetch.
- **Digest-pinned.** Every asset is verified on each build, so a warm cache is
  offline and reproducible, and a changed upstream asset fails loudly instead of
  silently altering the reference.
- **Overridable.** `WHISPER_ONNX_ENCODER` bypasses the fetch entirely.

The alternative — committing the ONNX — would put ~82 MB of binary into the
repository for one dev-only test. The alternative to *that* — generating nothing
and hand-writing expected activations — is what this crate exists to replace.

## Why this crate exists

bunsen's Whisper is a by-inspection transliteration. Its unit tests check it
against itself, which cannot catch a shared misreading of the reference — and
did not. Stepping it against the real implementation turned up four defects that
a green suite had hidden:

- `PytorchStore` ignores PyTorch tensor strides, so every `Linear` weight in an
  OpenAI checkpoint loaded mangled (see `bunsen::burner::repro`).
- The MLP projection biases were dropped at load.
- The MLP used ReLU where Whisper uses GELU.
- Cross-attention's shape contract required both sequences to be the same
  length, so the decoder could not run against a real encoder output at all.

Each was invisible to a test suite that only compared bunsen to itself.
