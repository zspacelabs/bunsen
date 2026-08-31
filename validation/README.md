# validation

Reference implementations, and the cross-checks that step bunsen against them.

A bunsen kit is a by-inspection transliteration of somebody else's model. Its own unit tests check it against itself,
which cannot catch a shared misreading of the reference — and repeatedly did not. These crates pin each kit to something
with independent provenance.

| crate                         |                                                                                                                                            |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `silero-model-validation`     | Silero VAD against the generated ONNX reference, plus a golden trace over real audio.                                                      |
| `whisper-model-validation`    | Whisper against a pretrained ONNX export *and* `openai-whisper`'s own decode, judged by word error rate against a ground-truth transcript. |

## Weights are not validation

The *weights* each kit loads live beside the library, not here:
`crates/bunsen-bundled-silero` and `crates/bunsen-bundled-whisper`,
behind `bunsen`'s `silero-weights` and `whisper-weights` features. They are
shipped artefacts — what a user of the kit loads — whereas a reference
implementation exists only to be disagreed with.

Each bundle also carries the *reference* implementation for its model, behind
an `onnx_gen` feature: the ONNX-generated transliteration these crates step
bunsen against. So a validation crate generates nothing itself — it enables
`onnx_gen` on the bundle and compares. The checkpoint side arrives through
`bunsen`'s own feature (`silero-weights`, `whisper-weights`), so the path under
test is the one users take.

## Why these are not in the kits

Two reasons.

A generated transliteration of a graph exists to be *disagreed with*. Shipping one inside the library put a second,
redundant implementation on the public surface — `kits::speech::silero_vad::reference` was exactly that.

And validation assets are large and awkward: ONNX graphs in the hundreds of megabytes, checkpoints that must be fetched
and digest-pinned, audio fixtures, a tokenizer vocabulary. None of it belongs in a published crate, and none of it
should be on the path of an ordinary `cargo build`.

## Running

Each crate needs a backend feature — `PerformanceBackend` falls back to CPU silently when none is set, and a validation
run that quietly uses the CPU backend is not worth the wall clock, so the crates refuse to build instead.

```sh
cargo test --release -p silero-model-validation --features wgpu
cargo test --release -p whisper-model-validation --features download,wgpu
```

`whisper-model-validation` hosts no model assets: `build.rs` fetches them under the `download` feature, pins each to a
SHA-256, and caches them outside
`OUT_DIR`. Without that feature the crate builds to almost nothing and touches no network, so `cargo build --workspace`
is unaffected — and the checks that need no model still run.
