# Validation against reference implementations

A kit that ports a published model makes a claim: it computes what the
original computes. This page describes how bunsen tests that claim. The two
validation crates are the worked examples, and their own docs carry the
details and the commands:
[`whisper-model-validation`](https://github.com/zspacelabs/bunsen/blob/main/crates/validation/whisper-model-validation/README.md)
and
[`silero-model-validation`](https://github.com/zspacelabs/bunsen/blob/main/crates/validation/silero-model-validation/src/lib.rs).

## Why unit tests are not enough

bunsen's Whisper and Silero kits are transliterations, written by reading
the reference. Their unit tests check the code against itself, so a
misreading that the code and its tests share passes: weights that load
scrambled, or a ReLU where the reference uses GELU, leave a unit suite
green. Validation pins a kit to something with independent provenance.

## A reference that runs inside burn

The reference is the upstream model's ONNX export, converted to Rust by
`burn_onnx::ModelGen`. Both implementations then run inside burn, on the same
backend, so a disagreement is a bunsen bug and not a difference between
frameworks. The export has to be the same model as the checkpoint bunsen
loads. Whisper's is a conversion of OpenAI's `base` checkpoint, so the two
agree only if bunsen's loader and its forward pass are both right.

The reference lives with the validation crate, or with the
[bundled-asset crate](./assets.md#the-two-bundled-crates) that supplies it,
and never in `bunsen`. It exists to be disagreed with, and shipping it would
put a second implementation of the model on the public surface.

## Staged, then composed

There are two kinds of test, because each catches what the other lets
through.

**Staged** tests feed each stage synthetic input and compare it with the
same stage of the reference, on identical weights. An error can't be
inherited from upstream, so a failure names the stage that is wrong. A
decoder's logits and its argmax are checked separately: across tens of
thousands of classes an elementwise tolerance can pass while the predicted
token differs, and the token is what a decoder is judged on.

**Composition** tests run the whole model on real input and judge the
result the way a user would. Whisper transcribes real speech and is scored
by word error rate, against a ground-truth transcript and against
`openai-whisper`'s own decode of the same clip. Silero runs a real
recording through its streaming context and pins the per-chunk
probabilities. Every stage can agree within tolerance while the composition
diverges, because a greedy decode turns a small numerical difference into a
different word.

## Tolerances from measured drift

The binding constraint is the backend, not the implementations. Two correct
implementations, several blocks deep, drift apart by as much as the
backend's arithmetic does, and that differs by backend: CUDA's
reduced-precision matmul drifts an order of magnitude further than wgpu. So
a tolerance is measured on each backend and set with headroom over the
worst. It is still far tighter than the error a real defect causes: a
scrambled weight or a wrong activation is off by 100% or more
([Tolerance](https://github.com/zspacelabs/bunsen/blob/main/crates/validation/whisper-model-validation/README.md#tolerance)).

## Fixtures generated once

Some references are not Rust and won't be: `librosa`'s mel filterbanks,
`openai-whisper`'s decodes. bunsen's tests never call them. A script under
[`tools/`](https://github.com/zspacelabs/bunsen/tree/main/tools) runs the
reference once and writes its outputs into a `testdata/` directory, and the
outputs are committed. The Rust tests read the files. Nothing shells out to
Python, and the reference is not a build or test dependency.

Regenerating a fixture is a deliberate manual step, never part of CI. Each
fixture script's docstring pins the tool versions it was run with and says
how to set up its environment. The scripts run the reference on the CPU,
because a fixture only has to be correct, and a CPU run is deterministic. Where the inputs came
from, and under what licence, is recorded in the `testdata/` READMEs, such
as
[Whisper's speech fixtures](https://github.com/zspacelabs/bunsen/blob/main/crates/validation/whisper-model-validation/testdata/README.md).

The same pattern pins data. `tools/gen_shard_digests.py` reads a dataset's
listing at a pinned revision and writes the digest table that a
[shard set](../systems/shards.md) verifies against.

## Running validation

The Whisper suite takes three features, and none implies another:
`download` fetches the assets it compares against, `gpu-tests` compiles the
tests that load a model, and a backend feature says what they run on.
Without `gpu-tests` only its fixture-integrity checks run, and that is all of
it that CI runs. The full suite is run by hand, in release mode, with a
backend; the
[README](https://github.com/zspacelabs/bunsen/blob/main/crates/validation/whisper-model-validation/README.md#running-it)
has the command. Without a backend feature the suite still passes, having
measured the CPU ([Testing and backends](./testing.md#the-silent-cpu-fallback)).

## Validating a new kit

A kit that claims parity with an upstream implementation gets a crate under
`crates/validation/`. Its reference and the checkpoint it is compared on
come through a bundled-asset crate, digest-pinned
([Assets, bundling and the network](./assets.md)), and the crate tests in
both layers: staged on synthetic input, composed on real input with a
measured tolerance. [The workspace](../organization/workspace.md#validation-crates)
lists the validation crates there are.
