# Testing and backends

This page covers how bunsen's tests choose a device, how they
keep an accelerator's memory in check, and how they compare numbers that no
two backends compute identically. The helpers are in
[`bunsen::support::testing`], behind the `testing`
feature.

## Two test devices

bunsen's tests take their device from one of two functions rather than
naming a burn backend:

- [`cpu_device`](bunsen::support::testing::cpu_device) is burn's `Flex`, a
  CPU backend. Tests of logic, shapes and plumbing use it: it starts at once
  and needs no hardware.
- [`performance_device`](bunsen::support::testing::performance_device) is
  the accelerator, chosen by a backend feature on **bunsen** (`cuda`,
  `metal`, `vulkan`, `wgpu`, in that order of preference), or by
  `BURN_DEVICE` in the environment. Tests that load a real model or exercise
  a real kernel use it, and so do the GPU benchmarks.

A test that trains calls `.autodiff()` on the device before it builds its
module and inputs.

### The silent CPU fallback

When no backend feature reaches bunsen, `performance_device()` is the CPU.
Nothing fails. The tests compile, run and pass, having measured the CPU. A
plain `cargo test` says nothing about a GPU, and an audit of `cpu_device()`
against `performance_device()` compares the CPU with itself.

**Always test with a backend feature**, such as `--features wgpu`. A crate
whose own backend features should move `performance_device()` must forward
them to bunsen (`wgpu = ["bunsen/wgpu"]`); enabling only `burn/wgpu` leaves
it on the CPU. [`performance_device`](bunsen::support::testing::performance_device#the-cpu-fallback-is-silent)
documents the selection order and the fallback, and STYLE.md's
[Backend selection](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#backend-selection)
has the rule.

The fallback matters in the other direction too. A defect in a GPU kernel
that the CPU backend doesn't share looks fixed on a CPU-only run, so a test
that pins such a defect is gated on the backend features
([Upstream bug reproductions](./repros.md)).

### Slow tests: `gpu-tests`

Some tests are too slow to be worth running without an accelerator, such as
loading a full checkpoint and stepping it against a reference. They compile
only under the `gpu-tests` feature, which is orthogonal to the backend
features: `gpu-tests` says *whether* the slow tests are built, and a backend
feature says *which* accelerator runs them. Pass both. Gate such a test on
`feature = "gpu-tests"`, never on "some backend feature is on", which would
leave no way to have one without the other
([GPU tests](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#gpu-tests)).

## Inputs that agree across backends

`Tensor::random` depends on the backend's generator and on global seeding.
[`seeded_tensor`](bunsen::support::testing::seeded_tensor) draws on the host
from a seeded generator and uploads the result, so every backend sees the
same values. Use it wherever two runs will be compared: across backends, in
an [audit](./audit.md), or against a stored fixture.

## Accelerator memory

`cargo test` runs every test in a binary in one process, so they share one
accelerator client and one cubecl memory pool. The pool keeps the pages it
has allocated, which is right within one model and wrong across a suite of
tests that each load their own: the pool only grows, until the card is full.
The failure is an out-of-memory error in whichever test was running at the
time, not in the one that allocated. Page sizes differ by backend, so a
suite that fits on wgpu can still run out on Vulkan.

A test that loads a model binds a
[`DeviceMemoryGuard`](bunsen::support::testing::DeviceMemoryGuard) at its
top, which hands the pool's free pages back when the test ends. Its docs
give the two binding rules that decide whether it reclaims anything. The
guard limits what a suite accumulates over time, not what concurrent tests
hold at once, so heavy tests also run `#[serial]`.

## Comparing numbers

Two backends rarely compute a float result bit for bit alike: they reduce in
different orders, and some accelerator kernels trade precision for speed. A
test that runs on whichever backend a developer builds therefore compares
computed values within a tolerance.
[`support::testing`](bunsen::support::testing#test-devices) states the rule
and has the assertions that apply it.

Set a tolerance from measurement, not by guessing. Run on each backend you
support, look at the drift, and leave headroom over the worst. A real defect
is usually far larger than any backend's drift
([Validation](./validation.md#tolerances-from-measured-drift)).

An elementwise tolerance is not always the right test. A decoder is judged
by the token it picks, and across tens of thousands of logits a tolerance
can pass while the argmax differs, so check the choice separately. Across a
whole transcript, compare text rather than token ids: one flipped token
cascades through the rest of the decode while the text barely moves.
[`support::testing::asr`](bunsen::support::testing::asr) has the word error
rate that bunsen's speech tests are judged by.

## Untrained toy models

The Whisper driver and decoder unit tests run on tiny models with random
weights, not on a checkpoint. An untrained model's logits are
full of near-ties, which flip between runs and backends, and on the CPU
backend its forward pass varies from call to call by more than a near-tie.
A test that needs a particular token sequence can't get it from the model.

Such a test scripts the decode instead. A test-only
[`LogitFilter`](bunsen::kits::speech::whisper::logit_filters::LogitFilter)
masks every logit except the scripted token for the current position, so
the decode follows the script whatever the model computes. The test is then
about what it set out to test, such as the stream driver's scheduling, seek
and emission, and not about an untrained model's numerics. A test that
wants the model's own choice seeds the model, so that a near-tie ranks the
same way on every run.

## Test modules

Every `#[cfg(test)] mod tests` opens with `use super::*;`, so its imports
track the parent's
([Test modules](https://github.com/zspacelabs/bunsen/blob/main/STYLE.md#test-modules)).
