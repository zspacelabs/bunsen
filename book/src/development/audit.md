# Audit probes and baselines

An audit records what a computation produced at named checkpoints, then
checks that another run produces the same thing. This page is about when to
reach for one and how to make it trustworthy.
[`bunsen::audit`] documents the probes, the harnesses and the
stream format, starting with a [quick start](bunsen::audit#quick-start).

## When an audit beats a unit test

A unit test asserts what a value *should* be, so its author has to know the
right answer. For a lot of numerical code nobody can write it down: a
softmax over seeded noise, the fourth block of an encoder, a simulation
after a thousand steps. An audit asserts instead that a value *has not
changed* from a run you trust. That suits three jobs:

- **Two backends agree.** Run the same code on a reference backend and a
  target, and compare every checkpoint.
- **A refactor changed nothing.** Record before the change, verify after it.
- **A result holds over time.** Store a run as a baseline, and verify later
  runs against it.

An audit doesn't say the trusted run was right. Pair it with something that
does: a unit test of the cases you can derive by hand, or a
[validation](./validation.md) against an independent implementation.

## Writing a body

The code under test is an [`AuditBody`](bunsen::audit::AuditBody), generic
over the backend so a harness can run it twice. It takes an
[`AuditProbe`](bunsen::audit::AuditProbe) and calls a checkpoint wherever an
intermediate value is worth pinning down. Three choices decide whether the
audit means anything:

- **Inputs.** Make them with
  [`seeded_tensor`](bunsen::support::testing::seeded_tensor), not
  `Tensor::random`, or the two runs start from different data.
- **Exact or approximate.** Check exactly what was uploaded, copied or
  rearranged, and check within a tolerance what was computed
  ([choosing a checkpoint](bunsen::audit#choosing-a-checkpoint)). The
  tolerance is recorded with the event, so a verifying run can't quietly
  loosen it.
- **Order.** Checkpoints match by position, not by label, so keep the
  sequence deterministic: no checkpoint inside a loop whose trip count
  depends on the backend
  ([how events match](bunsen::audit#how-events-match)).

Checkpoints return errors rather than panicking. Stop the body at the first
`Err`: the first divergence is the one worth reading, and later checkpoints
only repeat it.

## Across backends

[`audit_across`](bunsen::audit::audit_across) runs a body on a reference
backend, recording, then on a target backend, verifying, all in memory. The
usual pair is [`CpuBackend`](bunsen::support::testing::CpuBackend) as the
reference and
[`PerformanceBackend`](bunsen::support::testing::PerformanceBackend) as the
target. Build with a backend feature: without one, `PerformanceBackend` is
the CPU as well, and the audit compares the CPU with itself
([the silent CPU fallback](./testing.md#the-silent-cpu-fallback)).

## Baselines

[`audit_baseline`](bunsen::audit::audit_baseline) compares a run with a
stream stored on disk. Baselines are per backend: a CPU baseline never
verifies a GPU run. Use `audit_across` to compare backends, and a baseline to
catch change over time.

The library never chooses where baselines live. The call site does, and a
repository writes one constructor for its default location
([where baselines live](bunsen::audit#where-baselines-live)). Keep them
under the build directory to keep them out of git, or in the source tree to
check them in.

[`BaselineMode`](bunsen::audit::reports::BaselineMode) says what a run does
with a stored baseline. The default records a missing baseline and verifies
an existing one, which suits a developer's machine. Use `Record` to accept an
intended change, and `Verify` in CI, so that a missing baseline is an error
rather than silently becoming the new truth
([record or verify](bunsen::audit#record-or-verify)).

## Shapes and values

Audits are one half of how bunsen checks itself. The other half is
[shape contracts](../systems/contracts.md). A contract checks, on every
call, that a tensor has the shape its docs promise, and catches a wrong
reshape at the boundary where it happens. An audit checks that the values in
a correctly shaped tensor are the ones a trusted run produced.
