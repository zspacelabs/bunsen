# Upstream bug reproductions

bunsen works around some defects in burn and its ecosystem. This page
explains how each one is recorded, so that its workaround is removed when
the defect is fixed rather than left behind to become a bug of its own. The
reproductions live in
[`crates/dev/burn_bug_repro`](https://github.com/zspacelabs/bunsen/blob/main/crates/dev/burn_bug_repro/src/lib.rs),
whose crate doc is the reference.

## Two tests per defect

Each defect gets a module, and the module carries two kinds of test:

| Test | Asserts | Fails when | Runs by default |
|---|---|---|---|
| the reproduction | the **correct** semantics | the bug is present | no, it is `#[ignore]`d |
| the behaviour pin | the **current**, wrong, behaviour | the bug is fixed | yes |

The reproduction is what you would send upstream. It says what should
happen, and it fails today, so it is ignored to keep the suite green.

The behaviour pin is the half that protects bunsen. It asserts that the
defect is still there. When a dependency update fixes it, the pin fails, and
its message says what to do next: which workaround to remove, and which
reproduction to un-ignore. Without the pin, an upstream fix would arrive
silently, and the workaround would go on running against data that no longer
needs it.

## A pin names its workaround

`burn-store`'s PyTorch reader ignores tensor strides, which scrambles any
non-square weight that a checkpoint stores as a transposed view. bunsen
undoes the damage with
[`repair_pytorch_strided_weight`](bunsen::burner::store::repair_pytorch_strided_weight),
which [`FixPytorchLoadMappers`](bunsen::burner::store::FixPytorchLoadMappers)
attaches on the way to a PyTorch load
([PyTorch checkpoints](../systems/pretrained.md#pytorch-checkpoints)). Once
`burn-store` honours strides, the repair is itself the bug: a silent second
transpose. So the pin in the `pytorch_strided_weights` module fails with a
message that names `repair_pytorch_strided_weight` and its callers as the
thing to remove.

Not every workaround is dangerous to leave in place. `unfold` on the CubeCL
backends truncates an outer stride, and bunsen avoids it by trimming an
input to the span its windows cover. The trim changes nothing once the
defect is gone, so that pin's message asks for a re-check rather than a
removal. Write the message for the workaround you have.

## Writing a reproduction

- **Public API only.** Rest the reproduction on burn's public API, so that
  taking it upstream means swapping the test helpers, not rewriting the
  test. A test of bunsen's own workaround can sit in the same module; it
  just doesn't travel with the reproduction.
- **Document the defect in the module.** The existing modules each say what
  the expected semantics are, what goes wrong, why it is easy to miss, its
  scope, and how to recover from it.
- **Mind the CPU fallback.** A kernel defect that the CPU backend doesn't
  share can't be seen on a CPU-only run, where a pin would report a fix that
  hasn't happened. Gate such a pin on the backend features. A defect in a
  reader or a store is backend-independent and needs no gate
  ([Testing and backends](./testing.md#the-silent-cpu-fallback)).
- **Run it with a backend, in release mode.** The crate doc has the command.

## Why a separate crate

The reproductions used to live in `bunsen` itself. A reproduction is a
statement about somebody else's code: it carries a fixture, it wants a real
accelerator to say anything, and its whole purpose is to stop being true.
None of that belongs on a published library's public surface.
