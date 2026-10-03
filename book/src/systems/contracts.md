# Shape contracts

A shape contract is a runtime check of a tensor's shape against a pattern
that reads like the shape in the docs. This chapter explains why bunsen
checks shapes this way, where contracts go, how to choose what a check
costs, and how to read a failure. The pattern language, the solver and the
macros are documented in [`bunsen::contracts`].

## Why contracts

Shape errors in tensor code are hard to diagnose. A reshape that almost
works produces a tensor with the wrong meaning, not an error. An off-by-one
in a transpose silently swaps two axes. The eventual failure, a `matmul`
panic three layers later or a loss that won't go down, points at a symptom,
not the cause.

A contract writes the expected shape down, in code, at the boundary where
one piece of code hands a tensor to another. It is used in one of two ways:

- **Assert:** does this shape match the pattern? If not, fail with a
  message that says which dimension broke.
- **Unpack:** the shape matches, so return the named dimension sizes the
  function needs for its arithmetic and reshapes.

Unpacking asserts too: the same pass that checks the pattern solves for the
names the caller asked for. This removes a familiar kind of drift. Ad-hoc
code checks `dims()[1] % window == 0` in one line and computes
`dims()[1] / window` in the next, and the two can disagree. With a contract,
the check and the values come from one declaration.

## Contracts read like the docs

STYLE.md writes a tensor shape in rustdoc as one code span in square
brackets, such as `[batch, h_wins*size, w_wins*size, channels]`. A contract
for that tensor uses the same names and the same arithmetic, one term per
dimension. A reader can then check the docs against the code at a glance,
and the contract enforces what the docs promise.
[The module docs](bunsen::contracts#contracts-read-like-the-docs) have a
compiled example of the correspondence.

A contract's names are its own: `"batch"` in one function has nothing to do
with `"batch"` in another. What makes a contract readable is matching the
docstring next to it.

## Where contracts go

The natural home is a function or module boundary. The usual pattern has
three steps:

1. **Unpack** the inputs at the top of the function. You needed the
   dimensions anyway, and the check comes with them.
2. Do the work in terms of the unpacked names.
3. **Assert** the shapes of outputs or intermediates that the function
   promises but does not otherwise read, usually periodically.

Pass a tensor's shape as `&x.dims()`, which the check borrows, rather than
`&x`, which copies the shape on every call
([passing shapes](bunsen::contracts#passing-shapes)). The solver walks the
shape left to right and can solve for one unknown per dimension, so a
dimension like `"h_wins" * "window"` needs `window` bound by the caller or
by an earlier term ([matching](bunsen::contracts::ShapeContract#matching)).

## Choosing a cost model

Contracts are meant to stay on in release builds. A contract that is too
slow to leave on only catches the bugs its author already hit in tests. The
pattern is parsed at compile time and kept in a `static`, so a check is one
pass over the dimensions, but it still does some work: it allocates a small
scratch vector, and it formats a message on failure. The
[cost section](bunsen::contracts#cost) has the benchmark command and
indicative numbers.

When a check sits on a hot path, there are two ways to lower its cost, and a
function often uses both:

| Strategy | Debug builds | Release builds | Use when |
| --- | --- | --- | --- |
| [`assert_shape_contract!`](bunsen::contracts#macros-and-methods) / [`unpack_shape_contract!`](bunsen::contracts#macros-and-methods) | every call | every call | The path is cold or runs once per batch, or you want the failure in production. |
| [`assert_shape_contract_periodically!`](bunsen::contracts#macros-and-methods) | sampled | sampled | The path is hot, but you still want some coverage in production. |
| Either, under `#[cfg(debug_assertions)]` | as above | removed | The check is a developer aid only, or you measured a cost you can't afford. |

The periodic form runs the first few calls, then doubles the gap between
checks until it reaches one call in a thousand. The schedule is per call
site
([`run_periodically!`](bunsen::contracts#macros-and-methods)). An unpack
cannot be sampled, because the function needs its values on every call.
Sample the asserts on intermediates and outputs instead.

The default is no `#[cfg]`. Reach for `#[cfg(debug_assertions)]` only after
measuring, or when the check guards against mistakes that a release build
cannot make. When you do gate a check:

- **Gate the whole chain.** The names a gated unpack binds exist only in
  debug builds, so every check that uses them must be gated too. The
  compiler enforces this, which is useful: it tells you when release code
  depends on a debug-only binding.
- **Periodic and gated combine.** A periodic assert inside a gated block
  still keeps debug builds and test runs fast.

[`next_interior_3d`](bunsen::kits::sims::conway::ops::next_interior_3d) is a
small example: a gated unpack of its input and a gated periodic assert of
its shrunken output. The ResNet blocks, such as
[`LayerBlock::forward`](bunsen::kits::images::resnet::blocks::LayerBlock::forward),
use the same pattern.

## Reading a failure

A failure message opens with the code that called the check. The next line
names the first dimension that failed, the pattern term it failed against,
and why. Then come the actual shape, the whole contract, and every value
bound when the match stopped, including the values the match itself bound on
the way. Read the failing term first, then the bindings: they show what the
solver believed when it gave up.
[`ShapeContract`'s error messages](bunsen::contracts::ShapeContract#error-messages)
describe each part and each reason.

All the macros panic on a mismatch. To get the failure as a value, for
example to report a bad input as an error, define the contract with
[`shape_contract!`](bunsen::contracts#macros-and-methods) and call
[`try_unpack_shape`](bunsen::contracts::ShapeContract::try_unpack_shape) or
[`try_assert_shape`](bunsen::contracts::ShapeContract::try_assert_shape).
Their `# Errors` sections say what they report, and `try_unpack_shape`'s
`# Panics` says what still panics.

## Reference

- [How the types relate](bunsen::contracts#how-the-types-relate) and
  [macros and methods](bunsen::contracts#macros-and-methods): the module
  overview.
- [The pattern language](bunsen::contracts#macros-and-methods):
  terms, labels, params, constants and the grammar.
- [Matching](bunsen::contracts::ShapeContract#matching): the solver's rules
  and its known gaps.
- [`ShapeView`](bunsen::contracts::ShapeView): every accepted shape form and
  what each one costs.
