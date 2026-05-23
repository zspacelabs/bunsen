# Contracts Overview

`bunsen::contracts` is a `no_std` inline contract programming library
for tensor geometry. It is built around three goals:

- contracts should be **easy to read, write, and use**,
- contracts should be **fast enough at runtime to always be enabled**,
- contracts should produce **verbose, helpful error messages** when
  they fail.

In practice this means contracts are written next to the code they
guard, match the way shapes are written in a paper
($x \in \RR^{B \times C \times H \times W}$), and stay enabled in
release builds.

API: <https://docs.rs/bunsen/latest/bunsen/contracts/>

## Why a contract system?

Shape errors in tensor code are notoriously hard to diagnose. A
reshape that almost-works produces a tensor with the wrong meaning,
not an exception. An off-by-one in a transpose silently swaps batch
and channels. The eventual failure &mdash; a `matmul` panic three
layers later, a loss that won't go down &mdash; points at a symptom,
not a cause.

A *contract* answers that by writing the expected shape down, as a
pattern, in code:

```rust,ignore
"batch",
"height" = "h_wins" * "window_size",
"width"  = "w_wins" * "window_size",
"channels",
```

This pattern is both a piece of documentation (the paper-style shape
$(B,\ h\cdot w_s,\ w\cdot w_s,\ C)$) and a runtime check. Every
contract is then used for one or both of two things:

- **Assert.** Does this tensor's shape match the pattern? If not,
  fail loudly with a useful message.
- **Unpack.** Assuming it matches, give back the named dimension
  sizes as concrete `usize` values to use in arithmetic and reshapes.

These are not separate features &mdash; *unpacking* implies
asserting. The same single pass through the shape both validates the
pattern and solves for whatever named parameters the caller asks
for.

This unifies two things that are otherwise written separately. In
ad-hoc code you tend to see, for the same tensor:

```rust,ignore
assert_eq!(tensor.dims()[0], batch);
assert!(tensor.dims()[1] % window_size == 0);
let h_wins = tensor.dims()[1] / window_size;
// ...and so on.
```

With a contract the assertion and the variable bindings come from
one declaration, and they cannot drift out of sync.

## Where contracts go

The natural home for a contract is a module boundary or function
boundary: the place where one piece of code hands shape responsibility
to another. The pattern is usually:

1. *Unpack* the inputs at the top of the function. You needed the
   dimensions anyway; the validation comes free.
2. Do the actual work, expressed in terms of the unpacked names.
3. *Assert* (often periodically) the shapes of intermediates or
   outputs that the function promises but doesn't otherwise consume.

When the function's docstring says
"Input tensor of shape $(B,\ h\cdot w_s,\ w\cdot w_s,\ C)$", the
contract at the top is the machine-checked version of that same
sentence.

## Why it's enabled in release

A contract that's too slow to keep on in release is one that only
catches bugs the author already hit in tests. The library is
designed around the assumption that contracts stay on:

- the pattern parser is a `const`-evaluable macro &mdash; contracts
  compile down to a `static` value, no per-call construction,
- the runtime path is stack-allocated and allocation-free on the
  happy path,
- a full unpack on a four-dimensional shape benches at ~160 ns,
- and an exponential-backoff variant ([periodic
  asserts](./cost-control.md#periodic-asserts)) brings the amortized
  cost on a hot path down to ~4 ns/call.

## The user-facing surface

Most code only ever touches three macros:

- [`unpack_shape_contract!`](https://docs.rs/bunsen/latest/bunsen/contracts/macro.unpack_shape_contract.html)
  &mdash; assert a contract *and* return named dimension sizes.
- [`assert_shape_contract!`](https://docs.rs/bunsen/latest/bunsen/contracts/macro.assert_shape_contract.html)
  &mdash; assert a contract for its side effect.
- [`assert_shape_contract_periodically!`](https://docs.rs/bunsen/latest/bunsen/contracts/macro.assert_shape_contract_periodically.html)
  &mdash; same, but amortized via an exponential-backoff scheduler.

These wrap a small layer-2 API that you can reach for when you want
to hoist work out of a hot path:

- [`shape_contract!`](https://docs.rs/bunsen/latest/bunsen/contracts/macro.shape_contract.html)
  &mdash; parse a contract pattern into a value.
- [`define_shape_contract!`](https://docs.rs/bunsen/latest/bunsen/contracts/macro.define_shape_contract.html)
  &mdash; bind a parsed contract to a `static`.
- [`ShapeContract::assert_shape`](https://docs.rs/bunsen/latest/bunsen/contracts/struct.ShapeContract.html#method.assert_shape)
  / [`ShapeContract::unpack_shape`](https://docs.rs/bunsen/latest/bunsen/contracts/struct.ShapeContract.html#method.unpack_shape)
  &mdash; the underlying methods (plus their `try_*` siblings that
  return `Result<_, String>` instead of panicking).
- [`run_periodically!`](https://docs.rs/bunsen/latest/bunsen/contracts/macro.run_periodically.html)
  &mdash; the amortization primitive used by
  `assert_shape_contract_periodically!`.

## Sub-chapters

The rest of this section unpacks the surface above:

- **[Pattern Syntax](./pattern-syntax.md)** &mdash; the contract DSL
  itself, plus what counts as a shape (`ShapeView`).
- **[Asserting and Unpacking](./asserting-and-unpacking.md)** &mdash;
  the `unpack_shape_contract!` / `assert_shape_contract!` /
  `define_shape_contract!` mechanics, including the shorthand forms.
- **[Cost Control](./cost-control.md)** &mdash; how to keep contracts
  enabled even on hot paths: periodic asserts,
  `#[cfg(debug_assertions)]`, and a comparison table.
- **[Error Messages](./error-messages.md)** &mdash; what a failing
  contract looks like, and the panic-vs-`try_*` split.
