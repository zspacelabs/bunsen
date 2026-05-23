# Asserting and Unpacking

The three day-to-day contract macros, plus the hoisting helper for
naming a contract once and reusing it.

## Unpacking: `unpack_shape_contract!`

`unpack_shape_contract!` is the workhorse. It matches a shape against
a pattern, solves for unknown parameters, and returns the keys you
ask for as a fixed-size array.

```rust
# use bunsen::contracts::unpack_shape_contract;
let shape = [12, 3 * 4, 5 * 4, 3];

// In release builds this benchmarks at ~160 ns.
let [b, h_wins, w_wins, c] = unpack_shape_contract!(
    [
        "batch",
        "height" = "h_wins" * "window_size",
        "width"  = "w_wins" * "window_size",
        "channels",
    ],
    &shape,
    &["batch", "h_wins", "w_wins", "channels"],
    &[("window_size", 4)],
);

assert_eq!((b, h_wins, w_wins, c), (12, 3, 4, 3));
```

The arguments, in order, are:

1. The **contract pattern** (or the name of a pre-defined contract).
2. The **shape** (anything `ShapeView`).
3. The **keys** to unpack &mdash; a `&[&str]` of length `K`; the
   macro returns `[usize; K]` in the same order.
4. The **bindings** &mdash; a `&[(&str, usize)]` of parameters whose
   values are known up front.

Two shorthand forms are accepted:

- *No bindings.* When every parameter can be solved from the shape,
  the bindings slice may be omitted:

  ```rust
  # use bunsen::contracts::unpack_shape_contract;
  # let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
  let [h, h_win, w, w_win, ws, c] = unpack_shape_contract!(
      [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"],
      &shape,
      &["h", "h_win", "w", "w_win", "ws", "c"],
  );
  ```

- *Keys are the pattern.* When the pattern is just a list of bare
  identifiers, you can drop the keys slice as well:

  ```rust
  # use bunsen::contracts::unpack_shape_contract;
  let shape = [4, 5, 3];
  let [h, w, c] = unpack_shape_contract!(["h", "w", "c"], &shape);
  assert_eq!((h, w, c), (4, 5, 3));
  ```

If the shape does not match, or if the bindings are inconsistent
with the shape, the macro panics. See
[Error Messages](./error-messages.md) for what that panic looks like
and the `try_*` variants that return a `Result` instead.

## Asserting without unpacking: `assert_shape_contract!`

When you only care about *validating* a shape and don't need any
dimensions back, use `assert_shape_contract!`:

```rust
# use bunsen::contracts::assert_shape_contract;
let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];

assert_shape_contract!(
    [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"],
    &shape,
    &[("ws", 2)],
);
```

Same panic-on-mismatch semantics as `unpack_shape_contract!`. The
non-panicking form is
[`ShapeContract::try_assert_shape`](https://docs.rs/bunsen/latest/bunsen/contracts/struct.ShapeContract.html#method.try_assert_shape).

## Hoisting a contract: `define_shape_contract!`

If the same contract is used in more than one place &mdash; or you
simply want to read it once at the top of a function &mdash; bind it
to a `static` with `define_shape_contract!` and pass the name into
the assert or unpack macros:

```rust
# use bunsen::contracts::{
#     assert_shape_contract,
#     define_shape_contract,
#     unpack_shape_contract,
# };
# let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
define_shape_contract!(
    CONTRACT,
    [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"],
);

assert_shape_contract!(CONTRACT, &shape, &[("ws", 2)]);

let [h, h_win, w, w_win, c] = unpack_shape_contract!(
    CONTRACT,
    &shape,
    &["h", "h_win", "w", "w_win", "c"],
    &[("ws", 2)],
);
```

`shape_contract![...]` itself is a `const`-evaluable expression, so
the `static` carries no runtime construction cost.

For the hot-path variants of these macros &mdash;
`assert_shape_contract_periodically!` and the
`#[cfg(debug_assertions)]` pattern &mdash; see
[Cost Control](./cost-control.md).
