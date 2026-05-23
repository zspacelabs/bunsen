# Error Messages

When a contract fails, the panic message names exactly what didn't
fit. Producing useful diagnostics is the main user-visible reason to
write a contract in the first place; the unpacking and the
performance work exist to make using contracts cheap enough that you
reach for them by default.

## Anatomy of a failure

A failing contract reports the shape, the pattern that was expected,
the bindings in scope, and the specific dimension that did not
solve:

```rust
# use bunsen::contracts::{ShapeContract, shape_contract};
# use indoc::indoc;
static CONTRACT: ShapeContract = shape_contract![
    ...,
    "height" = "h_wins" * "window",
    "width"  = "w_wins" * "window",
    "color",
];

let h_wins = 2;
let w_wins = 3;
let window = 4;
let color  = 3;

let shape = [1, 2, 3, h_wins * window, w_wins * window, color];

// Passes: window=4 is consistent with the shape.
let [h, w] = CONTRACT.unpack_shape(
    &shape,
    &["h_wins", "w_wins"],
    &[("window", window), ("color", color)],
);
assert_eq!((h, w), (h_wins, w_wins));

// Fails: we lied about window. The error names the offending dim.
assert_eq!(
    CONTRACT
        .try_unpack_shape(
            &shape,
            &["h_wins", "w_wins"],
            &[("window", window + 1), ("color", color)],
        )
        .unwrap_err(),
    indoc! {r#"
        Shape Error:: 8 !~ height=(h_wins*window) :: No integer solution.
         shape:
          [1, 2, 3, 8, 12, 3]
         expected:
          [..., height=(h_wins*window), width=(w_wins*window), color]
          {"window": 5, "color": 3}"#
    },
);
```

Three things to notice in the error:

- the dimension is named (`height=(h_wins*window)`), not just
  indexed,
- the offending value (`8`) and the offending pattern are paired up,
- the active bindings (`{"window": 5, "color": 3}`) are printed so
  you can see what the contract was working with.

This is the payoff for writing the contract in the first place:
when something goes wrong, the panic message tells you *which*
invariant broke and *with what numbers*, instead of a bare
`assertion failed: lhs == rhs` from somewhere downstream.

## Panic vs `Result`

The macros (`assert_shape_contract!`, `unpack_shape_contract!`,
`assert_shape_contract_periodically!`) panic on mismatch. That's
the right default for the "contract guards a function boundary" use
case &mdash; a violation indicates a bug, and a panic is the
loudest, most-debuggable response.

When you want the failure as a value instead, use the underlying
methods on `ShapeContract`:

- [`ShapeContract::try_assert_shape`](https://docs.rs/bunsen/latest/bunsen/contracts/struct.ShapeContract.html#method.try_assert_shape)
  &mdash; `Result<(), String>`.
- [`ShapeContract::try_unpack_shape`](https://docs.rs/bunsen/latest/bunsen/contracts/struct.ShapeContract.html#method.try_unpack_shape)
  &mdash; `Result<[usize; K], String>`.

The `Err` payload is exactly the multi-line message shown above, so
you can log it, surface it to users, or run snapshot tests against
it (the example here uses `indoc!` for exactly that purpose).
