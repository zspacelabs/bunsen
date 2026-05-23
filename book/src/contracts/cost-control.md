# Cost Control

`bunsen::contracts` is designed to stay enabled in release builds,
but a 160 ns check inside a tight inner loop still adds up. Two
mechanisms exist to keep contracts on while bringing their cost
down: an exponential-backoff *periodic* check, and a
`#[cfg(debug_assertions)]` gate that strips the check entirely from
release builds.

## Periodic asserts

`assert_shape_contract_periodically!` wraps an
`assert_shape_contract!` in
[`run_periodically!`](https://docs.rs/bunsen/latest/bunsen/contracts/macro.run_periodically.html):
the assertion runs the first **10** calls, then on a doubling
schedule (every 16, 32, 64, &hellip; calls) until it reaches a
configurable maximum period (default **1000**).

The amortized cost drops to ~4 ns/call in release builds:

```terminaloutput
assert_shape_every_nth   time: [4.4057 ns 4.4769 ns 4.5726 ns]
```

It takes the same arguments as `assert_shape_contract!`:

```rust
# use bunsen::contracts::assert_shape_contract_periodically;
let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];

assert_shape_contract_periodically!(
    [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"],
    &shape,
    &[("ws", 2)],
);
```

The typical pattern in a module method is:

1. **Unpack** the inputs once at the top &mdash; you need the
   dimensions anyway, and the assertion comes free.
2. **Periodically assert** intermediate or output shapes you don't
   need to unpack, so violations are still caught without paying the
   full cost on every call.

## Debug-only contracts: `#[cfg(debug_assertions)]`

Periodic asserts amortize cost; sometimes you'd rather pay *zero*
cost in release and accept that the check only runs in development
builds. The standard Rust way to do that is
`#[cfg(debug_assertions)]`, which gates an item on the same flag
that controls `debug_assert!`. The flag is on for `cargo build` /
`cargo test` and off for `cargo build --release`.

Applied to a contract, this strips the entire macro call &mdash;
pattern parsing, binding solve, everything &mdash; from release
builds:

```rust,ignore
pub fn next_interior_3d<B: Backend>(
    state: Tensor<B, 3, Bool>,
    rules: &LifeRules,
) -> Tensor<B, 3, Bool> {
    // Debug only: unpack [H, W, Z] for the output check below.
    #[cfg(debug_assertions)]
    let [h, w, z] = unpack_shape_contract!(["h", "w", "z"], &state.dims());

    // ... do work, producing `update` with shape [H-2, W-2, Z-2] ...
    # let update = state;

    // Debug only: confirm the interior shrink happened correctly.
    #[cfg(debug_assertions)]
    assert_shape_contract_periodically!(
        ["h" - "pad", "w" - "pad", "z" - "pad"],
        &update.dims(),
        &[("h", h), ("w", w), ("z", z), ("pad", 2)],
    );

    update
}
```

This is the pattern used by `bunsen::kits::sims::conway::life3d` for
its interior-step kernel.

A few things to note:

- **Bindings produced by a gated unpack are themselves gated.**
  Above, `h`, `w`, and `z` only exist in debug builds. Anything that
  consumes them &mdash; the downstream periodic assert &mdash; must
  be gated too, or release builds won't compile. This is usually a
  feature: the compiler tells you when you accidentally depend on a
  debug-only binding outside its `#[cfg]`.
- **Pair `#[cfg(debug_assertions)]` with the unpack/assert that
  produced its bindings.** Don't sprinkle the attribute across only
  half of a chain.
- **Combine freely with `assert_shape_contract_periodically!`.**
  Even inside a debug-only branch, a periodic check is still
  worthwhile if the function is called many times in tests or
  development runs: the amortization keeps debug builds responsive.

## Choosing the cost model

`bunsen::contracts` gives you three strategies for managing the
runtime cost of a check; they're not exclusive, and a function often
uses more than one:

| Strategy | Debug cost | Release cost | Use when |
| --- | --- | --- | --- |
| `assert_shape_contract!` / `unpack_shape_contract!` | full (~160 ns) | full (~160 ns) | The check is on a cold or once-per-batch path, *or* you want production failures. |
| `assert_shape_contract_periodically!` | amortized (~4 ns/call) | amortized (~4 ns/call) | The check is on a hot path but you still want some coverage in production. |
| `#[cfg(debug_assertions)]` on either of the above | full or amortized | **zero** | The check is purely a developer aid; the calling code is trusted in release, or the cost is unacceptable on any hot path. |

The default should be "no `#[cfg]`": contracts are designed to stay
on. Reach for `#[cfg(debug_assertions)]` when you've measured a cost
you can't afford or when the check exists only to catch upstream
bugs that release builds cannot introduce.

## Putting it together

A canonical use site from the `window_partition` example in the mod
docs:

```rust,ignore
use burner::prelude::{Tensor, Backend};
use burner::tensor::BasicOps;
use bunsen::contracts::{
    assert_shape_contract_periodically,
    unpack_shape_contract,
};

/// Window Partition
///
/// ## Parameters
///
/// - `tensor`: Input tensor of shape (B, h_wins * window_size, w_wins * window_size, C).
/// - `window_size`: Window size.
///
/// ## Returns
///
/// Output tensor of shape (B * h_windows * w_windows, window_size, window_size, C).
pub fn window_partition<B: Backend, K>(
    tensor: Tensor<B, 4, K>,
    window_size: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    // Validate the input and pull out what we need. ~160 ns.
    let [b, h_wins, w_wins, c] = unpack_shape_contract!(
        [
            "batch",
            "height" = "h_wins" * "window_size",
            "width"  = "w_wins" * "window_size",
            "channels",
        ],
        &tensor,
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)],
    );

    let tensor = tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c]);

    // Amortized check on the output. ~4 ns averaged.
    assert_shape_contract_periodically!(
        [
            "batch" * "h_wins" * "w_wins",
            "window_size",
            "window_size",
            "channels",
        ],
        &tensor,
        &[
            ("batch", b),
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("window_size", window_size),
            ("channels", c),
        ],
    );

    tensor
}
```

The input contract *unpacks* the dimensions the function needs; the
output contract *asserts* &mdash; periodically &mdash; that the
reshape sequence produced what the docstring promised.
