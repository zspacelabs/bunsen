//! Support macros.

/// Parse a shape contract pattern at compile time.
///
/// Expands to a [`ShapeContract`](crate::contracts::ShapeContract) expression
/// built from `const fn`s, so it can initialise a `static`. The check macros
/// ([`unpack_shape_contract!`], [`assert_shape_contract!`],
/// [`assert_shape_contract_periodically!`]) and [`define_shape_contract!`]
/// take the same pattern in square brackets and call this for you.
///
/// ```rust
/// use bunsen::contracts::{
///     ShapeContract,
///     shape_contract,
/// };
///
/// static CONTRACT: ShapeContract =
///     shape_contract![_, "w" = "x" + "y", ..., "z" ^ 2];
///
/// // The caller binds `x`; `w`, `y` and `z` are solved.
/// let [w, y, z] =
///     CONTRACT.unpack_shape(&[1, 7, 2, 3, 9], &["w", "y", "z"], &[("x", 3)]);
/// assert_eq!([w, y, z], [7, 4, 3]);
/// ```
///
/// # Pattern language
///
/// A pattern is a comma-separated list of terms; a trailing comma is
/// allowed. The terms match the shape's dimensions in order:
///
/// - `_` matches one dimension of any size;
/// - `...` matches zero or more dimensions. A pattern holds at most one: a
///   second panics in
///   [`ShapeContract::new`](crate::contracts::ShapeContract::new), which is a
///   compile error when the contract is a `static`;
/// - an expression matches one dimension whose size equals its value.
///
/// A `_` or expression term can carry a label, `"name" = term`. The label is a
/// param, bound to the size of the dimension the term matched, so `"h" = _`
/// names a dimension without constraining it. `...` can't be labelled: it
/// matches a run of dimensions, not one size, so `"rest" = ...` is a compile
/// error.
///
/// An expression is built from:
///
/// - params: any string literal, such as `"batch"`. The text isn't checked, so
///   `"window size"` is a valid name;
/// - integer constants, such as the `3` in `3 * "c"` or the `2` in `"h" - 2`;
/// - binary `+`, `-` and `*`, and `^` with a positive integer literal exponent.
///   `^ 0` is a compile error: the term would be 1 for any base;
/// - unary `-` and `+`, and parentheses.
///
/// There is no division: write `"h" = 2 * "half"` and unpack `"half"`.
///
/// `^` binds tightest, then `*`, then `+` and `-`. A unary sign belongs to
/// its factor, so it applies before `^`: `-"x" ^ 2` parses as `(-"x") ^ 2`,
/// which equals `"x" ^ 2` but doesn't solve like it. The solver takes the
/// non-negative root, 3 for a size of 9, so it solves `-"x"` to 3, and the
/// solution `x = -3` fails as negative. Write `"x" ^ 2`, or `-("x" ^ 2)` for
/// the negated square.
///
/// ```bnf
/// ShapeContract => DimMatcher { ',' DimMatcher }* ','?
/// DimMatcher    => '...' | { Param '=' }? ( '_' | Expr )
/// Expr          => Term { AddOp Term }*
/// Term          => Power { MulOp Power }*
/// Power         => Factor { '^' <positive integer literal> }?
/// Factor        => Param | Const | '(' Expr ')' | NegOp Factor
/// Param         => <string literal>
/// Const         => <integer literal>
/// NegOp         => '+' | '-'
/// AddOp         => '+' | '-'
/// MulOp         => '*'
/// ```
///
/// # Names and order
///
/// Every param and label in the pattern becomes a name in the contract's
/// [`index`](crate::contracts::ShapeContract::index), sorted. Bindings and
/// unpack keys must use these names, and error messages list bound names in
/// this order.
///
/// A contract is solved left to right, and each dimension can solve at most
/// one unknown param. So put the term that introduces a name before the
/// products that use it: `["a", "a" * "b"]` solves both names, while
/// `["a" * "b", "a"]` fails unless the caller binds `"a"` or `"b"`.
/// [`ShapeContract`](crate::contracts::ShapeContract#matching) has the full
/// rules.
pub use crate::__shape_contract as shape_contract;

#[doc(hidden)]
#[macro_export]
macro_rules! __shape_contract {
    ($($t:tt)*) => {
        {
        extern crate alloc;
        #[allow(unused_imports)]
        use alloc::boxed::Box;
        #[allow(unused_imports)]
        use $crate::contracts::{ShapeContract, DimExpr, DimMatcher};
        $crate::__proc_shape_contract!($($t)*)
    }};
}

/// Run code on a schedule that thins out to once every `period` calls.
///
/// Each call site keeps its own schedule. The first 10 calls run the code.
/// After that, the gap between runs starts at 1 and doubles after every run
/// (1, 2, 4, ...) until it reaches `period`, then stays there. The doubling
/// is capped, so any period is reached exactly. With the default period of
/// 1000, and counting calls from 0, the code runs on calls 0 through 10, 12,
/// 16, 24, 40, 72, 136, 264, 520, 1032, 2032, 3032, and so on.
///
/// [`assert_shape_contract_periodically!`] runs its check through this macro,
/// with the default period.
///
/// # Arguments
///
/// `run_periodically!(period, code)` or `run_periodically!(code)`:
///
/// - `period`: an integer literal (not a variable or a `const`); optional,
///   default 1000.
/// - `code`: an expression or block of type `()`, evaluated only on the calls
///   that run.
///
/// # State
///
/// The schedule is a pair of `static` atomic counters inside the macro's
/// expansion, so it belongs to the call site:
///
/// - every caller and every thread that reaches the call site shares it, as
///   does every instantiation of a generic function around it;
/// - it never restarts: once a call site has thinned out, it stays thinned out;
/// - the counters use relaxed atomics, so under concurrent calls the schedule
///   is approximate.
///
/// # Example
///
/// ```rust
/// use bunsen::contracts::run_periodically;
///
/// let mut ran = Vec::new();
/// for call in 0..30 {
///     run_periodically!(4, ran.push(call));
/// }
/// assert_eq!(ran, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28]);
/// ```
pub use crate::__run_periodically as run_periodically;

#[doc(hidden)]
#[macro_export]
macro_rules! __run_periodically {
    ($code:expr) => {
        $crate::__run_periodically!(@internal 1000, $code)
    };

    ($period:literal, $code:expr) => {
        $crate::__run_periodically!(@internal $period, $code)
    };

    (@internal $period:literal, $($tt:tt)*) => {{
        if {
            use core::sync::atomic::AtomicUsize;
            use core::sync::atomic::Ordering;

            static PERIOD: AtomicUsize = AtomicUsize::new(1);
            static COUNTER: AtomicUsize = AtomicUsize::new(0);

            let effective_period = PERIOD.load(Ordering::Relaxed);
            let count = COUNTER.fetch_add(1, Ordering::Relaxed);

            if effective_period == 1 && count < 10 {
                true

            } else if (count % effective_period) == 0 {
                // Double the period, but do not exceed the specified maximum period.
                if effective_period < $period {
                    PERIOD.store(
                        (2 * effective_period).clamp(1, $period),
                        Ordering::Relaxed,
                    );
                }
                // Reset the counter when we alter the period;
                // or periodically reset it to avoid overflow.
                if effective_period < $period || count > $period * 100 {
                    COUNTER.store(1, Ordering::Relaxed);
                }
                true

            } else {
                false
            }
        } {
            $($tt)*
        }
    }};
}

/// Define a `static` [`ShapeContract`](crate::contracts::ShapeContract) from
/// a pattern.
///
/// `define_shape_contract!(NAME, [pattern])` expands to a private
/// `static NAME: ShapeContract<'static>` built by [`shape_contract!`], which
/// documents the pattern language. Pass `NAME` to the check macros, or call
/// its methods, such as the non-panicking
/// [`try_assert_shape`](crate::contracts::ShapeContract::try_assert_shape).
///
/// ```rust
/// use bunsen::contracts::define_shape_contract;
///
/// define_shape_contract!(
///     CONTRACT,
///     [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"]
/// );
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
/// assert!(CONTRACT.try_assert_shape(&shape, &[("ws", 2)]).is_ok());
/// assert!(CONTRACT.try_assert_shape(&shape, &[("ws", 3)]).is_err());
/// ```
pub use crate::__define_shape_contract as define_shape_contract;

#[macro_export]
#[doc(hidden)]
macro_rules! __define_shape_contract {
    ($name:ident, [ $($contract_expr:tt)* ] $(,)?) => {
        static $name: $crate::contracts::ShapeContract<'static> = $crate::contracts::shape_contract![$($contract_expr)*];
    };
}

/// Check a shape against a contract, and panic if it doesn't match.
///
/// Calls
/// [`ShapeContract::assert_shape`](crate::contracts::ShapeContract::assert_shape)
/// on a static contract.
///
/// # Arguments
///
/// `assert_shape_contract!(contract, shape, bindings)`:
///
/// - `contract`: a pattern in square brackets, in the [`shape_contract!`]
///   language (the macro defines a `static` for it at the call site); or the
///   name of a `static` contract, such as one from [`define_shape_contract!`].
/// - `shape`: anything that converts `Into<ShapeView>`; for a tensor, pass
///   `&x.dims()`. See [`ShapeView`](crate::contracts::ShapeView).
/// - `bindings`: a `&[(&str, usize)]` of the values the caller already knows.
///   Required: pass `&[]` when there are none. Unlike
///   [`unpack_shape_contract!`], this macro has no form without it.
///
/// # Panics
///
/// If the shape doesn't match, with the message described on
/// [`ShapeContract`](crate::contracts::ShapeContract#error-messages), located
/// at the macro call. Also if a binding names something the pattern doesn't
/// use.
///
/// There is no non-panicking macro form. For a `Result`, define the contract
/// and call
/// [`try_assert_shape`](crate::contracts::ShapeContract::try_assert_shape).
///
/// # Examples
///
/// ### With a pattern
///
/// ```rust
/// use bunsen::contracts::assert_shape_contract;
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
///
/// assert_shape_contract!(
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"],
///   &shape,
///   &[("ws", 2)],
/// );
/// ```
///
/// ### With a pre-defined contract
///
/// ```rust
/// use bunsen::contracts::{assert_shape_contract, define_shape_contract};
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
///
/// define_shape_contract!(
///   CONTRACT,
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"]);
///
/// assert_shape_contract!(CONTRACT,  &shape, &[("ws", 2)]);
/// ```
pub use crate::__assert_shape_contract as assert_shape_contract;

#[doc(hidden)]
#[macro_export]
macro_rules! __assert_shape_contract {
    ([ $($contract_expr:tt)* ], $($args:tt)*) => {{
        $crate::__define_shape_contract!(CONTRACT, [ $($contract_expr)* ]);
        $crate::__assert_shape_contract!(CONTRACT, $($args)*)
    }};

    ($name:ident, $shape:expr, $bindings:expr $(,)?) => {
        $name.assert_shape($shape, $bindings)
    };
}

/// Check a shape against a contract on a sampled schedule, and panic if a
/// sampled check fails.
///
/// Wraps [`assert_shape_contract!`] in [`run_periodically!`] with its default
/// period of 1000: the check runs on the first calls, then on fewer and
/// fewer, settling at one call in 1000. The schedule belongs to the call
/// site. There is no period argument; to choose one, call
/// [`run_periodically!`] around [`assert_shape_contract!`] yourself.
///
/// # Arguments
///
/// The same as [`assert_shape_contract!`]: `contract` (a pattern or the name
/// of a static), `shape`, and the required `bindings`.
///
/// # Panics
///
/// As [`assert_shape_contract!`] does, on the calls that run the check. A
/// mismatch on a skipped call goes unnoticed.
///
/// # Examples
///
/// ### With a pattern
///
/// ```rust
/// use bunsen::contracts::assert_shape_contract_periodically;
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 9];
///
/// assert_shape_contract_periodically!(
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", 3 * "c"],
///   &shape,
///   &[("ws", 2)],
/// );
/// ```
///
/// ### With a pre-defined contract
///
/// ```rust
/// use bunsen::contracts::{assert_shape_contract_periodically, define_shape_contract};
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
///
/// define_shape_contract!(
///    CONTRACT,
///    [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"]);
///
/// assert_shape_contract_periodically!(CONTRACT,  &shape, &[("ws", 2)]);
/// ```
pub use crate::__assert_shape_contract_periodically as assert_shape_contract_periodically;

#[doc(hidden)]
#[macro_export]
macro_rules! __assert_shape_contract_periodically {
    ($($args:tt)*) => {
        $crate::__run_periodically!($crate::__assert_shape_contract!($($args)*))
    };
}

/// Check a shape against a contract, and return the values of chosen names.
///
/// Calls
/// [`ShapeContract::unpack_shape`](crate::contracts::ShapeContract::unpack_shape)
/// on a static contract, and panics if the shape doesn't match.
///
/// # Arguments
///
/// `unpack_shape_contract!(contract, shape, keys, bindings)`:
///
/// - `contract`: a pattern in square brackets, in the [`shape_contract!`]
///   language (the macro defines a `static` for it at the call site); or the
///   name of a `static` contract, such as one from [`define_shape_contract!`].
/// - `shape`: anything that converts `Into<ShapeView>`; for a tensor, pass
///   `&x.dims()`. See [`ShapeView`](crate::contracts::ShapeView).
/// - `keys`: a `&[&str; K]` of the names to return. The macro returns a
///   `[usize; K]` of their values, in key order. Each key must be a name in the
///   pattern; its value may come from the bindings, a label, or the solver.
/// - `bindings`: a `&[(&str, usize)]` of the values the caller already knows.
///   Optional: omit it when there are none.
///
/// When the pattern is only string literals, such as `["h", "w", "c"]`,
/// `unpack_shape_contract!([...], shape)` uses the pattern as the keys.
///
/// # Panics
///
/// If the shape doesn't match, with the message described on
/// [`ShapeContract`](crate::contracts::ShapeContract#error-messages), located
/// at the macro call. Also if a key or a binding names something the pattern
/// doesn't use.
///
/// There is no non-panicking macro form. For a `Result`, define the contract
/// and call
/// [`try_unpack_shape`](crate::contracts::ShapeContract::try_unpack_shape).
///
/// # Examples
///
/// ### With a pattern
///
/// ```rust
/// use bunsen::contracts::unpack_shape_contract;
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 9];
///
/// let [h, h_win, w, w_win, c] = unpack_shape_contract!(
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", 3 * "c"],
///   &shape,
///   &["h", "h_win", "w", "w_win", "c"],
///   &[("ws", 2)],
/// );
/// assert_eq!(h, 8);
/// assert_eq!(h_win, 4);
/// assert_eq!(w, 10);
/// assert_eq!(w_win, 5);
/// assert_eq!(c, 3);
/// ```
///
/// ### With a pre-defined contract
///
/// ```rust
/// use bunsen::contracts::{define_shape_contract, unpack_shape_contract};
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
///
/// define_shape_contract!(
///    CONTRACT,
///    [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"]);
///
/// let [h, h_win, w, w_win, c] = unpack_shape_contract!(
///   CONTRACT,
///   &shape,
///   &["h", "h_win", "w", "w_win", "c"],
///   &[("ws", 2)],
/// );
/// assert_eq!(h, 8);
/// assert_eq!(h_win, 4);
/// assert_eq!(w, 10);
/// assert_eq!(w_win, 5);
/// assert_eq!(c, 3);
/// ```
///
/// ### With no bindings
///
/// This also works with pre-defined contracts.
///
/// ```rust
/// use bunsen::contracts::unpack_shape_contract;
///
/// let shape = [4, 12];
///
/// let [a, b] = unpack_shape_contract!(["a", "a" * "b"], &shape, &["a", "b"]);
/// assert_eq!(a, 4);
/// assert_eq!(b, 3);
/// ```
///
/// ### With the pattern as the keys
///
/// ```rust
/// use bunsen::contracts::unpack_shape_contract;
///
/// let shape = [4, 5, 3];
///
/// let [h, w, c] = unpack_shape_contract!(["h", "w", "c"], &shape);
/// assert_eq!(h, 4);
/// assert_eq!(w, 5);
/// assert_eq!(c, 3);
/// ```
pub use crate::__unpack_shape_contract as unpack_shape_contract;

#[doc(hidden)]
#[macro_export]
macro_rules! __unpack_shape_contract {
    ([ $($keys:literal),* $(,)? ], $shape:expr $(,)?) => {{
        $crate::__define_shape_contract!(CONTRACT, [ $($keys),* ]);
        $crate::__unpack_shape_contract!(CONTRACT, $shape, &[ $($keys),* ], &[])
    }};

    ([ $($contract_expr:tt)* ], $($args:tt)*) => {{
        $crate::__define_shape_contract!(CONTRACT, [ $($contract_expr)* ]);
        $crate::__unpack_shape_contract!(CONTRACT, $($args)*)
    }};

    ($contract:ident, $shape:expr, $keys:expr, $bindings:expr $(,)?) => {{
        $contract.unpack_shape($shape, $keys, $bindings)
    }};

    ($contract:ident, $shape:expr, $keys:expr $(,)?) => {{
        $crate::__unpack_shape_contract!($contract, $shape, $keys, &[])
    }};
}

#[cfg(test)]
mod tests {
    use alloc::{
        vec,
        vec::Vec,
    };

    use super::*;

    #[test]
    fn test_run_periodically() {
        let expected = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 40, 72, 136, 264, 520, 1032, 2032,
        ];

        // Block.
        {
            let mut results = Vec::new();
            for i in 0..2500 {
                run_periodically!({
                    results.push(i);
                });
            }
            assert_eq!(&results, &expected);
        }

        // Expression.
        {
            let mut results = Vec::new();
            for i in 0..2500 {
                run_periodically!(results.push(i));
            }
            assert_eq!(&results, &expected);
        }
    }
}
