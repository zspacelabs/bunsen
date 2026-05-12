//! Support macros.

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

/// A macro to run a block of code or an expression every nth time it is called.
///
/// Runs the first 10 times, then doubles the period on each subsequent call,
/// until it reaches the specified period, after which it continues to run at
/// that period.
///
/// This macro is useful for scenarios where you want to limit the execution
/// of code to every nth call, such as logging, sampling, or throttling
/// operations.
///
/// ## Arguments:
///
/// - `$period`: [optional; default=1000] the period.
/// - `$code`: An expression to be executed every nth time.
///
/// # Usage:
/// ```rust.no_run
///  use bunsen::contracts::run_periodically;
///
///  // Run a block of code every 3rd call
/// run_periodically!(3, {
///       println!("This will run every 3rd time.");
///       // Your code here
///  });
pub use crate::__run_periodically as run_periodically;

#[doc(hidden)]
#[macro_export]
macro_rules! __run_periodically {
    ($code:expr) => {
        $crate::__run_periodically!(@internal 1000, $code)
    };

    ($lock:block) => {
        $crate::__run_periodically!(@internal 1000, $block)
    };

    ($period:literal, $code:expr) => {
        $crate::__run_periodically!(@internal $period, $code)
    };

    ($period:literal, $lock:block) => {
        $crate::__run_periodically!(@internal $period, $block)
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

/// Define a static [`ShapeContract`](`crate::contracts::ShapeContract`).
///
/// See [`shape_contract`](`crate::contracts::shape_contract`) for documentation
/// on the contract syntax.
///
/// ```rust,no_run
/// use bunsen::contracts::define_shape_contract;
///
/// define_shape_contract!(
///   CONTRACT,
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"]);
/// ```
pub use crate::__define_shape_contract as define_shape_contract;

#[macro_export]
#[doc(hidden)]
macro_rules! __define_shape_contract {
    ($name:ident, [ $($contract_expr:tt)* ] $(,)?) => {
        static $name: $crate::contracts::ShapeContract<'static> = $crate::contracts::shape_contract![$($contract_expr)*];
    };
}

/// Define and call
/// [`ShapeContract::assert_shape`](`crate::contracts::ShapeContract::assert_shape`) on a static shape
/// contract.
///
/// See [`crate::contracts::shape_contract`] for documentation
/// on the contract syntax.
/// See [`crate::contracts::ShapeContract::assert_shape`] for documentation on
/// the assertion api.
///
/// ### With a Contract Expression:
///
/// ```rust,no_run
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
/// ### With a pre-defined contract:
///
/// ```rust,no_run
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

/// A macro which periodically calls
/// [`assert_shape_contract`](`crate::contracts::assert_shape_contract`).
///
/// See [`crate::contracts::shape_contract`] for documentation
/// on the contract syntax.
/// See [`crate::contracts::ShapeContract::assert_shape`] for documentation on
/// the assertion api. See [`crate::contracts::run_periodically`] for
/// documentation on the periodic runner.
///
/// ### With a Contract Expression:
///
/// ```rust,no_run
/// use bunsen::contracts::assert_shape_contract_periodically;
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
///
/// assert_shape_contract_periodically!(
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"],
///   &shape,
///   &[("ws", 2)],
/// );
/// ```
/// ### With a pre-defined contract:
///
/// ```rust,no_run
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

/// Define and call
/// [`ShapeContract::unpack_shape`](`crate::contracts::ShapeContract::unpack_shape`)
/// on a static shape contract.
///
/// See [`crate::contracts::shape_contract`] for documentation on the contract
/// syntax. See [`crate::contracts::ShapeContract::unpack_shape`] for
/// documentation on the unpack api.
///
/// ### With a Contract Expression:
///
/// ```rust,no_run
/// use bunsen::contracts::unpack_shape_contract;
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
///
/// let [h, h_win, w, w_win, c] = unpack_shape_contract!(
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"],
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
/// ### With a pre-defined contract:
///
/// ```rust,no_run
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
/// ### Special: When you have no bindings:
///
/// This also works with pre-defined contracts.
///
/// ```rust,no_run
/// use bunsen::contracts::unpack_shape_contract;
///
/// let shape = [1, 2, 3, 4 * 2, 5 * 2, 3];
///
/// let [h, h_win, w, w_win, ws, c] = unpack_shape_contract!(
///   [..., "h" = "h_win" * "ws", "w" = "w_win" * "ws", "c"],
///   &shape,
///   &["h", "h_win", "w", "w_win", "ws", "c"],
/// );
/// assert_eq!(ws, 2);
/// assert_eq!(h, 8);
/// assert_eq!(h_win, 4);
/// assert_eq!(w, 10);
/// assert_eq!(w_win, 5);
/// assert_eq!(c, 3);
/// ```
///
/// ### Special: When the keys are the expression:
///
/// ```rust,no_run
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
