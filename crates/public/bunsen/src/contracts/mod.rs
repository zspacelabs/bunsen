#![forbid(unsafe_code)]
#![deny(unused_must_use)]
#![warn(missing_docs)]
//! # Shape Contracts
//!
//! Runtime checks of tensor shapes against a small pattern language, for
//! [burn](https://burn.dev) tensors and plain shape arrays.
//!
//! A contract states what a function expects of a shape, in the spirit of
//! [Design by Contract](https://en.wikipedia.org/wiki/Design_by_contract).
//! One call checks the shape, solves for the dimension names the caller
//! doesn't know yet, and returns the ones it asks for:
//!
//! ```rust
//! use bunsen::contracts::unpack_shape_contract;
//!
//! let shape = [12, 3 * 4, 5 * 4, 3];
//!
//! let [b, h_wins, w_wins, c] = unpack_shape_contract!(
//!     [
//!         "batch",
//!         "height" = "h_wins" * "window_size",
//!         "width" = "w_wins" * "window_size",
//!         "channels"
//!     ],
//!     &shape,
//!     &["batch", "h_wins", "w_wins", "channels"],
//!     &[("window_size", 4)],
//! );
//!
//! assert_eq!([b, h_wins, w_wins, c], [12, 3, 5, 3]);
//! ```
//!
//! The goals:
//! - contracts are easy to read and write, and read like the shapes in the docs
//!   next to them;
//! - they are cheap enough to leave on in release builds;
//! - a failure names the dimension that broke, the pattern term, and the values
//!   in play.
//!
//! ## How the types relate
//!
//! - [`shape_contract!`] parses a pattern at compile time. It expands to a
//!   [`ShapeContract`] built from `const fn`s, so a contract is usually a
//!   `static`.
//! - A [`ShapeContract`] holds one [`DimMatcher`] per pattern term, plus an
//!   index of every name the pattern uses.
//! - A [`DimMatcher`] matches one dimension of any size ([`DimMatcher::Any`],
//!   `_`), a run of dimensions ([`DimMatcher::Ellipsis`], `...`), or one
//!   dimension against an expression ([`DimMatcher::Expr`]). Any of them can
//!   carry a label.
//! - A [`DimExpr`] is an integer expression over named params and constants.
//!   Matching it against a dimension size checks it, or solves it for one
//!   unknown param.
//! - A check takes the shape as a [`ShapeView`], and the values the caller
//!   already knows as [`StackEnvironment`] bindings (`&[(&str, usize)]`).
//!
//! [`shape_contract!`] documents the pattern language. [`ShapeContract`]
//! documents how a shape is matched (left to right, one unknown per
//! dimension) and what a failure reports.
//!
//! ## Macros and methods
//!
//! Most code calls one of three macros. Each defines a `static` contract at
//! the call site (or names one) and calls one method on it:
//!
//! - [`unpack_shape_contract!`] calls [`ShapeContract::unpack_shape`]: check
//!   the shape, then return the values of chosen names;
//! - [`assert_shape_contract!`] calls [`ShapeContract::assert_shape`]: check
//!   only;
//! - [`assert_shape_contract_periodically!`] runs the same check through
//!   [`run_periodically!`], on a schedule that thins out to one call in 1000.
//!
//! All three panic on a mismatch. To get the failure as a value, define the
//! contract with [`shape_contract!`] or [`define_shape_contract!`] and call
//! [`ShapeContract::try_assert_shape`] or [`ShapeContract::try_unpack_shape`].
//!
//! ## Contracts read like the docs
//!
//! STYLE.md writes a tensor shape in rustdoc as one code span in square
//! brackets, such as `[batch, h_wins*size, w_wins*size, channels]`. Write the
//! contract so that it reads the same: one term per dimension, the same
//! names, the same arithmetic. A reader can then check the docs against the
//! code at a glance, and the contract enforces what the docs promise:
//!
//! ```rust
//! use bunsen::contracts::{
//!     assert_shape_contract_periodically,
//!     unpack_shape_contract,
//! };
//! use burn::prelude::{
//!     Backend,
//!     Tensor,
//! };
//! # use bunsen::support::testing::CpuBackend;
//!
//! /// Splits an image into square windows.
//! ///
//! /// # Arguments
//! ///
//! /// - `x`: a `[batch, h_wins*size, w_wins*size, channels]` input tensor.
//! /// - `size`: the window size.
//! ///
//! /// # Returns
//! ///
//! /// A `[batch*h_wins*w_wins, size, size, channels]` tensor of windows.
//! pub fn windows<B: Backend>(
//!     x: Tensor<B, 4>,
//!     size: usize,
//! ) -> Tensor<B, 4> {
//!     let [batch, h_wins, w_wins, channels] = unpack_shape_contract!(
//!         ["batch", "h_wins" * "size", "w_wins" * "size", "channels"],
//!         &x.dims(),
//!         &["batch", "h_wins", "w_wins", "channels"],
//!         &[("size", size)],
//!     );
//!
//!     let x = x
//!         .reshape([batch, h_wins, size, w_wins, size, channels])
//!         .swap_dims(2, 3)
//!         .reshape([batch * h_wins * w_wins, size, size, channels]);
//!
//!     assert_shape_contract_periodically!(
//!         ["batch" * "h_wins" * "w_wins", "size", "size", "channels"],
//!         &x.dims(),
//!         &[
//!             ("batch", batch),
//!             ("h_wins", h_wins),
//!             ("w_wins", w_wins),
//!             ("size", size),
//!             ("channels", channels),
//!         ],
//!     );
//!     x
//! }
//!
//! let x = Tensor::<CpuBackend, 4>::zeros([2, 6, 9, 5], &Default::default());
//! assert_eq!(windows(x, 3).dims(), [2 * 2 * 3, 3, 3, 5]);
//! ```
//!
//! ## Passing shapes
//!
//! Every check takes its shape as `S: Into<ShapeView>`. For a tensor, pass
//! `&x.dims()`: `dims()` returns a `[usize; D]` array for a tensor of any
//! rank `D`, and the check borrows it. Passing `&x` also works, but converts
//! the tensor's [`Shape`](burn::prelude::Shape) into a new `Vec<usize>` on
//! every call. [`ShapeView`] lists every accepted form and what each costs.
//!
//! ## Cost
//!
//! The pattern is parsed at compile time, and the macros keep the contract in
//! a `static`. A check is one pass over the shape's dimensions. It is not
//! allocation-free: every check allocates a small `Vec` with one slot per
//! name in the pattern, an unpack collects its result through another, some
//! [`ShapeView`] conversions allocate, and a failure formats its message.
//!
//! To measure it:
//!
//! ```text
//! cargo bench -p bunsen --bench contracts
//! ```
//!
//! Indicative numbers, from a release build on one development machine:
//! about 170 ns per `unpack_shape` or `assert_shape` call, matching a
//! 9-dimension shape against the 7-term pattern
//! `[_, "b", ..., "h"*"p", "w"*"p", "z"^3, "c"]`; and about 4.5 ns per call,
//! averaged, for the same `assert_shape` under [`run_periodically!`]. Rerun
//! the bench before relying on them.
//!
//! When a check still costs too much, sample it with
//! [`assert_shape_contract_periodically!`], or gate it with
//! `#[cfg(debug_assertions)]` so that release builds drop it, as
//! [`next_interior_3d`](crate::kits::sims::conway::ops::next_interior_3d)
//! does.
//!
//! ## Error messages
//!
//! A failed check names the dimension and the pattern term that failed, and
//! prints the shape, the pattern, and every bound value.
//! [`ShapeContract`](ShapeContract#error-messages) describes each part.
//!
//! ```rust
//! use bunsen::contracts::{
//!     ShapeContract,
//!     shape_contract,
//! };
//! use indoc::indoc;
//!
//! static CONTRACT: ShapeContract = shape_contract![
//!     ...,
//!     "height" = "h_wins" * "window",
//!     "width" = "w_wins" * "window",
//!     "color",
//! ];
//!
//! let shape = [1, 2, 3, 2 * 4, 3 * 4, 3];
//!
//! // Correct bindings: `window` divides both sizes.
//! let [h_wins, w_wins] = CONTRACT.unpack_shape(
//!     &shape,
//!     &["h_wins", "w_wins"],
//!     &[("window", 4), ("color", 3)],
//! );
//! assert_eq!([h_wins, w_wins], [2, 3]);
//!
//! // Wrong bindings: 5 does not divide 8.
//! let err = CONTRACT
//!     .try_unpack_shape(
//!         &shape,
//!         &["h_wins", "w_wins"],
//!         &[("window", 5), ("color", 3)],
//!     )
//!     .unwrap_err();
//!
//! // The first line is `at <file>:<line>: Shape Error`, naming the caller.
//! let (location, body) = err.split_once('\n').unwrap();
//! assert!(location.starts_with("at ") && location.ends_with(": Shape Error"));
//! assert_eq!(
//!     body,
//!     indoc! {r#"
//!           8 !~ height=(h_wins*window) :: No integer solution.
//!         Actual:
//!           [1, 2, 3, 8, 12, 3]
//!         Contract:
//!           [..., height=(h_wins*window), width=(w_wins*window), color]
//!         Bindings:
//!           {"color": 3, "height": 8, "window": 5}"#
//!     },
//! );
//! ```

mod macros;
#[doc(inline)]
pub use macros::{
    assert_shape_contract,
    assert_shape_contract_periodically,
    define_shape_contract,
    run_periodically,
    shape_contract,
    unpack_shape_contract,
};

mod shape_view;
pub use shape_view::*;

mod expressions;
pub use expressions::*;

mod bindings;
pub use bindings::*;

mod shape_contracts;
pub use shape_contracts::*;
