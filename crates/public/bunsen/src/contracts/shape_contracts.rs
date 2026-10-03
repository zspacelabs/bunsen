//! Shape contracts: dimension matchers, matching, and failure messages.

use alloc::{
    format,
    string::{
        String,
        ToString,
    },
    vec,
    vec::Vec,
};
use core::{
    fmt::{
        Display,
        Formatter,
    },
    panic::Location,
};

use crate::contracts::{
    StackEnvironment,
    expressions::{
        DimExpr,
        ExprDisplayAdapter,
        MatchResult,
    },
    shape_view::ShapeView,
};

/// One term of a shape pattern: `_`, `...`, or an expression, each with an
/// optional label.
///
/// A [`ShapeContract`] holds one per pattern term, and
/// [`shape_contract!`](crate::contracts::shape_contract!) builds them. A
/// `label_id` is a position in the contract's
/// [`index`](ShapeContract::index). [Matching](ShapeContract#matching)
/// describes how each variant matches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimMatcher<'a> {
    /// Matches one dimension of any size (`_`).
    Any {
        /// An optional label for the matcher.
        label_id: Option<usize>,
    },

    /// Matches zero or more dimensions (`...`).
    ///
    /// It can't be labelled, because it has no single size to bind:
    /// `label_id` must be `None`, or [`ShapeContract::new`] panics.
    /// [`shape_contract!`](crate::contracts::shape_contract!) rejects the label
    /// at compile time.
    Ellipsis {
        /// Must be `None`.
        label_id: Option<usize>,
    },

    /// Matches one dimension whose size equals `expr`.
    Expr {
        /// An optional label for the matcher.
        label_id: Option<usize>,

        /// The dimension expression that must match a specific value.
        expr: DimExpr<'a>,
    },
}

impl<'a> DimMatcher<'a> {
    /// Creates a new `DimMatcher` that matches any dimension size.
    pub const fn any() -> Self {
        DimMatcher::Any { label_id: None }
    }

    /// Creates a new `DimMatcher` that matches a variable number of dimensions
    /// (ellipsis).
    pub const fn ellipsis() -> Self {
        DimMatcher::Ellipsis { label_id: None }
    }

    /// Creates a new `DimMatcher` from a dimension expression.
    ///
    /// # Arguments
    ///
    /// - `expr`: a dimension expression that must match a specific value.
    ///
    /// # Returns
    ///
    /// A new `DimMatcher` that matches the given expression.
    pub const fn expr(expr: DimExpr<'a>) -> Self {
        DimMatcher::Expr {
            label_id: None,
            expr,
        }
    }

    /// Returns the label of the matcher, if any.
    pub const fn label_id(&self) -> Option<usize> {
        match self {
            DimMatcher::Any { label_id } => *label_id,
            DimMatcher::Ellipsis { label_id } => *label_id,
            DimMatcher::Expr { label_id, .. } => *label_id,
        }
    }

    /// Attach a label to the matcher.
    ///
    /// [`ShapeContract::new`] rejects a labelled [`DimMatcher::Ellipsis`].
    ///
    /// # Arguments
    ///
    /// - `label_id`: an optional label to attach to the matcher.
    ///
    /// # Returns
    ///
    /// A new `DimMatcher` with the label attached.
    pub const fn with_label_id(
        self,
        label_id: Option<usize>,
    ) -> Self {
        match self {
            DimMatcher::Any { .. } => DimMatcher::Any { label_id },
            DimMatcher::Ellipsis { .. } => DimMatcher::Ellipsis { label_id },
            DimMatcher::Expr { expr, .. } => DimMatcher::Expr { label_id, expr },
        }
    }
}

/// Formats a [`DimMatcher`] with the names in a contract's index, as failure
/// messages print it: `height=(h_wins*window)`.
pub struct MatcherDisplayAdapter<'a> {
    index: &'a [&'a str],
    matcher: &'a DimMatcher<'a>,
}

impl<'a> Display for MatcherDisplayAdapter<'a> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> core::fmt::Result {
        if let Some(label_id) = self.matcher.label_id() {
            write!(f, "{}=", self.index[label_id])?;
        }
        match self.matcher {
            DimMatcher::Any { .. } => write!(f, "_"),
            DimMatcher::Ellipsis { .. } => write!(f, "..."),
            DimMatcher::Expr { expr, .. } => write!(
                f,
                "{}",
                ExprDisplayAdapter {
                    index: self.index,
                    expr
                }
            ),
        }
    }
}

/// A compiled shape pattern: checks a shape, and solves for the names it
/// doesn't know yet.
///
/// Build one with [`shape_contract!`](crate::contracts::shape_contract!),
/// usually as a `static`: directly, through
/// [`define_shape_contract!`](crate::contracts::define_shape_contract!), or
/// implicitly inside the check macros. A contract holds one [`DimMatcher`]
/// per pattern term, and an [`index`](Self::index) of every name in the
/// pattern; labels, and the params inside each [`DimExpr`], refer to names by
/// their position in it.
///
/// Every check takes the shape as anything that converts to a [`ShapeView`],
/// and the values the caller already knows as a [`StackEnvironment`]:
///
/// - [`assert_shape`](Self::assert_shape) and
///   [`try_assert_shape`](Self::try_assert_shape) check the shape;
/// - [`unpack_shape`](Self::unpack_shape) and
///   [`try_unpack_shape`](Self::try_unpack_shape) check it, then return the
///   values of chosen names.
///
/// The `try_` forms return the failure message as `Err`; the others panic
/// with it.
///
/// ```rust
/// use bunsen::contracts::{
///     ShapeContract,
///     shape_contract,
/// };
///
/// static CONTRACT: ShapeContract = shape_contract![
///     ...,
///     "height" = "h_wins" * "window",
///     "width" = "w_wins" * "window",
///     "channels",
/// ];
///
/// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
///
/// let [h_wins, w_wins] =
///     CONTRACT.unpack_shape(&shape, &["h_wins", "w_wins"], &[("window", 8)]);
/// assert_eq!([h_wins, w_wins], [2, 3]);
/// ```
///
/// # Matching
///
/// A check walks the shape once, left to right. If the pattern has a `...`,
/// it takes however many dimensions the other terms leave over, and the
/// shape's rank must be at least the number of other terms. Without one, the
/// rank must equal the number of terms.
///
/// Each other dimension is matched against its term, starting from the
/// caller's bindings:
///
/// 1. A label is handled first. An unbound label is bound to the dimension's
///    size; a bound one must equal it.
/// 2. `_` matches any size. Unless labelled, it binds nothing.
/// 3. An expression is evaluated with the names bound so far. If they are all
///    bound, its value must equal the size. If exactly one occurrence of a
///    param is unbound, the expression is solved for it, and the solution is
///    bound; it must be a non-negative integer, and the only one. More than one
///    unbound occurrence fails: `"a" * "a"` counts as two, while `"a" ^ 2` is
///    one. A product whose bound factors are 0 can't be solved: if `b` is 0,
///    every `t` makes `"b" * "t"` equal 0, so the caller must bind `t`.
///
/// A name bound at one dimension is bound for every dimension after it, so
/// order matters:
///
/// ```rust
/// use bunsen::contracts::{
///     ShapeContract,
///     shape_contract,
/// };
///
/// // Dim 0 binds `a`, so dim 1 has one unknown.
/// static FORWARD: ShapeContract = shape_contract!["a", "a" * "b"];
/// assert_eq!(FORWARD.unpack_shape(&[4, 12], &["a", "b"], &[]), [4, 3]);
///
/// // Dim 0 has two unknowns, unless the caller binds one of them.
/// static BACKWARD: ShapeContract = shape_contract!["a" * "b", "a"];
/// let err = BACKWARD
///     .try_unpack_shape(&[12, 4], &["a", "b"], &[])
///     .unwrap_err();
/// assert!(err.contains("Too many unbound params."));
/// assert_eq!(
///     BACKWARD.unpack_shape(&[12, 4], &["a", "b"], &[("b", 3)]),
///     [4, 3],
/// );
/// ```
///
/// Labels bind before their expression is checked, and only integer solutions
/// match:
///
/// ```rust
/// use bunsen::contracts::{
///     ShapeContract,
///     shape_contract,
/// };
///
/// // The label binds `n` at dim 0; dim 1 checks it.
/// static PAIRS: ShapeContract = shape_contract!["n" = 2 * "k", "n"];
/// assert_eq!(PAIRS.unpack_shape(&[6, 6], &["n", "k"], &[]), [6, 3]);
/// assert!(PAIRS.try_assert_shape(&[6, 5], &[]).is_err());
///
/// // 7 is odd, so `2 * "k"` has no integer solution.
/// let err = PAIRS.try_assert_shape(&[7, 7], &[]).unwrap_err();
/// assert!(err.contains("No integer solution."));
///
/// // A labelled `_` binds its size.
/// static SQUARE: ShapeContract = shape_contract!["n" = _, "n"];
/// assert_eq!(SQUARE.unpack_shape(&[5, 5], &["n"], &[]), [5]);
/// ```
///
/// # Error messages
///
/// A failed match produces this message. The `try_` methods return it as
/// `Err`; the other methods and the macros panic with it:
///
/// ```text
/// at src/model.rs:42: Shape Error
///   8 !~ height=(h_wins*window) :: No integer solution.
/// Actual:
///   [1, 2, 3, 8, 12, 3]
/// Contract:
///   [..., height=(h_wins*window), width=(w_wins*window), color]
/// Bindings:
///   {"color": 3, "height": 8, "window": 5}
/// ```
///
/// - `at file:line` is the code that called the check (the methods are
///   `#[track_caller]`); for the macros, it is the macro call.
/// - The second line is `size !~ term :: reason` for the first dimension that
///   failed. The term prints with its label, and with every compound expression
///   in parentheses.
/// - `Actual:` is the shape, and `Contract:` is the whole pattern.
/// - `Bindings:` lists every name bound when the match stopped, in
///   [`index`](Self::index) order, which is alphabetical. It includes the
///   caller's bindings and every value bound during the match, up to and
///   including the failing term's label (`"height": 8` above).
///
/// The reason is one of:
///
/// - `Value MissMatch.` (sic): a bound label, or an expression with every name
///   bound, doesn't equal the size;
/// - `No integer solution.`: the unknown has no integer solution (for example,
///   `2*"k"` against 7), or a `^` term's size has no integer root, even when
///   its base is bound;
/// - `No unique solution.`: every value of the unknown solves the term, as in
///   `"b"*"t"` with `b = 0` against 0;
/// - `Negative solution.`: the unknown solves to a negative value, which is no
///   size, as `"b"` does in `["a", "a"+"b"]` against `[5, 3]`;
/// - `Too many unbound params.`: the term still has more than one unknown.
///
/// A rank mismatch replaces the second line with
/// `Shape rank R != pattern dim count N` (no `...`) or
/// `Shape rank R < non-ellipsis pattern term count N`.
///
/// Three failures don't use this format. A binding whose name is not in the
/// pattern fails with `The key "k" is not indexed in the contract:`, then the
/// pattern, and no location; the `try_` methods return it as `Err`. An unpack
/// key that is not in the pattern panics with the same message, even from
/// [`try_unpack_shape`](Self::try_unpack_shape). A shape of `i32`s with a
/// negative size panics with `Shape [-1, 3] has a negative size` (for that
/// shape) while it converts to a [`ShapeView`], before any matching, even
/// from a `try_` method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeContract<'a> {
    /// Every name in the pattern, params and labels, sorted.
    ///
    /// Labels and [`DimExpr::Param`] ids are positions in this slice, and
    /// failure messages list bindings in its order.
    pub index: &'a [&'a str],

    /// One matcher per pattern term, in pattern order.
    pub terms: &'a [DimMatcher<'a>],

    /// The position of the `...` term in `terms`, if any.
    pub ellipsis_pos: Option<usize>,
}

impl Display for ShapeContract<'_> {
    fn fmt(
        &self,
        f: &mut Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "[")?;
        for (idx, term) in self.terms.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{}",
                MatcherDisplayAdapter {
                    index: self.index,
                    matcher: term
                }
            )?;
        }
        write!(f, "]")
    }
}

impl<'a> ShapeContract<'a> {
    /// Creates a contract from its name index and terms.
    ///
    /// Prefer [`shape_contract!`](crate::contracts::shape_contract!), which
    /// builds both arguments from a pattern.
    ///
    /// # Arguments
    ///
    /// - `index`: every name in the pattern; labels and params refer to names
    ///   by position.
    /// - `terms`: one [`DimMatcher`] per pattern term.
    ///
    /// # Panics
    ///
    /// If `terms` holds more than one [`DimMatcher::Ellipsis`], or a labelled
    /// one. When the contract is a `static`, this is a compile error.
    ///
    /// # Examples
    ///
    /// ```
    /// use bunsen::contracts::{
    ///     ShapeContract,
    ///     shape_contract,
    /// };
    ///
    /// static CONTRACT: ShapeContract = shape_contract![
    ///    ...,
    ///    "height" = "h_wins" * "window",
    ///    "width" = "w_wins" * "window",
    ///    "channels",
    /// ];
    /// ```
    pub const fn new(
        index: &'a [&'a str],
        terms: &'a [DimMatcher<'a>],
    ) -> Self {
        let mut i = 0;
        let mut ellipsis_pos: Option<usize> = None;

        while i < terms.len() {
            if let DimMatcher::Ellipsis { label_id } = &terms[i] {
                if label_id.is_some() {
                    panic!("Labelled ellipsis in pattern");
                }
                match ellipsis_pos {
                    Some(_) => panic!("Multiple ellipses in pattern"),
                    None => ellipsis_pos = Some(i),
                }
            }
            i += 1;
        }

        ShapeContract {
            index,
            terms,
            ellipsis_pos,
        }
    }

    /// Returns the position of `key` in [`index`](Self::index), or `None` if
    /// the pattern doesn't use it.
    pub fn maybe_key_to_index(
        &self,
        key: &str,
    ) -> Option<usize> {
        self.index.iter().position(|&s| s == key)
    }

    /// Checks that the shape matches the pattern, and panics if it doesn't.
    ///
    /// # Arguments
    ///
    /// - `shape`: the shape to match; see [`ShapeView`] for the accepted forms.
    /// - `env`: the names the caller already knows, as `(name, value)` pairs.
    ///
    /// # Panics
    ///
    /// If the shape doesn't match, with the message described under
    /// [Error messages](Self#error-messages), located at the caller. Also if
    /// `env` binds a name that is not in the pattern, or if an `i32` shape has
    /// a negative size ([`ShapeView`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bunsen::contracts::{
    ///     ShapeContract,
    ///     run_periodically,
    ///     shape_contract,
    /// };
    ///
    /// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
    ///
    /// // Run under backoff amortization.
    /// run_periodically! {{
    ///     // Statically allocated contract.
    ///     static CONTRACT : ShapeContract = shape_contract![
    ///        ...,
    ///        "height" = "h_wins" * "window",
    ///        "width" = "w_wins" * "window",
    ///        "channels",
    ///     ];
    ///
    ///     // Assert the shape, given the bindings.
    ///     CONTRACT.assert_shape(
    ///         &shape,
    ///         &[("h_wins", 2), ("w_wins", 3), ("channels", 4)]
    ///     );
    /// }}
    /// ```
    #[track_caller]
    pub fn assert_shape<'b, S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
    ) where
        S: Into<ShapeView<'b>>,
    {
        let shape = shape.into();
        match self._loc_try_assert_shape(&shape, env, Location::caller()) {
            Ok(()) => (),
            Err(msg) => panic!("{}", msg),
        }
    }

    /// Checks that the shape matches the pattern.
    ///
    /// # Arguments
    ///
    /// - `shape`: the shape to match; see [`ShapeView`] for the accepted forms.
    /// - `env`: the names the caller already knows, as `(name, value)` pairs.
    ///
    /// # Errors
    ///
    /// Returns `Err` with the message described under
    /// [Error messages](Self#error-messages) if the shape doesn't match, and
    /// with the unknown-key message if `env` binds a name that is not in the
    /// pattern.
    ///
    /// # Panics
    ///
    /// Even though this is the `try_` form: if an `i32` shape has a negative
    /// size ([`ShapeView`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bunsen::contracts::{
    ///     ShapeContract,
    ///     run_periodically,
    ///     shape_contract,
    /// };
    ///
    /// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
    ///
    /// // Statically allocated contract.
    /// static CONTRACT: ShapeContract = shape_contract![
    ///    ...,
    ///    "height" = "h_wins" * "window",
    ///    "width" = "w_wins" * "window",
    ///    "channels",
    /// ];
    ///
    /// // Assert the shape, given the bindings; or throw.
    /// CONTRACT
    ///     .try_assert_shape(
    ///         &shape,
    ///         &[("h_wins", 2), ("w_wins", 3), ("channels", 4)],
    ///     )
    ///     .unwrap();
    /// ```
    #[track_caller]
    pub fn try_assert_shape<'b, S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
    ) -> Result<(), String>
    where
        S: Into<ShapeView<'b>>,
    {
        let sv = shape.into();
        self._loc_try_assert_shape(&sv, env, Location::caller())
    }

    fn _key_index(
        &'a self,
        env: StackEnvironment<'a>,
    ) -> Result<Vec<Option<isize>>, String> {
        let mut key_index: Vec<Option<isize>> = vec![None; self.index.len()];
        for (k, v) in env.iter() {
            let v = *v as isize;
            match self.maybe_key_to_index(k) {
                Some(param_id) => key_index[param_id] = Some(v),
                None => {
                    return Err(
                        format!("The key \"{k}\" is not indexed in the contract:\n{self}")
                            .to_string(),
                    );
                }
            }
        }
        Ok(key_index)
    }

    fn _loc_try_assert_shape(
        &'a self,
        shape: &ShapeView,
        env: StackEnvironment<'a>,
        loc: &Location<'a>,
    ) -> Result<(), String> {
        let mut key_index = self._key_index(env)?;
        self.format_resolve(shape, key_index.as_mut_slice(), loc)
    }

    /// Checks that the shape matches the pattern, then returns the values of
    /// `K` names; panics if it doesn't match.
    ///
    /// ## Generics
    ///
    /// - `K`: the length of the `keys` array.
    ///
    /// # Arguments
    ///
    /// - `shape`: the shape to match; see [`ShapeView`] for the accepted forms.
    /// - `keys`: the names to return. Each must be in the pattern; its value
    ///   may come from `env`, a label, or the solver.
    /// - `env`: the names the caller already knows, as `(name, value)` pairs.
    ///
    /// # Returns
    ///
    /// The values of `keys`, in key order.
    ///
    /// # Panics
    ///
    /// If the shape doesn't match, with the message described under
    /// [Error messages](Self#error-messages), located at the caller. Also if
    /// a key or a binding names something that is not in the pattern, or if
    /// an `i32` shape has a negative size ([`ShapeView`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bunsen::contracts::{
    ///     ShapeContract,
    ///     run_periodically,
    ///     shape_contract,
    /// };
    ///
    /// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
    ///
    /// // Statically allocated contract.
    /// static CONTRACT: ShapeContract = shape_contract![
    ///    ...,
    ///    "height" = "h_wins" * "window",
    ///    "width" = "w_wins" * "window",
    ///    "channels",
    /// ];
    ///
    /// // Unpack the shape, given the bindings.
    /// let [h, w, c] = CONTRACT.unpack_shape(
    ///     &shape,
    ///     &["h_wins", "w_wins", "channels"],
    ///     &[("window", 8)],
    /// );
    /// assert_eq!(h, 2);
    /// assert_eq!(w, 3);
    /// assert_eq!(c, 4);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn unpack_shape<'b, S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> [usize; K]
    where
        S: Into<ShapeView<'b>>,
    {
        let sv: ShapeView = shape.into();
        self._loc_unpack_shape(&sv, keys, env, Location::caller())
    }

    fn _loc_unpack_shape<const K: usize>(
        &'a self,
        shape: &ShapeView,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
        loc: &Location<'a>,
    ) -> [usize; K] {
        match self._loc_try_unpack_shape(shape, keys, env, loc) {
            Ok(values) => values,
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Checks that the shape matches the pattern, then returns the values of
    /// `K` names.
    ///
    /// ## Generics
    ///
    /// - `K`: the length of the `keys` array.
    ///
    /// # Arguments
    ///
    /// - `shape`: the shape to match; see [`ShapeView`] for the accepted forms.
    /// - `keys`: the names to return. Each must be in the pattern; its value
    ///   may come from `env`, a label, or the solver.
    /// - `env`: the names the caller already knows, as `(name, value)` pairs.
    ///
    /// # Returns
    ///
    /// The values of `keys`, in key order.
    ///
    /// # Errors
    ///
    /// As [`try_assert_shape`](Self::try_assert_shape): `Err` with the message
    /// described under [Error messages](Self#error-messages) if the shape
    /// doesn't match, and with the unknown-key message if `env` binds a name
    /// that is not in the pattern.
    ///
    /// # Panics
    ///
    /// Even though this is the `try_` form: if a key is not in the pattern, and
    /// if an `i32` shape has a negative size ([`ShapeView`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use bunsen::contracts::{
    ///     ShapeContract,
    ///     run_periodically,
    ///     shape_contract,
    /// };
    ///
    /// let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
    ///
    /// // Statically allocated contract.
    /// static CONTRACT: ShapeContract = shape_contract![
    ///    ...,
    ///    "height" = "h_wins" * "window",
    ///    "width" = "w_wins" * "window",
    ///    "channels",
    /// ];
    ///
    /// // Unpack the shape, given the bindings; or throw.
    /// let [h, w, c] = CONTRACT
    ///     .try_unpack_shape(
    ///         &shape,
    ///         &["h_wins", "w_wins", "channels"],
    ///         &[("window", 8)],
    ///     )
    ///     .unwrap();
    /// assert_eq!(h, 2);
    /// assert_eq!(w, 3);
    /// assert_eq!(c, 4);
    /// ```
    #[track_caller]
    pub fn try_unpack_shape<'b, S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> Result<[usize; K], String>
    where
        S: Into<ShapeView<'b>>,
    {
        let sv: ShapeView = shape.into();
        self._loc_try_unpack_shape(&sv, keys, env, Location::caller())
    }

    fn _loc_try_unpack_shape<const K: usize>(
        &'a self,
        shape: &ShapeView,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
        loc: &Location<'a>,
    ) -> Result<[usize; K], String> {
        let selection = self.expect_keys_to_selection(keys);

        let mut key_index = self._key_index(env)?;

        let selected: [isize; K] =
            self._loc_try_select(shape, &selection, key_index.as_mut_slice(), loc)?;

        let result: [usize; K] = selected
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap();

        Ok(result)
    }

    /// Converts unpack keys to their positions in [`index`](Self::index).
    ///
    /// # Panics
    ///
    /// If a key is not in the index.
    pub fn expect_keys_to_selection<const D: usize>(
        &'a self,
        keys: &[&'a str; D],
    ) -> [usize; D] {
        let mut selection = [0; D];
        for (i, key) in keys.iter().enumerate() {
            match self.maybe_key_to_index(key) {
                Some(param_id) => selection[i] = param_id,
                None => panic!("The key \"{key}\" is not indexed in the contract:\n{self}"),
            }
        }
        selection
    }

    fn _loc_try_select<const K: usize>(
        &'a self,
        shape: &ShapeView,
        selection: &[usize; K],
        env: &mut [Option<isize>],
        loc: &Location<'a>,
    ) -> Result<[isize; K], String> {
        let num_slots = self.index.len();
        assert_eq!(env.len(), num_slots);

        self.format_resolve(shape, env, loc)?;

        let mut out = [0; K];
        for (i, &k) in selection.iter().enumerate() {
            out[i] = env[k].unwrap();
        }
        Ok(out)
    }

    /// Resolve the match for the shape against the pattern.
    ///
    /// # Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the mutable environment to bind parameters.
    /// - `location`: the location reference from ``#[track_caller]``.
    ///
    /// # Returns
    ///
    /// - `Ok(())`: if the shape matches the pattern; will update the `env`.
    /// - `Err(String)`: if the shape does not match the pattern, with the
    ///   message described under [Error messages](Self#error-messages).
    pub(crate) fn format_resolve(
        &'a self,
        shape: &ShapeView,
        env: &mut [Option<isize>],
        location: &Location,
    ) -> Result<(), String> {
        match self._resolve(shape.as_ref(), env) {
            Ok(()) => Ok(()),
            Err(msg) => Err(format!(
                "at {file}:{line}: Shape Error\n  {msg}\nActual:\n  {shape:?}\nContract:\n  {self}\nBindings:\n  {{{}}}",
                self.index
                    .iter()
                    .zip(env.iter())
                    .filter(|(_, v)| v.is_some())
                    .map(|(k, v)| format!("\"{}\": {}", *k, v.unwrap()))
                    .collect::<Vec<_>>()
                    .join(", "),
                file = location.file(),
                line = location.line(),
                shape = shape.as_ref(),
            )),
        }
    }

    /// Low-level resolver.
    #[doc(hidden)]
    pub fn _resolve(
        &'a self,
        shape: &[usize],
        env: &mut [Option<isize>],
    ) -> Result<(), String> {
        let rank = shape.len();

        let fail_at = |shape_idx: usize, term_idx: usize, msg: &str| -> String {
            format!(
                "{} !~ {} :: {msg}",
                shape[shape_idx],
                MatcherDisplayAdapter {
                    index: self.index,
                    matcher: &self.terms[term_idx]
                }
            )
        };

        let (e_start, e_size) = match self.try_ellipsis_split(rank) {
            Ok((e_start, e_size)) => (e_start, e_size),
            Err(msg) => return Err(msg),
        };

        for (shape_idx, &dim_size) in shape.iter().enumerate() {
            let dim_size = dim_size as isize;

            let term_idx = if shape_idx < e_start {
                shape_idx
            } else if shape_idx < (e_start + e_size) {
                continue;
            } else {
                shape_idx + 1 - e_size
            };

            let matcher = &self.terms[term_idx];
            if let Some(label_id) = matcher.label_id() {
                match env[label_id] {
                    Some(value) => {
                        if value != dim_size {
                            return Err(fail_at(shape_idx, term_idx, "Value MissMatch."));
                        }
                    }
                    None => {
                        env[label_id] = Some(dim_size);
                    }
                }
            }

            let expr = match matcher {
                DimMatcher::Any { .. } => continue,
                DimMatcher::Expr { expr, .. } => expr,
                DimMatcher::Ellipsis { .. } => {
                    unreachable!("Ellipsis should have been handled before")
                }
            };

            match expr.try_match(dim_size, env) {
                Ok(MatchResult::Match) => continue,
                Ok(MatchResult::Conflict) => {
                    return Err(fail_at(shape_idx, term_idx, "Value MissMatch."));
                }
                Ok(MatchResult::ParamConstraint { id, value }) => {
                    if value < 0 {
                        // Every name is a size.
                        return Err(fail_at(shape_idx, term_idx, "Negative solution."));
                    }
                    env[id] = Some(value);
                }
                Err(msg) => return Err(fail_at(shape_idx, term_idx, msg)),
            }
        }

        Ok(())
    }

    /// Checks if the pattern has an ellipsis.
    ///
    /// # Arguments
    ///
    /// - `rank`: the number of dims of the shape to match.
    ///
    /// # Returns
    ///
    /// - `Ok((usize, usize))`: the position of the ellipsis and the number of
    ///   dimensions it matches.
    /// - `Err(String)`: an error message if the pattern does not match the
    ///   expected size.
    fn try_ellipsis_split(
        &self,
        rank: usize,
    ) -> Result<(usize, usize), String> {
        let k = self.terms.len();
        match self.ellipsis_pos {
            None => {
                if rank != k {
                    Err(format!("Shape rank {rank} != pattern dim count {k}",))
                } else {
                    Ok((k, 0))
                }
            }
            Some(pos) => {
                let non_ellipsis_terms = k - 1;
                if rank < non_ellipsis_terms {
                    return Err(format!(
                        "Shape rank {rank} < non-ellipsis pattern term count {non_ellipsis_terms}",
                    ));
                }
                Ok((pos, rank - non_ellipsis_terms))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use super::*;
    use crate::contracts::{
        DimExpr,
        assert_shape_contract,
        shape_contract,
    };

    static WINDOWS: ShapeContract = shape_contract![
        ...,
        "height" = "h_wins" * "window",
        "width" = "w_wins" * "window",
        "color",
    ];

    /// The second line of a failure message: `size !~ term :: reason`.
    fn reason(msg: String) -> String {
        msg.lines().nth(1).unwrap().trim().to_string()
    }

    #[test]
    fn test_error_message_format() {
        let err = WINDOWS
            .try_unpack_shape(
                &[1usize, 2, 3, 8, 12, 3],
                &["h_wins", "w_wins"],
                &[("window", 5), ("color", 3)],
            )
            .unwrap_err();

        let (location, body) = err.split_once('\n').unwrap();
        assert!(
            location.starts_with(&format!("at {}:", file!())),
            "{location}"
        );
        assert!(location.ends_with(": Shape Error"), "{location}");
        assert_eq!(
            body,
            indoc! {r#"
                  8 !~ height=(h_wins*window) :: No integer solution.
                Actual:
                  [1, 2, 3, 8, 12, 3]
                Contract:
                  [..., height=(h_wins*window), width=(w_wins*window), color]
                Bindings:
                  {"color": 3, "height": 8, "window": 5}"#
            },
        );
    }

    #[test]
    fn test_error_reasons() {
        static PAIRS: ShapeContract = shape_contract!["n" = 2 * "k", "n"];
        static BACKWARD: ShapeContract = shape_contract!["a" * "b", "a"];

        fn fail(
            contract: &ShapeContract,
            shape: &[usize],
            env: StackEnvironment,
        ) -> String {
            reason(contract.try_assert_shape(shape, env).unwrap_err())
        }

        assert_eq!(fail(&PAIRS, &[6, 5], &[]), "5 !~ n :: Value MissMatch.");
        assert_eq!(
            fail(&PAIRS, &[6, 6], &[("n", 4)]),
            "6 !~ n=(2*k) :: Value MissMatch."
        );
        assert_eq!(
            fail(&PAIRS, &[7, 7], &[]),
            "7 !~ n=(2*k) :: No integer solution."
        );
        assert_eq!(
            fail(&BACKWARD, &[12, 4], &[]),
            "12 !~ (a*b) :: Too many unbound params."
        );
        assert_eq!(
            fail(&PAIRS, &[6], &[]),
            "Shape rank 1 != pattern dim count 2"
        );
        assert_eq!(
            fail(&WINDOWS, &[8, 12], &[]),
            "Shape rank 2 < non-ellipsis pattern term count 3"
        );
    }

    #[test]
    fn test_unknown_binding_key() {
        let err = WINDOWS
            .try_assert_shape(&[8usize, 12, 3], &[("nope", 1)])
            .unwrap_err();
        assert_eq!(
            err,
            "The key \"nope\" is not indexed in the contract:\n\
             [..., height=(h_wins*window), width=(w_wins*window), color]"
        );
    }

    #[test]
    #[should_panic(expected = "The key \"nope\" is not indexed in the contract:")]
    fn test_unknown_unpack_key_panics_in_try() {
        let _ = WINDOWS.try_unpack_shape(&[8usize, 12, 3], &["nope"], &[]);
    }

    #[test]
    #[should_panic(expected = "Labelled ellipsis in pattern")]
    fn test_new_rejects_labelled_ellipsis() {
        let terms = [
            DimMatcher::ellipsis().with_label_id(Some(0)),
            DimMatcher::expr(DimExpr::Param { id: 1 }),
        ];
        let contract = ShapeContract::new(&["rest", "x"], &terms);

        // Without the check in `new`, `rest` is never bound, and unpacking it
        // panics, even from the `try_` form.
        let _ = contract.try_unpack_shape(&[2usize, 3, 4], &["rest"], &[]);
    }

    #[test]
    fn test_zero_factor_fails_without_panicking() {
        static CONTRACT: ShapeContract = shape_contract!["b", "b" * "t"];

        fn fail(shape: &[usize]) -> String {
            reason(CONTRACT.try_assert_shape(shape, &[]).unwrap_err())
        }

        // `b = 0`, so every `t` gives 0.
        assert_eq!(fail(&[0, 0]), "0 !~ (b*t) :: No unique solution.");
        // `b = 0`, so no `t` gives 3.
        assert_eq!(fail(&[0, 3]), "3 !~ (b*t) :: No integer solution.");

        // With `t` bound, there is nothing to solve.
        assert_eq!(
            CONTRACT.unpack_shape(&[0usize, 0], &["b", "t"], &[("t", 4)]),
            [0, 4]
        );
    }

    #[test]
    fn test_negative_solution_is_an_error() {
        static SUM: ShapeContract = shape_contract!["a", "a" + "b"];

        // `b = 3 - 5` is no size.
        let err = SUM
            .try_unpack_shape(&[5usize, 3], &["a", "b"], &[])
            .unwrap_err();
        assert_eq!(reason(err), "3 !~ (a+b) :: Negative solution.");
        assert_eq!(SUM.unpack_shape(&[5usize, 8], &["a", "b"], &[]), [5, 3]);

        // `(-x)^2 = 9` takes the root 3, so `x = -3`.
        static NEG_SQUARE: ShapeContract = shape_contract![-"x" ^ 2];
        let err = NEG_SQUARE
            .try_unpack_shape(&[9usize], &["x"], &[])
            .unwrap_err();
        assert_eq!(reason(err), "9 !~ ((-x)^2) :: Negative solution.");
    }

    #[test]
    fn test_macro_failure_location_is_the_call() {
        use std::panic::catch_unwind;

        let line = line!() + 1;
        let r = catch_unwind(|| assert_shape_contract!(["a"], &[2usize], &[("a", 3)]));

        let payload = r.unwrap_err();
        let msg = payload.downcast_ref::<String>().unwrap();
        assert!(
            msg.starts_with(&format!("at {}:{line}: Shape Error", file!())),
            "{msg}"
        );
    }

    #[test]
    fn test_unpack_shape() {
        static CONTRACT: ShapeContract = ShapeContract::new(
            &["b", "h", "w", "p", "z", "c"],
            &[
                DimMatcher::any(),
                DimMatcher::expr(DimExpr::Param { id: 0 }),
                DimMatcher::ellipsis(),
                DimMatcher::expr(DimExpr::Prod {
                    children: &[DimExpr::Param { id: 1 }, DimExpr::Param { id: 3 }],
                }),
                DimMatcher::expr(DimExpr::Prod {
                    children: &[DimExpr::Param { id: 2 }, DimExpr::Param { id: 3 }],
                }),
                DimMatcher::expr(DimExpr::Pow {
                    base: &DimExpr::Param { id: 4 },
                    exp: 3,
                }),
                DimMatcher::expr(DimExpr::Param { id: 5 }),
            ],
        );

        let b = 2;
        let h = 3;
        let w = 2;
        let p = 4;
        let c = 5;
        let z = 4;

        let shape = [12, b, 1, 2, 3, h * p, w * p, z * z * z, c];
        let env = [("p", p), ("c", c)];

        CONTRACT.assert_shape(&shape, &env);

        let [u_b, u_h, u_w, u_z] = CONTRACT.unpack_shape(&shape, &["b", "h", "w", "z"], &env);

        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
        assert_eq!(u_w, w);
        assert_eq!(u_z, z);
    }

    #[test]
    fn test_shape_contract_macro_const() {
        use crate::contracts::shape_contract;

        static CONTRACT: ShapeContract = shape_contract!["b", 3, 2 * "h" + 1];

        let b = 2;
        let h = 5;

        let shape = [b, 3, 2 * h + 1];
        CONTRACT.assert_shape(&shape, &[]);

        let [u_b, u_h] = CONTRACT.unpack_shape(&shape, &["b", "h"], &[]);
        assert_eq!(u_b, b);
        assert_eq!(u_h, h);
    }
}
