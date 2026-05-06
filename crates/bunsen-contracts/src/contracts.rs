//! # Shape Contracts.
//!
//! `bunsen-contracts` is built around the [`ShapeContract`] interface.
//! - A [`ShapeContract`] is a sequence of [`DimMatcher`]s.
//! - A [`DimMatcher`] matches one or more dimensions of a shape:
//!   - [`DimMatcher::Any`] matches any dimension size.
//!   - [`DimMatcher::Ellipsis`] matches a variable number of dimensions
//!     (ellipsis).
//!   - [`DimMatcher::Expr`] matches a dimension expression that must match a
//!     specific value.
//!
//! A [`ShapeContract`] should usually be constructed using the
//! [`crate::shape_contract`] macro.
//!
//! ## Example
//!
//! ```rust
//! use bunsen_contracts::{
//!     ShapeContract,
//!     shape_contract,
//! };
//!
//! static CONTRACT: ShapeContract = shape_contract![
//!    ...,
//!    "height" = "h_wins" * "window",
//!    "width" = "w_wins" * "window",
//!    "channels",
//! ];
//!
//! let shape = [1, 2, 3, 2 * 8, 3 * 8, 4];
//!
//! // Assert the shape, given the bindings.
//! let [h_wins, w_wins] =
//!     CONTRACT.unpack_shape(&shape, &["h_wins", "w_wins"], &[("window", 8)]);
//! assert_eq!(h_wins, 2);
//! assert_eq!(w_wins, 3);
//! ```

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

use crate::{
    StackEnvironment,
    expressions::{
        DimExpr,
        ExprDisplayAdapter,
        MatchResult,
    },
    shape_argument::ShapeArgument,
};

/// A term in a shape pattern.
///
/// Users should generally use [`crate::shape_contract`] to construct patterns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimMatcher<'a> {
    /// Matches any dimension size.
    Any {
        /// An optional label for the matcher.
        label_id: Option<usize>,
    },

    /// Matches a variable number of dimensions (ellipsis).
    Ellipsis {
        /// An optional label for the matcher.
        label_id: Option<usize>,
    },

    /// A dimension size expression that must match a specific value.
    Expr {
        /// An optional label for the matcher.
        label_id: Option<usize>,

        /// The dimension expression that must match a specific value.
        expr: DimExpr<'a>,
    },
}

impl<'a> DimMatcher<'a> {
    /// Create a new `DimMatcher` that matches any dimension size.
    pub const fn any() -> Self {
        DimMatcher::Any { label_id: None }
    }

    /// Create a new `DimMatcher` that matches a variable number of dimensions
    /// (ellipsis).
    pub const fn ellipsis() -> Self {
        DimMatcher::Ellipsis { label_id: None }
    }

    /// Create a new `DimMatcher` from a dimension expression.
    ///
    /// ## Arguments
    ///
    /// - `expr`: a dimension expression that must match a specific value.
    ///
    /// ## Returns
    ///
    /// A new `DimMatcher` that matches the given expression.
    pub const fn expr(expr: DimExpr<'a>) -> Self {
        DimMatcher::Expr {
            label_id: None,
            expr,
        }
    }

    /// Get the label of the matcher, if any.
    pub const fn label_id(&self) -> Option<usize> {
        match self {
            DimMatcher::Any { label_id } => *label_id,
            DimMatcher::Ellipsis { label_id } => *label_id,
            DimMatcher::Expr { label_id, .. } => *label_id,
        }
    }

    /// Attach a label to the matcher.
    ///
    /// ## Arguments
    ///
    /// - `label_id`: an optional label to attach to the matcher.
    ///
    /// ## Returns
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

/// Display Adapter to format `DimMatchers` with a `Index`.
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

/// A shape pattern, which is a sequence of terms that can match a shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeContract<'a> {
    /// The slot index of the contract.
    pub index: &'a [&'a str],

    /// The terms in the pattern.
    pub terms: &'a [DimMatcher<'a>],

    /// The position of the ellipsis in the pattern, if any.
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
    /// Create a new shape pattern from a slice of terms.
    ///
    /// ## Arguments
    ///
    /// - `terms`: a slice of `ShapePatternTerm` that defines the pattern.
    ///
    /// ## Returns
    ///
    /// A new `ShapePattern` instance.
    ///
    /// ## Macro Support
    ///
    /// Consider using the [`crate::shape_contract`] macro instead.
    ///
    /// ```
    /// use bunsen_contracts::{
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
            if matches!(terms[i], DimMatcher::Ellipsis { .. }) {
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

    /// Convert a key to an index.
    pub fn maybe_key_to_index(
        &self,
        key: &str,
    ) -> Option<usize> {
        self.index.iter().position(|&s| s == key)
    }

    /// Assert that the shape matches the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the params which are already bound.
    ///
    /// ## Panics
    ///
    /// If the shape does not match the pattern, or if there is a conflict in
    /// the bindings.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bunsen_contracts::{
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
    pub fn assert_shape<S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
    ) where
        S: ShapeArgument,
    {
        match self._loc_try_assert_shape(shape, env, Location::caller()) {
            Ok(()) => (),
            Err(msg) => panic!("{}", msg),
        }
    }

    /// Assert that the shape matches the pattern.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// - `Ok(())`: if the shape matches the pattern.
    /// - `Err(String)`: if the shape does not match the pattern, with an error
    ///   message.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bunsen_contracts::{
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
    pub fn try_assert_shape<S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
    ) -> Result<(), String>
    where
        S: ShapeArgument,
    {
        self._loc_try_assert_shape(shape, env, Location::caller())
    }

    fn _loc_try_assert_shape<S>(
        &'a self,
        shape: S,
        env: StackEnvironment<'a>,
        loc: &Location<'a>,
    ) -> Result<(), String>
    where
        S: ShapeArgument,
    {
        let mut scratch: Vec<Option<isize>> = vec![None; self.index.len()];
        for (k, v) in env.iter() {
            let v = *v as isize;
            match self.maybe_key_to_index(k) {
                Some(param_id) => scratch[param_id] = Some(v),
                None => {
                    return Err(
                        format!("The key \"{k}\" is not indexed in the contract:\n{self}")
                            .to_string(),
                    );
                }
            }
        }

        self.format_resolve(shape, scratch.as_mut_slice(), loc)
    }

    /// Match and unpack `K` keys from a shape pattern.
    ///
    /// Wraps `try_unpack_shape` and panics if the shape does not match.
    ///
    /// ## Generics
    ///
    /// - `K`: the length of the `keys` array.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// An `[usize; K]` of the unpacked `keys` values.
    ///
    /// ## Panics
    ///
    /// If the shape does not match the pattern, or if there is a conflict in
    /// the bindings.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bunsen_contracts::{
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
    pub fn unpack_shape<S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> [usize; K]
    where
        S: ShapeArgument,
    {
        self._loc_unpack_shape(shape, keys, env, Location::caller())
    }

    fn _loc_unpack_shape<S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
        loc: &Location<'a>,
    ) -> [usize; K]
    where
        S: ShapeArgument,
    {
        match self._loc_try_unpack_shape(shape, keys, env, loc) {
            Ok(values) => values,
            Err(msg) => panic!("{msg}"),
        }
    }

    /// Try and match and unpack `K` keys from a shape pattern.
    ///
    /// ## Generics
    ///
    /// - `K`: the length of the `keys` array.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `keys`: the bound keys to export.
    /// - `env`: the params which are already bound.
    ///
    /// ## Returns
    ///
    /// A `Result<[usize; K], String>` of the unpacked `keys` values.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bunsen_contracts::{
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
    pub fn try_unpack_shape<S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
    ) -> Result<[usize; K], String>
    where
        S: ShapeArgument,
    {
        self._loc_try_unpack_shape(shape, keys, env, Location::caller())
    }

    fn _loc_try_unpack_shape<S, const K: usize>(
        &'a self,
        shape: S,
        keys: &[&'a str; K],
        env: StackEnvironment<'a>,
        loc: &Location<'a>,
    ) -> Result<[usize; K], String>
    where
        S: ShapeArgument,
    {
        let selection = self.expect_keys_to_selection(keys);

        let mut scratch: Vec<Option<isize>> = vec![None; self.index.len()];
        for (k, v) in env.iter() {
            let v = *v as isize;
            match self.maybe_key_to_index(k) {
                Some(param_id) => scratch[param_id] = Some(v),
                None => {
                    return Err(
                        format!("The key \"{k}\" is not indexed in the contract:\n{self}")
                            .to_string(),
                    );
                }
            }
        }

        let selected: [isize; K] =
            self._loc_try_select(shape, &selection, scratch.as_mut_slice(), loc)?;

        let result: [usize; K] = selected
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap();

        Ok(result)
    }

    /// Convert a list of keys to a selection.
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

    fn _loc_try_select<S, const K: usize>(
        &'a self,
        shape: S,
        selection: &[usize; K],
        env: &mut [Option<isize>],
        loc: &Location<'a>,
    ) -> Result<[isize; K], String>
    where
        S: ShapeArgument,
    {
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
    /// ## Arguments
    ///
    /// - `shape`: the shape to match.
    /// - `env`: the mutable environment to bind parameters.
    /// - `location`: the location reference from ``#[track_caller]``.
    ///
    /// ## Returns
    ///
    /// - `Ok(())`: if the shape matches the pattern; will update the `env`.
    /// - `Err(&str)`: if the shape does not match the pattern, with an error
    ///   message.
    pub(crate) fn format_resolve<S>(
        &'a self,
        shape: S,
        env: &mut [Option<isize>],
        location: &Location,
    ) -> Result<(), String>
    where
        S: ShapeArgument,
    {
        let shape = shape.get_shape_vec();
        match self._resolve(&shape, env) {
            Ok(()) => Ok(()),
            Err(msg) => Err(format!(
                "at {}:{}: Shape Error\n  {msg}\nActual:\n  {shape:?}\nContract:\n  {self}\nBindings:\n  {{{}}}",
                location.file(),
                location.line(),
                self.index
                    .iter()
                    .zip(env.iter())
                    .filter(|(_, v)| v.is_some())
                    .map(|(k, v)| format!("\"{}\": {}", *k, v.unwrap()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }

    /// Low-level resolver.
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
                    env[id] = Some(value);
                }
                Err(msg) => return Err(fail_at(shape_idx, term_idx, msg)),
            }
        }

        Ok(())
    }

    /// Check if the pattern has an ellipsis.
    ///
    /// ## Arguments
    ///
    /// - `rank`: the number of dims of the shape to match.
    ///
    /// ## Returns
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
    use super::*;
    use crate::expressions::DimExpr;

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
}
