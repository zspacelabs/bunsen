//! `TensorMatcher`: the shape, dtype and values of a tensor, every clause
//! checked at once.
//!
//! Crate-internal until it settles (decision `[cd_tensor_matcher]`): bunsen's
//! own tests use it, beside the plain assertion functions.

use alloc::{
    boxed::Box,
    format,
    string::String,
    vec,
    vec::Vec,
};
use core::fmt::Write;

use burn::{
    prelude::Tensor,
    tensor::DType,
};

use crate::{
    burner::tensor::TensorDataCheckExt,
    support::testing::{
        DeviceTolerance,
        matching::{
            Description,
            Matcher,
        },
    },
};

/// One clause of a [`TensorMatcher`].
enum Clause<const D: usize> {
    Shape([usize; D]),
    DType(DType),
    Close {
        expected: Box<Tensor<D>>,
        tolerance: Box<dyn DeviceTolerance + Send + Sync>,
        what: &'static str,
    },
    Finite,
    InRange(f64, f64),
}

/// A matcher on a `Tensor<D>`, built fluently: its shape, its dtype, its
/// values against another tensor's, finiteness, a range. A check reports
/// every clause that missed, not the first; with
/// [`valid_prefix`](Self::valid_prefix) the value clauses compare only each
/// item's valid frames.
pub(crate) struct TensorMatcher<const D: usize> {
    what: Option<String>,
    clauses: Vec<Clause<D>>,
    valid: Option<(usize, Vec<usize>)>,
}

impl<const D: usize> TensorMatcher<D> {
    /// A matcher with no clauses.
    pub(crate) fn new() -> Self {
        Self {
            what: None,
            clauses: Vec::new(),
            valid: None,
        }
    }

    /// Values within `tolerance` of `expected`'s, which also fixes the shape.
    pub(crate) fn close_to(
        expected: &Tensor<D>,
        tolerance: impl DeviceTolerance + Send + Sync + 'static,
    ) -> Self {
        Self::new().with_close(expected, tolerance, "close to")
    }

    /// Values equal to `expected`'s: moved or copied values.
    pub(crate) fn equal_to(expected: &Tensor<D>) -> Self {
        Self::new().with_close(expected, crate::support::testing::IDENTICAL, "equal to")
    }

    fn with_close(
        mut self,
        expected: &Tensor<D>,
        tolerance: impl DeviceTolerance + Send + Sync + 'static,
        what: &'static str,
    ) -> Self {
        self.clauses.push(Clause::Shape(expected.dims()));
        self.clauses.push(Clause::Close {
            expected: Box::new(expected.clone()),
            tolerance: Box::new(tolerance),
            what,
        });
        self
    }

    /// The shape is `dims`.
    pub(crate) fn shape(
        mut self,
        dims: [usize; D],
    ) -> Self {
        self.clauses.push(Clause::Shape(dims));
        self
    }

    /// The dtype is `dtype`.
    pub(crate) fn dtype(
        mut self,
        dtype: DType,
    ) -> Self {
        self.clauses.push(Clause::DType(dtype));
        self
    }

    /// Every value is finite.
    pub(crate) fn finite(mut self) -> Self {
        self.clauses.push(Clause::Finite);
        self
    }

    /// Every value is in `low..=high`.
    pub(crate) fn in_range(
        mut self,
        low: f64,
        high: f64,
    ) -> Self {
        self.clauses.push(Clause::InRange(low, high));
        self
    }

    /// The value clauses compare, for each item `b` (axis 0), only its first
    /// `lengths[b]` positions on `axis`: past them a padded batch's values mean
    /// nothing.
    pub(crate) fn valid_prefix(
        mut self,
        axis: usize,
        lengths: &[usize],
    ) -> Self {
        assert!(
            axis > 0 && axis < D,
            "a valid prefix on axis {axis} of a rank-{D} tensor"
        );
        self.valid = Some((axis, lengths.to_vec()));
        self
    }

    /// A name for the tensor, which a failure leads with.
    pub(crate) fn named(
        mut self,
        what: impl Into<String>,
    ) -> Self {
        self.what = Some(what.into());
        self
    }

    /// The views the value clauses compare: the whole tensor, or each item's
    /// valid prefix.
    fn views(
        &self,
        tensor: &Tensor<D>,
    ) -> Vec<(String, Tensor<D>)> {
        match &self.valid {
            None => vec![(String::new(), tensor.clone())],
            Some((axis, lengths)) => lengths
                .iter()
                .enumerate()
                .map(|(b, &n)| {
                    let item = tensor.clone().slice_dim(0, b..b + 1).slice_dim(*axis, 0..n);
                    (format!("item {b} (..{n}): "), item)
                })
                .collect(),
        }
    }

    /// Every clause's miss, as a line; empty when `actual` matches.
    pub(crate) fn misses(
        &self,
        actual: &Tensor<D>,
    ) -> Vec<String> {
        let mut misses = Vec::new();
        let dims = actual.dims();
        let mut shape_ok = true;
        for clause in &self.clauses {
            match clause {
                Clause::Shape(want) if dims != *want => {
                    shape_ok = false;
                    misses.push(format!("dims were {dims:?}, not {want:?}"));
                }
                Clause::DType(want) if actual.dtype() != *want => {
                    misses.push(format!("dtype was {:?}, not {want:?}", actual.dtype()));
                }
                _ => {}
            }
        }
        if !shape_ok {
            return misses;
        }
        let device = actual.device();
        let actual_views = self.views(actual);
        for clause in &self.clauses {
            match clause {
                Clause::Close {
                    expected,
                    tolerance,
                    what,
                } => {
                    let tolerance = tolerance.on(&device);
                    for ((at, a), (_, e)) in actual_views.iter().zip(self.views(expected)) {
                        let (a, e) = (a.to_data_as::<f32>(), e.to_data_as::<f32>());
                        if let Err(err) = a.try_assert_approx_eq::<f32>(&e, tolerance, false) {
                            misses.push(format!("{at}not {what} the expected values: {err:#}"));
                        }
                    }
                }
                Clause::Finite => {
                    for (at, a) in &actual_views {
                        let values = a.to_data_as::<f32>();
                        let bad = values.iter::<f32>().filter(|v| !v.is_finite()).count();
                        if bad > 0 {
                            misses.push(format!("{at}{bad} values are not finite"));
                        }
                    }
                }
                Clause::InRange(low, high) => {
                    for (at, a) in &actual_views {
                        let values = a.to_data_as::<f32>();
                        let outside: Vec<f32> = values
                            .iter::<f32>()
                            .filter(|&v| !(*low..=*high).contains(&f64::from(v)))
                            .collect();
                        if !outside.is_empty() {
                            misses.push(format!(
                                "{at}{} values are outside {low}..={high}, the first {}",
                                outside.len(),
                                outside[0]
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        misses
    }

    /// Asserts that `actual` matches.
    ///
    /// # Panics
    /// When any clause misses; the message lists every miss.
    #[track_caller]
    pub(crate) fn assert(
        &self,
        actual: &Tensor<D>,
    ) {
        let misses = self.misses(actual);
        if !misses.is_empty() {
            panic!(
                "{}: {}",
                self.what.as_deref().unwrap_or("the tensor"),
                misses.join("; ")
            );
        }
    }
}

impl<const D: usize> Matcher<Tensor<D>> for TensorMatcher<D> {
    fn matches(
        &self,
        actual: &Tensor<D>,
    ) -> bool {
        self.misses(actual).is_empty()
    }

    fn describe(
        &self,
        d: &mut Description,
    ) {
        let parts: Vec<String> = self
            .clauses
            .iter()
            .map(|clause| match clause {
                Clause::Shape(dims) => format!("dims {dims:?}"),
                Clause::DType(dtype) => format!("dtype {dtype:?}"),
                Clause::Close { what, .. } => format!("values {what} a reference"),
                Clause::Finite => String::from("finite values"),
                Clause::InRange(low, high) => format!("values in {low}..={high}"),
            })
            .collect();
        let _ = write!(d, "a tensor with {}", parts.join(", "));
    }

    fn describe_mismatch(
        &self,
        actual: &Tensor<D>,
        d: &mut Description,
    ) {
        let _ = write!(d, "{}", self.misses(actual).join("; "));
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Tolerance,
        s,
    };

    use super::*;
    use crate::support::testing::{
        ROUNDING,
        cpu_device,
    };

    fn t(values: [[f32; 3]; 2]) -> Tensor<2> {
        Tensor::from_data(values, &cpu_device())
    }

    #[test]
    fn test_matches() {
        let x = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        TensorMatcher::close_to(&x, ROUNDING)
            .dtype(DType::F32)
            .finite()
            .in_range(1.0, 6.0)
            .named("x")
            .assert(&x);
        TensorMatcher::equal_to(&x).assert(&x.clone());
        assert!(TensorMatcher::new().shape([2, 3]).matches(&x));
    }

    /// Every clause that misses is reported.
    #[test]
    fn test_every_miss_is_reported() {
        let x = t([[1.0, 2.0, 3.0], [4.0, 5.0, f32::NAN]]);
        let misses = TensorMatcher::close_to(&t([[1.0, 2.0, 3.0]; 2]), Tolerance::absolute(1e-3))
            .dtype(DType::BF16)
            .finite()
            .in_range(0.0, 4.0)
            .misses(&x);
        assert_eq!(misses.len(), 4, "{misses:#?}");
        assert!(misses[0].starts_with("dtype was F32, not BF16"));
        assert!(misses[1].starts_with("not close to the expected values"));
        assert_eq!(misses[2], "1 values are not finite");
        assert!(misses[3].starts_with("2 values are outside 0..=4"));
    }

    /// A wrong shape stops the value clauses.
    #[test]
    #[should_panic(expected = "y: dims were [2, 3], not [1, 3]")]
    fn test_shape_stops_the_values() {
        let x = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        TensorMatcher::equal_to(&x.clone().slice(s![0..1, ..]))
            .named("y")
            .assert(&x);
    }

    /// Past each item's valid prefix, values may differ.
    #[test]
    fn test_valid_prefix() {
        let got = t([[1.0, 2.0, 9.0], [4.0, 8.0, 9.0]]);
        let want = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        TensorMatcher::equal_to(&want)
            .valid_prefix(1, &[2, 1])
            .assert(&got);
        let misses = TensorMatcher::equal_to(&want)
            .valid_prefix(1, &[2, 2])
            .misses(&got);
        assert_eq!(misses.len(), 1);
        assert!(misses[0].starts_with("item 1 (..2): "), "{misses:?}");
    }
}
