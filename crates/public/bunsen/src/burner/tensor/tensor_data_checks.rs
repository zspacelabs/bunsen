use burn::{
    prelude::TensorData,
    tensor::{
        BoolStore,
        DType,
        Element,
        Tolerance,
        bf16,
        f16,
    },
};
use num_traits::{
    Float,
    ToPrimitive,
};

use crate::{
    burner::descriptors::unpack_tolerance,
    errors::{
        BunsenError,
        BunsenResult,
        ValueMismatch,
    },
};

/// How many differing values a check lists; the rest are only counted.
const MAX_LISTED_DIFFS: usize = 5;

/// Non-panicking tensor data checks.
///
/// # Errors
/// Each check fails with a [`Policy`](crate::errors::BunsenErrorKind::Policy)
/// error whose cause is a [`ValueMismatch`]: `DType`, `Quantization`, `Shape`,
/// or `Values`, which lists the first differing values in the error's
/// details.
pub trait TensorDataCheckExt {
    /// Try-Asserts the data is equal to another data.
    ///
    /// [`TensorData::assert_eq`] with a [`Result`].
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `strict` - If true, the data types must the be same. Otherwise, the
    ///   comparison is done in the current data type.
    ///
    /// # Returns
    /// `Ok(())` if the data is equal; `Err(BunsenError)` otherwise.
    ///
    /// # Errors
    /// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
    /// [`ValueMismatch`] cause, if the data differs.
    fn try_assert_eq(
        &self,
        expected: &TensorData,
        strict: bool,
    ) -> BunsenResult<()>;

    /// Try-Asserts element equality.
    ///
    /// [`TensorData::assert_eq_elem`] with a [`Result`].
    ///
    /// # Errors
    /// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
    /// [`ValueMismatch`] cause, if the shapes or any elements differ.
    #[track_caller]
    fn try_assert_eq_elem<E: Element>(
        &self,
        other: &Self,
    ) -> BunsenResult<()>;

    /// Try-Asserts the data is approximately equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `tolerance` - The tolerance of the comparison.
    /// * `strict` - Whether to strictly compare data types.
    ///
    /// # Errors
    /// [`Policy`](crate::errors::BunsenErrorKind::Policy), with a
    /// [`ValueMismatch`] cause, if the data is not approximately equal.
    #[track_caller]
    fn try_assert_approx_eq<F: Float + Element>(
        &self,
        other: &Self,
        tolerance: Tolerance<F>,
        strict: bool,
    ) -> BunsenResult<()>;
}

impl TensorDataCheckExt for TensorData {
    fn try_assert_eq(
        &self,
        other: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        if strict && self.dtype != other.dtype {
            return Err(dtype_mismatch(self, other));
        }

        match self.dtype {
            DType::F64 => self.try_assert_eq_elem::<f64>(other),
            DType::F32 | DType::Flex32 => self.try_assert_eq_elem::<f32>(other),
            DType::F16 => self.try_assert_eq_elem::<f16>(other),
            DType::BF16 => self.try_assert_eq_elem::<bf16>(other),
            DType::I64 => self.try_assert_eq_elem::<i64>(other),
            DType::I32 => self.try_assert_eq_elem::<i32>(other),
            DType::I16 => self.try_assert_eq_elem::<i16>(other),
            DType::I8 => self.try_assert_eq_elem::<i8>(other),
            DType::U64 => self.try_assert_eq_elem::<u64>(other),
            DType::U32 => self.try_assert_eq_elem::<u32>(other),
            DType::U16 => self.try_assert_eq_elem::<u16>(other),
            DType::U8 => self.try_assert_eq_elem::<u8>(other),
            DType::Bool(BoolStore::Native) => self.try_assert_eq_elem::<bool>(other),
            DType::Bool(BoolStore::U8) => self.try_assert_eq_elem::<u8>(other),
            DType::Bool(BoolStore::U32) => self.try_assert_eq_elem::<u32>(other),
            DType::QFloat(q) => {
                // Strict or not, it doesn't make sense to compare quantized
                // data to not quantized data for equality
                let q_other = if let DType::QFloat(q_other) = other.dtype {
                    q_other
                } else {
                    return Err(ValueMismatch::Quantization {
                        actual: format!("{q:?}"),
                        expected: format!("not quantized ({:?})", other.dtype),
                    }
                    .into());
                };

                // Data equality mostly depends on input quantization type, but
                // we also check level
                if q.value == q_other.value && q.level == q_other.level {
                    self.try_assert_eq_elem::<i8>(other)
                } else {
                    Err(ValueMismatch::Quantization {
                        actual: format!("{q:?}"),
                        expected: format!("{q_other:?}"),
                    }
                    .into())
                }
            }
        }
    }

    #[track_caller]
    fn try_assert_eq_elem<E: Element>(
        &self,
        other: &Self,
    ) -> BunsenResult<()> {
        if self.shape != other.shape {
            return Err(shape_mismatch(self, other));
        }

        let mut num_diff = 0;
        let mut total = 0;
        let mut first = Vec::new();
        for (i, (a, b)) in self.iter::<E>().zip(other.iter::<E>()).enumerate() {
            total += 1;
            if !a.eq(&b) {
                if num_diff < MAX_LISTED_DIFFS {
                    first.push(format!("position {i}: {a} != {b}"));
                }
                num_diff += 1;
            }
        }

        values_result(num_diff, total, first)
    }

    #[track_caller]
    fn try_assert_approx_eq<F: Float + Element>(
        &self,
        other: &Self,
        tolerance: Tolerance<F>,
        strict: bool,
    ) -> BunsenResult<()> {
        if strict && self.dtype != other.dtype {
            return Err(dtype_mismatch(self, other));
        }
        if self.shape != other.shape {
            return Err(shape_mismatch(self, other));
        }

        let mut num_diff = 0;
        let mut total = 0;
        let mut first = Vec::new();

        let (relative, absolute) = unpack_tolerance(tolerance);
        let tol_rel = ToPrimitive::to_f64(&relative).unwrap();
        let tol_abs = ToPrimitive::to_f64(&absolute).unwrap();

        for (i, (a, b)) in self.iter::<F>().zip(other.iter::<F>()).enumerate() {
            total += 1;
            //if they are both nan, then they are equally nan
            let both_nan = a.is_nan() && b.is_nan();
            //this works for both infinities
            let both_inf =
                a.is_infinite() && b.is_infinite() && ((a > F::zero()) == (b > F::zero()));

            if both_nan || both_inf {
                continue;
            }

            if !tolerance.approx_eq(F::from(a).unwrap(), F::from(b).unwrap()) {
                if num_diff < MAX_LISTED_DIFFS {
                    let diff_abs = ToPrimitive::to_f64(&(a - b).abs()).unwrap();
                    let max = F::max(a.abs(), b.abs());
                    let diff_rel = diff_abs / ToPrimitive::to_f64(&max).unwrap();

                    first.push(format!(
                        "position {i}: {a} != {b}; diff (rel = {diff_rel:+.2e}, abs = {diff_abs:+.2e}), tol (rel = {tol_rel:+.2e}, abs = {tol_abs:+.2e})"
                    ));
                }
                num_diff += 1;
            }
        }

        values_result(num_diff, total, first)
    }
}

/// The data types of `actual` and `expected` differ.
#[track_caller]
fn dtype_mismatch(
    actual: &TensorData,
    expected: &TensorData,
) -> BunsenError {
    ValueMismatch::DType {
        actual: format!("{:?}", actual.dtype),
        expected: format!("{:?}", expected.dtype),
    }
    .into()
}

/// The shapes of `actual` and `expected` differ.
#[track_caller]
fn shape_mismatch(
    actual: &TensorData,
    expected: &TensorData,
) -> BunsenError {
    ValueMismatch::Shape {
        actual: actual.shape.to_vec(),
        expected: expected.shape.to_vec(),
    }
    .into()
}

/// `Ok` if no values differ, else a `Values` mismatch.
#[track_caller]
fn values_result(
    count: usize,
    total: usize,
    first: Vec<String>,
) -> BunsenResult<()> {
    if count == 0 {
        Ok(())
    } else {
        Err(ValueMismatch::Values {
            count,
            total,
            first,
        }
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{
        BunsenErrorKind,
        testing::{
            ErrorMatcher,
            predicate,
        },
    };

    #[test]
    fn test_try_assert_eq_dtype() {
        let a = TensorData::from([1.0f32]);
        let b = TensorData::from([1.0f64]);
        a.try_assert_eq(&b, false).unwrap();
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq("data types differ: F32 != F64")
            .assert_err(&a.try_assert_eq(&b, true));
    }

    #[test]
    fn test_try_assert_eq_shape() {
        let a = TensorData::from([1.0f32, 2.0]);
        let b = TensorData::from([[1.0f32, 2.0]]);
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq("shapes differ: [2] != [1, 2]")
            .assert_err(&a.try_assert_eq(&b, true));
    }

    #[test]
    fn test_try_assert_eq_lists_five_values() {
        let a = TensorData::from([0.0f32; 7]);
        let b = TensorData::from([1.0f32; 7]);
        let five = TensorData::from([1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);

        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq("7 of 7 values differ")
            .details_contains("position 4: 0 != 1\n... and 2 more")
            .cause(predicate("five listed", |m: &ValueMismatch| {
                matches!(m, ValueMismatch::Values { count: 7, total: 7, first } if first.len() == 5)
            }))
            .assert_err(&a.try_assert_eq(&b, true));

        let e = a.try_assert_eq(&five, true).unwrap_err();
        assert_eq!(e.message(), "5 of 7 values differ");
        assert!(!e.details().unwrap().contains("more"), "{e:#}");
    }

    #[test]
    fn test_try_assert_approx_eq() {
        let a = TensorData::from([1.0f32, 2.0]);
        let b = TensorData::from([1.0f32, 2.5]);
        ErrorMatcher::kind(BunsenErrorKind::Policy)
            .message_eq("1 of 2 values differ")
            .details_contains("position 1: 2 != 2.5")
            .assert_err(&a.try_assert_approx_eq(&b, Tolerance::<f32>::balanced(), true));
    }
}
