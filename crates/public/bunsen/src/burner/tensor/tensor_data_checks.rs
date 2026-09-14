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
    },
};

/// Non-panicking tensor data checks.
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
    fn try_assert_eq(
        &self,
        expected: &TensorData,
        strict: bool,
    ) -> BunsenResult<()>;

    /// Try-Asserts element equality.
    ///
    /// [`TensorData::assert_eq_elem`] with a [`Result`].
    #[track_caller]
    fn try_assert_eq_elem<E: Element>(
        &self,
        other: &Self,
    ) -> BunsenResult<()>;

    /// Asserts the data is approximately equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `tolerance` - The tolerance of the comparison.
    /// * `strict` - Whether to strictly compare data types.
    ///
    /// # Panics
    ///
    /// Panics if the data is not approximately equal.
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
            return Err(BunsenError::AssertionError(format!(
                "Data types differ ({:?} != {:?})",
                self.dtype, other.dtype
            )));
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
                    return Err(BunsenError::AssertionError(
                        "Quantized data differs from other not quantized data".to_string(),
                    ));
                };

                // Data equality mostly depends on input quantization type, but
                // we also check level
                if q.value == q_other.value && q.level == q_other.level {
                    self.try_assert_eq_elem::<i8>(other)
                } else {
                    Err(BunsenError::AssertionError(format!(
                        "Quantization schemes differ ({q:?} != {q_other:?})"
                    )))
                }
            }
        }
    }

    #[track_caller]
    fn try_assert_eq_elem<E: Element>(
        &self,
        other: &Self,
    ) -> BunsenResult<()> {
        let mut message = String::new();
        if self.shape != other.shape {
            message += format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape, other.shape
            )
            .as_str();
        }

        let mut num_diff = 0;
        let max_num_diff = 5;
        for (i, (a, b)) in self.iter::<E>().zip(other.iter::<E>()).enumerate() {
            if !a.eq(&b) {
                // Only print the first 5 different values.
                if num_diff < max_num_diff {
                    message += format!("\n  => Position {i}: {a} != {b}").as_str();
                }
                num_diff += 1;
            }
        }

        if num_diff >= max_num_diff {
            message += format!("\n{} more errors...", num_diff - max_num_diff).as_str();
        }

        if !message.is_empty() {
            Err(BunsenError::AssertionError(message))
        } else {
            Ok(())
        }
    }

    #[track_caller]
    fn try_assert_approx_eq<F: Float + Element>(
        &self,
        other: &Self,
        tolerance: Tolerance<F>,
        strict: bool,
    ) -> BunsenResult<()> {
        if strict && self.dtype != other.dtype {
            return Err(BunsenError::AssertionError(format!(
                "Data types differ ({:?} != {:?})",
                self.dtype, other.dtype
            )));
        }
        if self.shape != other.shape {
            return Err(BunsenError::AssertionError(format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape, other.shape
            )));
        }

        let mut num_diff = 0;
        let max_num_diff = 5;

        let (relative, absolute) = unpack_tolerance(tolerance);
        let tol_rel = ToPrimitive::to_f64(&relative).unwrap();
        let tol_abs = ToPrimitive::to_f64(&absolute).unwrap();

        let mut message = String::new();
        for (i, (a, b)) in self.iter::<F>().zip(other.iter::<F>()).enumerate() {
            //if they are both nan, then they are equally nan
            let both_nan = a.is_nan() && b.is_nan();
            //this works for both infinities
            let both_inf =
                a.is_infinite() && b.is_infinite() && ((a > F::zero()) == (b > F::zero()));

            if both_nan || both_inf {
                continue;
            }

            if !tolerance.approx_eq(F::from(a).unwrap(), F::from(b).unwrap()) {
                // Only print the first 5 different values.
                if num_diff < max_num_diff {
                    let diff_abs = ToPrimitive::to_f64(&(a - b).abs()).unwrap();
                    let max = F::max(a.abs(), b.abs());
                    let diff_rel = diff_abs / ToPrimitive::to_f64(&max).unwrap();

                    message += format!(
                        "\n  => Position {i}: {a} != {b}\n     diff (rel = {diff_rel:+.2e}, abs = {diff_abs:+.2e}), tol (rel = {tol_rel:+.2e}, abs = {tol_abs:+.2e})"
                    )
                        .as_str();
                }
                num_diff += 1;
            }
        }

        if num_diff >= max_num_diff {
            message += format!("\n{} more errors...", num_diff - 5).as_str();
        }

        if !message.is_empty() {
            Err(BunsenError::AssertionError(message))
        } else {
            Ok(())
        }
    }
}
