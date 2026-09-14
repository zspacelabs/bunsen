use burn::{
    prelude::TensorData,
    tensor::{
        DType,
        Element,
        FloatDType,
        Tolerance,
        bf16,
        f16,
    },
};
use num_traits::Float;
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    prelude::TensorDataCheckExt,
};

/// Unpacks [`Tolerance<F>`] into `(relative, absolute)`.
///
/// The fields of [`Tolerance`] aren't public, so this is a workaround.
///
/// TODO: remove when <https://github.com/tracel-ai/burn/pull/5674> is released.
pub fn unpack_tolerance<F: Float + Sized>(tolerance: Tolerance<F>) -> (F, F) {
    #[repr(C)]
    struct ToleranceShim<F> {
        pub relative: F,
        pub absolute: F,
    }

    // `std::mem::transmute` requires the compiler to prove that both types have
    // the same size at monomorphization time, which it cannot do through a
    // generic parameter `F`. `transmute_copy` performs the same bitwise move
    // without that static check; guard it with a runtime size assertion to
    // preserve safety in case burn's `Tolerance<F>` layout ever changes.
    assert_eq!(
        std::mem::size_of::<Tolerance<F>>(),
        std::mem::size_of::<ToleranceShim<F>>(),
    );
    let tolerance = std::mem::ManuallyDrop::new(tolerance);
    let shim: ToleranceShim<F> =
        unsafe { std::mem::transmute_copy::<Tolerance<F>, ToleranceShim<F>>(&tolerance) };
    (shim.relative, shim.absolute)
}

/// Float-type-independent description of a [`Tolerance`] rule.
///
/// Each variant corresponds to one [`Tolerance`] constructor. The rule is held
/// symbolically rather than as resolved numbers, because the resolved values
/// depend on the float type: [`Tolerance::strict`] is
/// `64 * F::min_positive_value()`, which differs by many orders of magnitude
/// between `f16` and `f64`. Deferring resolution to
/// [`to_tolerance`](`TolerancePolicy::to_tolerance`) also reproduces burn's
/// rounding of the literal constants into `F`.
///
/// # Equality
/// [`PartialEq`] is *syntactic*, not semantic: [`TolerancePolicy::Strict`] and
/// a [`TolerancePolicy::RelAbs`] carrying the same resolved values are not
/// equal. This is the desired behaviour for audit matching, which compares
/// recorded intent.
#[derive(Default, Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind")]
pub enum TolerancePolicy {
    /// `relative = 0`, `absolute = 64 * F::MIN_POSITIVE`.
    ///
    /// See [`Tolerance::strict`].
    Strict,

    /// `relative = 0.5%`, `absolute = 1e-5`.
    ///
    /// See [`Tolerance::balanced`].
    #[default]
    Balanced,

    /// `relative = 1%`, `absolute = 1%`.
    ///
    /// See [`Tolerance::permissive`].
    Permissive,

    /// Relative difference only; `|x - y| < R * max(|x|, |y|)`.
    ///
    /// See [`Tolerance::relative`].
    Relative {
        /// Relative tolerance; must be in `0.0..=1.0`.
        relative: f64,
    },

    /// Absolute difference only; `|x - y| < A`.
    ///
    /// See [`Tolerance::absolute`].
    Absolute {
        /// Absolute tolerance; must be non-negative.
        absolute: f64,
    },

    /// `|x - y| < max(R * max(|x|, |y|), A)`.
    ///
    /// See [`Tolerance::rel_abs`].
    RelAbs {
        /// Relative tolerance; must be in `0.0..=1.0`.
        relative: f64,

        /// Absolute tolerance; must be non-negative.
        absolute: f64,
    },
}

impl TolerancePolicy {
    /// Validate the tolerance bounds.
    ///
    /// [`Tolerance`]'s own constructors assert these, so a policy which has
    /// not been checked will panic inside burn rather than returning an error.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] if a relative tolerance is outside
    /// `0.0..=1.0`, or an absolute tolerance is negative.
    pub fn validate(&self) -> BunsenResult<()> {
        let check_relative = |relative: f64| -> BunsenResult<()> {
            if !(0.0..=1.0).contains(&relative) {
                return Err(BunsenError::InvalidArgument {
                    msg: format!("relative tolerance ({relative}) is not in 0.0..=1.0"),
                });
            }
            Ok(())
        };
        let check_absolute = |absolute: f64| -> BunsenResult<()> {
            if absolute < 0.0 {
                return Err(BunsenError::InvalidArgument {
                    msg: format!("absolute tolerance ({absolute}) is negative"),
                });
            }
            Ok(())
        };

        match *self {
            Self::Strict | Self::Balanced | Self::Permissive => Ok(()),
            Self::Relative { relative } => check_relative(relative),
            Self::Absolute { absolute } => check_absolute(absolute),
            Self::RelAbs { relative, absolute } => {
                check_relative(relative)?;
                check_absolute(absolute)
            }
        }
    }

    /// Resolve to a burn [`Tolerance`] at the float type `F`.
    ///
    /// # Panics
    /// If the policy has not been validated; see
    /// [`validate`](`TolerancePolicy::validate`).
    pub fn to_tolerance<F: Float>(&self) -> Tolerance<F> {
        match *self {
            Self::Strict => Tolerance::strict(),
            Self::Balanced => Tolerance::balanced(),
            Self::Permissive => Tolerance::permissive(),
            Self::Relative { relative } => Tolerance::relative(relative),
            Self::Absolute { absolute } => Tolerance::absolute(absolute),
            Self::RelAbs { relative, absolute } => Tolerance::rel_abs(relative, absolute),
        }
    }
}

/// Reference-free description of a [`Tolerance<F>`], including the `F`.
///
/// A [`TolerancePolicy`] alone under-specifies a comparison, because the
/// resolved tolerance depends on the float type. Pairing the element type with
/// the policy makes the description self-contained, and lets
/// [`try_assert_tensor_data_approx_eq`](`ToleranceDesc::try_assert_tensor_data_approx_eq`)
/// dispatch on it without a type parameter.
///
/// # Element Type
/// `dtype` is the type the *comparison* runs at, which need not match the
/// dtype of the data being compared: [`TensorData`] iteration converts. A
/// `dtype` narrower than the data deliberately compares at reduced precision.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ToleranceDesc {
    /// The float type the comparison runs at.
    #[serde(with = "crate::burner::descriptors::shims::FloatDTypeShim")]
    float_dtype: FloatDType,

    /// The float-type-independent tolerance rule.
    policy: TolerancePolicy,
}

impl ToleranceDesc {
    /// Construct a [`ToleranceDesc`].
    ///
    /// # Arguments
    /// * `dtype` - the float type to compare at.
    /// * `policy` - the tolerance rule.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] if `policy` fails
    /// [`validate`](`TolerancePolicy::validate`).
    pub fn new(
        dtype: FloatDType,
        policy: TolerancePolicy,
    ) -> BunsenResult<Self> {
        policy.validate()?;
        Ok(Self {
            float_dtype: dtype,
            policy,
        })
    }

    /// Construct a [`ToleranceDesc`], narrowing a burn [`DType`].
    ///
    /// # Arguments
    /// * `dtype` - the float type to compare at; see [`FloatDType`].
    /// * `policy` - the tolerance rule.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] if `dtype` is not a float type, or if
    /// `policy` fails [`validate`](`TolerancePolicy::validate`).
    pub fn try_from_dtype(
        dtype: DType,
        policy: TolerancePolicy,
    ) -> BunsenResult<Self> {
        if !dtype.is_float() {
            return Err(BunsenError::InvalidArgument {
                msg: format!("dtype ({dtype:?}) is not a float type"),
            });
        }
        Self::new(FloatDType::from(dtype), policy)
    }

    /// Construct a [`ToleranceDesc`] comparing at the float type `F`.
    ///
    /// # Arguments
    /// * `policy` - the tolerance rule.
    ///
    /// # Errors
    /// [`BunsenError::InvalidArgument`] if `policy` fails
    /// [`validate`](`TolerancePolicy::validate`).
    pub fn of<F: Float + Element>(policy: TolerancePolicy) -> BunsenResult<Self> {
        Self::new(FloatDType::from(F::dtype()), policy)
    }

    /// The [`FloatDType`] the comparison runs at.
    pub fn float_dtype(&self) -> FloatDType {
        self.float_dtype
    }

    /// The [`DType`] the comparison runs at.
    pub fn dtype(&self) -> DType {
        self.float_dtype.into()
    }

    /// The float-type-independent tolerance rule.
    pub fn policy(&self) -> TolerancePolicy {
        self.policy
    }

    /// Resolve to a burn [`Tolerance`] at the float type `F`.
    ///
    /// Prefer
    /// [`try_assert_tensor_data_approx_eq`](`ToleranceDesc::try_assert_tensor_data_approx_eq`),
    /// which resolves `F` from [`dtype`](`ToleranceDesc::dtype`); this bypasses
    /// that and lets `F` disagree with the recorded element type.
    pub fn to_tolerance_as<F: Float>(&self) -> Tolerance<F> {
        self.policy.to_tolerance()
    }

    /// Assert two [`TensorData`] are approximately equal under this tolerance.
    ///
    /// Dispatches [`TensorDataCheckExt::try_assert_approx_eq`] at
    /// [`dtype`](`ToleranceDesc::dtype`).
    ///
    /// # Arguments
    /// * `actual` - the observed data.
    /// * `expected` - the expected data.
    /// * `strict` - whether to enforce strict dtype equality.
    ///
    /// # Errors
    /// [`BunsenError::AssertionError`] describing the first differing
    /// positions, if the data are not approximately equal.
    pub fn try_assert_tensor_data_approx_eq(
        &self,
        actual: &TensorData,
        expected: &TensorData,
        strict: bool,
    ) -> BunsenResult<()> {
        match self.float_dtype {
            FloatDType::F64 => {
                actual.try_assert_approx_eq::<f64>(expected, self.policy.to_tolerance(), strict)
            }
            FloatDType::F32 | FloatDType::Flex32 => {
                actual.try_assert_approx_eq::<f32>(expected, self.policy.to_tolerance(), strict)
            }
            FloatDType::F16 => {
                actual.try_assert_approx_eq::<f16>(expected, self.policy.to_tolerance(), strict)
            }
            FloatDType::BF16 => {
                actual.try_assert_approx_eq::<bf16>(expected, self.policy.to_tolerance(), strict)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_default_is_balanced() {
        assert_eq!(TolerancePolicy::default(), TolerancePolicy::Balanced);
    }

    /// `Strict` resolves differently per float type; this is the property that
    /// justifies holding the policy symbolically instead of as resolved values.
    #[test]
    fn test_strict_depends_on_float_type() {
        let policy = TolerancePolicy::Strict;

        // 64 * f16::MIN_POSITIVE ~= 3.9e-3; 64 * f32::MIN_POSITIVE ~= 7.5e-37.
        let half = policy.to_tolerance::<f16>();
        let single = policy.to_tolerance::<f32>();

        assert!(half.approx_eq(f16::from_f32(1.0), f16::from_f32(1.001)));
        assert!(!single.approx_eq(1.0f32, 1.001f32));
    }

    #[test]
    fn test_policies_resolve_to_burn_behaviour() {
        assert!(
            TolerancePolicy::Permissive
                .to_tolerance::<f32>()
                .approx_eq(1.0, 1.005)
        );
        assert!(
            !TolerancePolicy::Strict
                .to_tolerance::<f32>()
                .approx_eq(1.0, 1.005)
        );

        let rel_abs = TolerancePolicy::RelAbs {
            relative: 0.0,
            absolute: 0.01,
        };
        assert!(rel_abs.to_tolerance::<f32>().approx_eq(1.0, 1.005));
        assert!(!rel_abs.to_tolerance::<f32>().approx_eq(1.0, 1.05));

        let relative = TolerancePolicy::Relative { relative: 0.1 };
        assert!(relative.to_tolerance::<f32>().approx_eq(100.0, 105.0));
        assert!(!relative.to_tolerance::<f32>().approx_eq(100.0, 120.0));

        let absolute = TolerancePolicy::Absolute { absolute: 0.5 };
        assert!(absolute.to_tolerance::<f32>().approx_eq(100.0, 100.4));
        assert!(!absolute.to_tolerance::<f32>().approx_eq(100.0, 100.6));
    }

    #[test]
    fn test_validate() -> BunsenResult<()> {
        TolerancePolicy::Strict.validate()?;
        TolerancePolicy::Balanced.validate()?;
        TolerancePolicy::Permissive.validate()?;
        TolerancePolicy::Relative { relative: 1.0 }.validate()?;
        TolerancePolicy::Absolute { absolute: 0.0 }.validate()?;

        let err = TolerancePolicy::Relative { relative: 2.0 }
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("not in 0.0..=1.0"), "{err}");

        let err = TolerancePolicy::Absolute { absolute: -1.0 }
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("is negative"), "{err}");

        let err = TolerancePolicy::RelAbs {
            relative: 0.5,
            absolute: -1.0,
        }
        .validate()
        .unwrap_err();
        assert!(err.to_string().contains("is negative"), "{err}");

        Ok(())
    }

    #[test]
    fn test_desc_records_the_float_type() -> BunsenResult<()> {
        let policy = TolerancePolicy::Balanced;

        assert_eq!(
            ToleranceDesc::of::<f32>(policy)?.float_dtype(),
            FloatDType::F32
        );
        assert_eq!(
            ToleranceDesc::of::<f64>(policy)?.float_dtype(),
            FloatDType::F64
        );
        assert_eq!(
            ToleranceDesc::of::<f16>(policy)?.float_dtype(),
            FloatDType::F16
        );
        assert_eq!(
            ToleranceDesc::of::<bf16>(policy)?.float_dtype(),
            FloatDType::BF16
        );

        assert_eq!(ToleranceDesc::of::<f32>(policy)?.policy(), policy);

        Ok(())
    }

    /// A desc differing only in dtype is a different desc; this is what makes
    /// a replay at a different float type fail on params equality.
    #[test]
    fn test_desc_equality_includes_dtype() -> BunsenResult<()> {
        let policy = TolerancePolicy::Strict;
        assert_ne!(
            ToleranceDesc::of::<f32>(policy)?,
            ToleranceDesc::of::<f16>(policy)?
        );
        Ok(())
    }

    #[test]
    fn test_new_rejects_non_float_dtype() {
        let err = ToleranceDesc::try_from_dtype(DType::I32, TolerancePolicy::Balanced).unwrap_err();
        assert!(err.to_string().contains("is not a float type"), "{err}");
    }

    #[test]
    fn test_new_rejects_invalid_policy() {
        let err = ToleranceDesc::new(FloatDType::F32, TolerancePolicy::Relative { relative: 2.0 })
            .unwrap_err();
        assert!(err.to_string().contains("not in 0.0..=1.0"), "{err}");
    }

    #[test]
    fn test_flex32_is_preserved_and_compares_as_f32() -> BunsenResult<()> {
        let desc = ToleranceDesc::try_from_dtype(DType::Flex32, TolerancePolicy::Balanced)?;
        assert_eq!(desc.float_dtype(), FloatDType::Flex32);
        desc.try_assert_tensor_data_approx_eq(
            &TensorData::from([1.0f32]),
            &TensorData::from([1.001f32]),
            true,
        )
    }

    /// The desc dispatches at its own recorded dtype: the same policy accepts a
    /// difference at `f16` that it rejects at `f32`.
    #[test]
    fn test_assert_dispatches_at_recorded_dtype() -> BunsenResult<()> {
        let a = TensorData::from([1.0f32]);
        let b = TensorData::from([1.001f32]);

        ToleranceDesc::of::<f16>(TolerancePolicy::Strict)?
            .try_assert_tensor_data_approx_eq(&a, &b, true)
    }

    #[test]
    fn test_assert_dispatches_at_recorded_dtype_f32() -> BunsenResult<()> {
        let a = TensorData::from([1.0f32]);
        let b = TensorData::from([1.001f32]);

        let err = ToleranceDesc::of::<f32>(TolerancePolicy::Strict)?
            .try_assert_tensor_data_approx_eq(&a, &b, true)
            .unwrap_err();
        assert!(err.to_string().contains("Position 0: 1 != 1.001"), "{err}");

        Ok(())
    }

    #[test]
    fn test_policy_serde_round_trip() -> BunsenResult<()> {
        for policy in [
            TolerancePolicy::Strict,
            TolerancePolicy::Balanced,
            TolerancePolicy::Permissive,
            TolerancePolicy::Relative { relative: 0.25 },
            TolerancePolicy::Absolute { absolute: 1e-5 },
            TolerancePolicy::RelAbs {
                relative: 0.01,
                absolute: 1e-6,
            },
        ] {
            let json = serde_json::to_string(&policy).unwrap();
            let back: TolerancePolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(policy, back, "round trip failed for {json}");
        }

        assert_eq!(
            serde_json::to_string(&TolerancePolicy::Strict).unwrap(),
            r#"{"kind":"Strict"}"#
        );

        Ok(())
    }

    #[test]
    fn test_desc_serde_round_trip() -> BunsenResult<()> {
        let desc = ToleranceDesc::of::<f16>(TolerancePolicy::RelAbs {
            relative: 0.01,
            absolute: 1e-6,
        })?;

        let json = serde_json::to_string(&desc).unwrap();
        let back: ToleranceDesc = serde_json::from_str(&json).unwrap();

        assert_eq!(desc, back);
        assert!(json.contains("F16"), "{json}");

        Ok(())
    }
}
