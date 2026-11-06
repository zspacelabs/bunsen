//! [`Widening`] and its [`WideningScope`].

use burn::{
    Tensor,
    tensor::{
        DType,
        FloatDType,
    },
};

use super::widen_to;

/// The dtype a computation runs in, from the dtype of what it computes on:
/// widened to at least [`at_least`](Self::at_least), never narrowed
/// ([`widen_to`]).
///
/// A [`scope`](Self::scope) widens the tensors a computation takes, and
/// narrows its results back; [`around`](Self::around) runs a closure in one,
/// and narrows what it returns:
///
/// ```
/// use bunsen::{
///     burner::tensor::Widening,
///     support::testing::cpu_device,
/// };
/// use burn::{
///     Tensor,
///     tensor::{
///         DType,
///         activation::silu,
///     },
/// };
///
/// fn gated(
///     x: Tensor<2>,
///     gate: Tensor<2>,
/// ) -> Tensor<2> {
///     // A half `x` and its gate multiplied in f32, the product in `x`'s dtype.
///     Widening::F32.around(x.dtype(), |w| w.widen(x) * silu(w.widen(gate)))
/// }
///
/// let device = cpu_device();
/// let x = Tensor::<2>::ones([2, 3], &device).cast(DType::BF16);
/// assert_eq!(gated(x.clone(), x).dtype(), DType::BF16);
/// ```
///
/// Serialized as its [`at_least`](Self::at_least), `null` for
/// [`NONE`](Self::NONE).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(transparent)]
pub struct Widening {
    /// The dtype a computation is widened to at least; `None`, it runs in
    /// the dtype of what it computes on.
    pub at_least: Option<FloatDType>,
}

impl Widening {
    /// Half precision widened to f32; any wider dtype kept
    /// ([`widen_half`](super::widen_half)'s).
    pub const F32: Self = Self::to(FloatDType::F32);
    /// No widening: a computation runs in the dtype of what it computes on.
    pub const NONE: Self = Self { at_least: None };

    /// Widening to at least `at_least`.
    pub const fn to(at_least: FloatDType) -> Self {
        Self {
            at_least: Some(at_least),
        }
    }

    /// The dtype a computation on `dtype` runs in.
    pub fn compute(
        &self,
        dtype: DType,
    ) -> DType {
        match self.at_least {
            Some(at_least) => widen_to(dtype, at_least),
            None => dtype,
        }
    }

    /// A scope for a computation on `dtype`: it widens to
    /// [`compute`](Self::compute)`(dtype)`, and narrows back to `dtype`.
    pub fn scope(
        &self,
        dtype: DType,
    ) -> WideningScope {
        WideningScope {
            dtype,
            compute: self.compute(dtype),
        }
    }

    /// `f`, run in a [`scope`](Self::scope) for `dtype`; its result narrowed
    /// to `dtype`.
    pub fn around<const R: usize>(
        &self,
        dtype: DType,
        f: impl FnOnce(&WideningScope) -> Tensor<R>,
    ) -> Tensor<R> {
        let scope = self.scope(dtype);
        scope.narrow(f(&scope))
    }
}

impl From<Option<FloatDType>> for Widening {
    fn from(at_least: Option<FloatDType>) -> Self {
        Self { at_least }
    }
}

/// A computation's two dtypes, from a [`Widening`]: the one it runs in
/// ([`compute`](Self::compute)), and the one its results are narrowed to
/// ([`dtype`](Self::dtype)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideningScope {
    dtype: DType,
    compute: DType,
}

impl WideningScope {
    /// The dtype results are narrowed to: that of what the computation is
    /// on.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The dtype the computation runs in.
    pub fn compute(&self) -> DType {
        self.compute
    }

    /// `x`, in the [`compute`](Self::compute) dtype: an input, or a
    /// parameter it meets.
    pub fn widen<const R: usize>(
        &self,
        x: Tensor<R>,
    ) -> Tensor<R> {
        cast_to(x, self.compute)
    }

    /// `x`, in the [`dtype`](Self::dtype) results are narrowed to.
    pub fn narrow<const R: usize>(
        &self,
        x: Tensor<R>,
    ) -> Tensor<R> {
        cast_to(x, self.dtype)
    }
}

/// `x` in `dtype`; as it is when it already is.
fn cast_to<const R: usize>(
    x: Tensor<R>,
    dtype: DType,
) -> Tensor<R> {
    if x.dtype() == dtype { x } else { x.cast(dtype) }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::support::testing::{
        cpu_device,
        seeded_tensor,
    };

    #[test]
    fn test_compute() {
        assert_eq!(Widening::F32.compute(DType::BF16), DType::F32);
        assert_eq!(Widening::F32.compute(DType::F16), DType::F32);
        assert_eq!(Widening::F32.compute(DType::F32), DType::F32);
        assert_eq!(Widening::F32.compute(DType::F64), DType::F64);
        assert_eq!(Widening::NONE.compute(DType::BF16), DType::BF16);
        assert_eq!(
            Widening::to(FloatDType::F64).compute(DType::F32),
            DType::F64
        );
        assert_eq!(Widening::default(), Widening::NONE);
        assert_eq!(Widening::from(Some(FloatDType::F32)), Widening::F32);
    }

    /// A scope widens and narrows between its two dtypes; a tensor already
    /// in the one it is cast to comes back as it is.
    #[test]
    fn test_scope() {
        let device = cpu_device();
        let scope = Widening::F32.scope(DType::BF16);
        assert_eq!((scope.dtype(), scope.compute()), (DType::BF16, DType::F32));
        let x = seeded_tensor::<2>(1, [2, 3], Distribution::Default, &device).cast(DType::BF16);
        let wide = scope.widen(x.clone());
        assert_eq!(wide.dtype(), DType::F32);
        assert_eq!(scope.widen(wide.clone()).dtype(), DType::F32);
        let back = scope.narrow(wide);
        assert_eq!(back.dtype(), DType::BF16);
        back.into_data().assert_eq(&x.into_data(), true);
    }

    /// `around` runs in the compute dtype and narrows the result: a bf16
    /// product computed in f32 is that product rounded once.
    #[test]
    fn test_around() {
        let device = cpu_device();
        let a = seeded_tensor::<2>(2, [3, 4], Distribution::Default, &device).cast(DType::BF16);
        let b = seeded_tensor::<2>(3, [3, 4], Distribution::Default, &device).cast(DType::BF16);
        let got = Widening::F32.around(a.dtype(), |w| {
            assert_eq!(w.compute(), DType::F32);
            w.widen(a.clone()) * w.widen(b.clone()).exp()
        });
        let expected =
            (a.clone().cast(DType::F32) * b.clone().cast(DType::F32).exp()).cast(DType::BF16);
        assert_eq!(got.dtype(), DType::BF16);
        got.into_data().assert_eq(&expected.into_data(), true);

        let unwidened =
            Widening::NONE.around(a.dtype(), |w| w.widen(a.clone()) * w.widen(b.clone()).exp());
        unwidened
            .into_data()
            .assert_eq(&(a * b.exp()).into_data(), true);
    }

    #[test]
    fn test_serde_form() {
        assert_eq!(serde_json::to_string(&Widening::NONE).unwrap(), "null");
        let f32 = serde_json::to_string(&Widening::F32).unwrap();
        assert_eq!(f32, serde_json::to_string(&FloatDType::F32).unwrap());
        assert_eq!(
            serde_json::from_str::<Widening>(&f32).unwrap(),
            Widening::F32
        );
    }
}
