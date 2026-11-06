//! [`Gate`], one module's output gated by another's.

use burn::{
    Tensor,
    tensor::FloatDType,
};

use super::{
    Forward1Module,
    InputDTypes,
    ResultDType,
};
use crate::burner::tensor::Widening;

/// A view of two modules as a gate: `a(x) ⊙ b(gate)`, the product taken in
/// a [`Widening`] of `a(x)`'s dtype and narrowed back to it.
///
/// ```text
/// y = a(x)
/// out = narrow(widen(y) ⊙ b(widen(gate)))
/// ```
///
/// A view borrows its modules: a block holds them as its own fields, under
/// its own record paths, and builds the view in its forward. Its dtypes are
/// `a`'s: it takes the inputs `a` takes, and returns `a`'s result; `b` runs
/// on the gate in the widened dtype.
///
/// `Gate::new(&norm, &silu)` is the gated `DeltaNet`'s output norm,
/// `Gate::new(&identity, &sigmoid)` an attention's output gate.
#[derive(Debug)]
pub struct Gate<'a, A, B> {
    a: &'a A,
    b: &'a B,
    widening: Widening,
}

impl<A, B> Clone for Gate<'_, A, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A, B> Copy for Gate<'_, A, B> {}

impl<'a, A, B> Gate<'a, A, B> {
    /// `a(x)` gated by `b(gate)`, the product in `a(x)`'s dtype.
    pub fn new(
        a: &'a A,
        b: &'a B,
    ) -> Self {
        Self {
            a,
            b,
            widening: Widening::NONE,
        }
    }

    /// Sets the [`Widening`] the gate's product is taken in.
    pub fn with_widening(
        self,
        widening: Widening,
    ) -> Self {
        Self { widening, ..self }
    }

    /// The gated module.
    pub fn a(&self) -> &'a A {
        self.a
    }

    /// The gate's module.
    pub fn b(&self) -> &'a B {
        self.b
    }

    /// The widening the product is taken in.
    pub fn widening(&self) -> Widening {
        self.widening
    }

    /// `a(x) ⊙ b(gate)`, in `a(x)`'s dtype.
    ///
    /// # Arguments
    /// - `x`: `a`'s input.
    /// - `gate`: `b`'s input, `a(x)`'s shape.
    ///
    /// # Returns
    /// - `a(x)`'s shape, in its dtype.
    pub fn forward<const R: usize>(
        &self,
        x: Tensor<R>,
        gate: Tensor<R>,
    ) -> Tensor<R>
    where
        A: Forward1Module<Tensor<R>, Output = Tensor<R>>,
        B: Forward1Module<Tensor<R>, Output = Tensor<R>>,
    {
        let y = self.a.forward(x);
        assert_eq!(gate.dims(), y.dims(), "a gate has its gated output's shape");
        self.widening
            .around(y.dtype(), |w| w.widen(y) * self.b.forward(w.widen(gate)))
    }
}

impl<A: InputDTypes, B> InputDTypes for Gate<'_, A, B> {
    /// `a`'s.
    fn preferred_dtype(&self) -> FloatDType {
        self.a.preferred_dtype()
    }

    /// `a`'s.
    fn accepts_dtype(
        &self,
        dtype: FloatDType,
    ) -> bool {
        self.a.accepts_dtype(dtype)
    }
}

impl<A: ResultDType, B> ResultDType for Gate<'_, A, B> {
    /// `a`'s.
    fn result_dtype(
        &self,
        input: FloatDType,
    ) -> FloatDType {
        self.a.result_dtype(input)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Module,
        nn::RmsNormConfig,
        tensor::{
            DType,
            Distribution,
            Tolerance,
            activation::silu,
        },
    };

    use super::*;
    use crate::{
        blocks::{
            activation::{
                BunsenActivation,
                BunsenActivationConfig,
            },
            norm::{
                BunsenNormalization,
                BunsenNormalizationConfig,
            },
        },
        burner::module::DTypeMapper,
        support::testing::{
            assert_tensors_close,
            cpu_device,
            seeded_tensor,
        },
    };

    fn parts(device: &burn::prelude::Device) -> (BunsenNormalization, BunsenActivation) {
        let norm =
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(6).with_epsilon(1e-6)).init(device);
        (norm, BunsenActivationConfig::Silu.init(device))
    }

    /// `norm(x) ⊙ silu(z)`, as written out.
    #[test]
    fn test_gate() {
        let device = cpu_device();
        let (norm, silu_act) = parts(&device);
        let gate = Gate::new(&norm, &silu_act).with_widening(Widening::F32);
        assert_eq!(gate.widening(), Widening::F32);
        let x = seeded_tensor::<3>(1, [2, 4, 6], Distribution::Default, &device);
        let z = seeded_tensor::<3>(2, [2, 4, 6], Distribution::Default, &device);
        let expected = norm.forward(x.clone()) * silu(z.clone());
        assert_tensors_close(&gate.forward(x, z), &expected, Tolerance::default());
    }

    /// A bf16 input: the norm's output, widened, meets the gate in f32, and
    /// the product is narrowed once; unwidened, it is all bf16.
    #[test]
    fn test_widening() {
        let device = cpu_device();
        let (norm, silu_act) = parts(&device);
        let x = seeded_tensor::<2>(3, [3, 6], Distribution::Default, &device).cast(DType::BF16);
        let z = seeded_tensor::<2>(4, [3, 6], Distribution::Default, &device).cast(DType::BF16);
        let y = norm.forward(x.clone());

        let widened = Gate::new(&norm, &silu_act)
            .with_widening(Widening::F32)
            .forward(x.clone(), z.clone());
        let expected =
            (y.clone().cast(DType::F32) * silu(z.clone().cast(DType::F32))).cast(DType::BF16);
        assert_eq!(widened.dtype(), DType::BF16);
        widened.into_data().assert_eq(&expected.into_data(), true);

        let narrow = Gate::new(&norm, &silu_act).forward(x, z.clone());
        narrow
            .into_data()
            .assert_eq(&(y * silu(z)).into_data(), true);
    }

    /// Its dtypes are the gated module's.
    #[test]
    fn test_dtypes() {
        let device = cpu_device();
        let (norm, silu_act) = parts(&device);
        let norm = norm.map(&mut DTypeMapper::new(DType::BF16));
        let gate = Gate::new(&norm, &silu_act);
        assert_eq!(gate.preferred_dtype(), FloatDType::BF16);
        assert!(gate.accepts_dtype(FloatDType::F32));
        assert_eq!(gate.result_dtype(FloatDType::F16), FloatDType::F16);
        assert!(std::ptr::eq(gate.a(), &norm));
        assert!(std::ptr::eq(gate.b(), &silu_act));
    }

    #[test]
    #[should_panic(expected = "a gate has its gated output's shape")]
    fn test_gate_shape() {
        let device = cpu_device();
        let (norm, silu_act) = parts(&device);
        let _ = Gate::new(&norm, &silu_act).forward(
            Tensor::<2>::zeros([2, 6], &device),
            Tensor::zeros([3, 6], &device),
        );
    }
}
