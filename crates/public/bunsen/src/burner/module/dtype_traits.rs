//! A block's float dtypes: what it computes in, takes, and gives.

use burn::{
    module::{
        Module,
        ModuleVisitor,
        Param,
    },
    prelude::*,
    tensor::{
        DType,
        FloatDType,
    },
};

/// The float dtype a block wants its float inputs in, and the ones it takes
/// as they are.
///
/// burn does not cast a float op's operands to one dtype, and on the cubecl
/// backends an op on mixed float dtypes runs in the left operand's dtype and
/// is wrong ([`burner::dtypes`](crate::burner::dtypes)). So a block says which
/// float dtype it computes in, and the code that calls it casts to that:
///
/// - [`InputDTypes`]: the float dtype a block wants its float inputs in
///   ([`preferred_dtype`](InputDTypes::preferred_dtype)), the dtypes it takes
///   as they are ([`accepts_dtype`](InputDTypes::accepts_dtype)), and the cast
///   that makes an input one of them
///   ([`coerce_input`](InputDTypes::coerce_input)). One dtype for all of a
///   block's float inputs.
/// - [`CoercesInput`]: a marker that the block's forward coerces its own
///   inputs, so a caller may pass any float dtype.
///   `support::testing::assert_coerces_input` (under the `testing` feature)
///   checks it.
/// - [`ResultDType`]: the float dtype a block's float output is in, given its
///   input's: a fixed dtype, or the input's own.
///
/// Which blocks are wrapped is the block author's decision, made when the
/// block is written: a field holds either the block itself, when it
/// implements the traits, or a [`DTypeWrapped<M>`] around it, which casts the
/// input to the module's dtype, calls it, and casts the result back to the
/// caller's dtype.
pub trait InputDTypes {
    /// The float dtype the block computes in, and casts its inputs to.
    fn preferred_dtype(&self) -> FloatDType;

    /// Whether the block takes a float input of `dtype` as it is; by default,
    /// only the [`preferred_dtype`](Self::preferred_dtype).
    fn accepts_dtype(
        &self,
        dtype: FloatDType,
    ) -> bool {
        dtype == self.preferred_dtype()
    }

    /// `x` as the block takes it: as it is when its dtype is accepted, else
    /// cast to the [`preferred_dtype`](Self::preferred_dtype). A no-op on an
    /// input that was cast already.
    fn coerce_input<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D>
    where
        Self: Sized,
    {
        if self.accepts_dtype(FloatDType::from(x.dtype())) {
            x
        } else {
            x.cast(DType::from(self.preferred_dtype()))
        }
    }
}

/// A marker: the block's forward coerces its own float inputs
/// ([`InputDTypes::coerce_input`]), so a caller may pass any float dtype.
pub trait CoercesInput: InputDTypes {}

/// The float dtype a block's float output is in, given its input's.
pub trait ResultDType {
    /// The output's float dtype for an input of `input`: fixed, or `input`
    /// itself for a block that passes it through.
    fn result_dtype(
        &self,
        input: FloatDType,
    ) -> FloatDType;
}

/// Asserts `block` takes a float input of `dtype` as it is: the gated
/// contract at a block's seam, run under debug assertions and tests. burn
/// does not catch a missed cast, and on the cubecl backends the mixed op it
/// leads to runs and is wrong.
///
/// # Panics
/// If `block` does not [accept](InputDTypes::accepts_dtype) `dtype`; the
/// message names `what`.
#[track_caller]
pub fn assert_input_dtype<B: InputDTypes + ?Sized>(
    block: &B,
    dtype: DType,
    what: &str,
) {
    let dtype = FloatDType::from(dtype);
    assert!(
        block.accepts_dtype(dtype),
        "{what}: a {dtype:?} input to a block that computes in {:?}; cast it first",
        block.preferred_dtype()
    );
}

/// The float dtype of `module`'s parameters, from its first float
/// parameter; `None` for a module with none.
pub fn module_float_dtype<M: Module>(module: &M) -> Option<FloatDType> {
    #[derive(Default)]
    struct First(Option<FloatDType>);

    impl ModuleVisitor for First {
        fn visit_float<const D: usize>(
            &mut self,
            param: &Param<Tensor<D>>,
        ) {
            if self.0.is_none() {
                self.0 = Some(FloatDType::from(param.val().dtype()));
            }
        }
    }

    let mut first = First::default();
    module.visit(&mut first);
    first.0
}

/// A module that computes in its own float dtype, called from code in
/// another: [`call`](Self::call) casts the input to the module's dtype, runs
/// it, and casts the result back to the input's.
///
/// Its dtype is its module's parameters' ([`module_float_dtype`]), or, for a
/// module whose dtype its parameters don't say (none of them, or a mixed
/// set), the one it was given ([`with_dtype`](Self::with_dtype)).
#[derive(Module, Debug)]
pub struct DTypeWrapped<M: Module> {
    /// The wrapped module.
    pub inner: M,

    /// The dtype the module computes in, when its parameters don't say.
    #[module(skip)]
    pub dtype: Option<FloatDType>,
}

impl<M: Module> DTypeWrapped<M> {
    /// `inner`, computing in its parameters' dtype.
    pub fn new(inner: M) -> Self {
        Self { inner, dtype: None }
    }

    /// `inner`, computing in `dtype`.
    pub fn with_dtype(
        inner: M,
        dtype: FloatDType,
    ) -> Self {
        Self {
            inner,
            dtype: Some(dtype),
        }
    }

    /// The dtype the module computes in.
    ///
    /// # Panics
    /// If it was given none and its module has no float parameter.
    pub fn dtype(&self) -> FloatDType {
        self.dtype
            .or_else(|| module_float_dtype(&self.inner))
            .expect("a wrapped module with no float parameter needs `with_dtype`")
    }

    /// `forward` on the module, its input `x` cast to the module's dtype and
    /// its result cast back to `x`'s.
    pub fn call<const D: usize, const E: usize>(
        &self,
        x: Tensor<D>,
        forward: impl FnOnce(&M, Tensor<D>) -> Tensor<E>,
    ) -> Tensor<E> {
        let caller = x.dtype();
        let x = x.cast(DType::from(self.dtype()));
        forward(&self.inner, x).cast(caller)
    }
}

impl<M: Module> InputDTypes for DTypeWrapped<M> {
    fn preferred_dtype(&self) -> FloatDType {
        self.dtype()
    }

    /// Every float dtype: [`call`](Self::call) casts to the module's and back.
    fn accepts_dtype(
        &self,
        _dtype: FloatDType,
    ) -> bool {
        true
    }
}

impl<M: Module> CoercesInput for DTypeWrapped<M> {}

impl<M: Module> ResultDType for DTypeWrapped<M> {
    /// The input's: [`call`](Self::call) casts the result back to it.
    fn result_dtype(
        &self,
        input: FloatDType,
    ) -> FloatDType {
        input
    }
}

#[cfg(test)]
mod tests {
    use burn::nn::{
        Linear,
        LinearConfig,
    };

    use super::*;
    use crate::{
        burner::module::DTypeMapper,
        support::testing::cpu_device,
    };

    /// A linear layer in bf16, on a device whose default float is f32.
    fn bf16_linear(device: &Device) -> Linear {
        LinearConfig::new(3, 2)
            .init(device)
            .map(&mut DTypeMapper::new(DType::BF16))
    }

    #[test]
    fn test_module_float_dtype() {
        let device = cpu_device();
        assert_eq!(
            module_float_dtype(&LinearConfig::new(3, 2).init(&device)),
            Some(FloatDType::F32)
        );
        assert_eq!(
            module_float_dtype(&bf16_linear(&device)),
            Some(FloatDType::BF16)
        );
    }

    /// The wrapper runs its module in the module's dtype, between casts of
    /// the caller's: f32 in, bf16 inside, f32 out, equal to the bf16 run.
    #[test]
    fn test_wrapped_call_casts_in_and_out() {
        let device = cpu_device();
        let wrapped = DTypeWrapped::new(bf16_linear(&device));
        assert_eq!(wrapped.preferred_dtype(), FloatDType::BF16);
        assert!(wrapped.accepts_dtype(FloatDType::F32));
        assert_eq!(wrapped.result_dtype(FloatDType::F32), FloatDType::F32);

        let x = Tensor::<2>::from_data([[0.5f32, -1.0, 2.0]], (&device, DType::F32));
        let mut seen = None;
        let y = wrapped.call(x.clone(), |m, x| {
            seen = Some(x.dtype());
            m.forward(x)
        });
        assert_eq!(seen, Some(DType::BF16), "the module runs in bf16");
        assert_eq!(y.dtype(), DType::F32, "the result is the caller's dtype");
        let direct = wrapped.inner.forward(x.cast(DType::BF16)).cast(DType::F32);
        y.into_data().assert_eq(&direct.into_data(), true);
    }

    #[test]
    fn test_with_dtype_and_coerce() {
        let device = cpu_device();
        let wrapped =
            DTypeWrapped::with_dtype(LinearConfig::new(3, 2).init(&device), FloatDType::F16);
        assert_eq!(wrapped.dtype(), FloatDType::F16);

        struct Wants16;
        impl InputDTypes for Wants16 {
            fn preferred_dtype(&self) -> FloatDType {
                FloatDType::F16
            }
        }
        let x = Tensor::<1>::from_data([1.0f32, 2.0], (&device, DType::F32));
        assert_eq!(Wants16.coerce_input(x.clone()).dtype(), DType::F16);
        let already = x.cast(DType::F16);
        assert_eq!(Wants16.coerce_input(already).dtype(), DType::F16);
    }

    /// The seam's contract passes an accepted dtype and refuses another,
    /// naming the block.
    #[test]
    fn test_assert_input_dtype() {
        struct Wants16;
        impl InputDTypes for Wants16 {
            fn preferred_dtype(&self) -> FloatDType {
                FloatDType::F16
            }
        }
        assert_input_dtype(&Wants16, DType::F16, "a block");
        let refused =
            std::panic::catch_unwind(|| assert_input_dtype(&Wants16, DType::F32, "a block"));
        let message = refused.unwrap_err();
        let message = message.downcast_ref::<String>().unwrap();
        assert!(message.contains("a block: a F32 input"), "{message}");
    }
}
