//! The dtype traits' shared test helper.

use burn::{
    prelude::*,
    tensor::{
        DType,
        FloatDType,
    },
};

use crate::burner::module::{
    CoercesInput,
    ResultDType,
};

/// Asserts `block` coerces its own float input ([`CoercesInput`]): `forward`
/// on `input` at an off dtype gives what it gives at the block's
/// [`preferred_dtype`](crate::burner::module::InputDTypes::preferred_dtype),
/// to within `atol`, each result in the dtype the block's [`ResultDType`]
/// says for its input.
///
/// The off dtype is bf16 for a block that prefers f32 or f64, and f32 for one
/// that prefers a half dtype. The input is rounded through the half dtype
/// first, so both runs see the same values; a block that coerces then runs
/// the same computation twice. Build the block and the input on the CPU
/// device ([`cpu_device`](super::cpu_device)), which computes in every float
/// dtype, so the off run is meaningful wherever the suite runs.
///
/// # Panics
/// If the two results differ by more than `atol` anywhere, or either is not
/// in the dtype [`ResultDType::result_dtype`] gives.
pub fn assert_coerces_input<B: CoercesInput + ResultDType, const D: usize, const E: usize>(
    block: &B,
    input: Tensor<D>,
    forward: impl Fn(&B, Tensor<D>) -> Tensor<E>,
    atol: f64,
) {
    let preferred = block.preferred_dtype();
    let half = match preferred {
        FloatDType::F16 | FloatDType::BF16 => preferred,
        _ => FloatDType::BF16,
    };
    let off = if half == preferred {
        FloatDType::F32
    } else {
        FloatDType::BF16
    };
    let x = input.cast(DType::from(half));
    let at_preferred = forward(block, x.clone().cast(DType::from(preferred)));
    let at_off = forward(block, x.cast(DType::from(off)));
    for (input, result) in [(preferred, &at_preferred), (off, &at_off)] {
        assert_eq!(
            result.dtype(),
            DType::from(block.result_dtype(input)),
            "the result of a {input:?} input"
        );
    }
    let gap = (at_preferred.cast(DType::F32) - at_off.cast(DType::F32))
        .abs()
        .max()
        .into_scalar::<f32>();
    assert!(
        f64::from(gap) <= atol,
        "a {off:?} input is {gap} from a {preferred:?} one, over {atol}: the block does not \
         coerce its input"
    );
}

#[cfg(test)]
mod tests {
    use burn::nn::LinearConfig;

    use super::*;
    use crate::{
        burner::module::{
            DTypeMapper,
            DTypeWrapped,
        },
        support::testing::cpu_device,
    };

    /// A wrapped bf16 layer coerces; the bare layer fed an f32 input does not
    /// (burn-flex refuses the mixed op).
    #[test]
    fn test_assert_coerces_input() {
        let device = cpu_device();
        let linear = LinearConfig::new(3, 2)
            .init(&device)
            .map(&mut DTypeMapper::new(DType::BF16));
        let x = Tensor::<2>::from_data([[0.5f32, -1.0, 2.0]], (&device, DType::F32));
        let wrapped = DTypeWrapped::new(linear.clone());
        assert_coerces_input(
            &wrapped,
            x.clone(),
            |b, x| b.call(x, |m, x| m.forward(x)),
            0.0,
        );

        struct Bare(burn::nn::Linear);
        impl crate::burner::module::InputDTypes for Bare {
            fn preferred_dtype(&self) -> FloatDType {
                FloatDType::BF16
            }
        }
        impl CoercesInput for Bare {}
        impl ResultDType for Bare {
            fn result_dtype(
                &self,
                _input: FloatDType,
            ) -> FloatDType {
                FloatDType::BF16
            }
        }
        let bare = Bare(linear);
        let refused = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
            assert_coerces_input(&bare, x, |b, x| b.0.forward(x), 0.0)
        }));
        assert!(
            refused.is_err(),
            "a block that does not coerce fails the check"
        );
    }
}
