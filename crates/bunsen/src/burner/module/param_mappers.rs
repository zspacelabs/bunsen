//! Parameter load/save mappers.
//!
//! `burn` attaches transformations to a [`Param`] itself, so they run whenever
//! the parameter is loaded from or saved to a store. [`LinearConfig`] uses this
//! for [`LinearLayout::Col`](burn::nn::LinearLayout::Col): the weight is held
//! in `PyTorch`'s `[d_output, d_input]` orientation and transposed on the way
//! through.
//!
//! These helpers apply the same trick to a parameter that has already been
//! built, for the case where the enclosing module does not expose a layout
//! knob.
//!
//! [`LinearConfig`]: burn::nn::LinearConfig

use burn::{
    Tensor,
    module::Param,
    prelude::Backend,
};

/// Repairs a 2-D weight mangled by `burn-store`'s stride-blind `PyTorch` read.
///
/// See [`repro::pytorch_strided_weights`](crate::burner::repro::pytorch_strided_weights)
/// for the defect. In short: `PyTorch` may store a `[R, C]` tensor as a
/// column-major view (strides `(1, R)`), and `PytorchStore` reads the raw
/// storage as if it were row-major. Every `Linear` weight in an `OpenAI`
/// Whisper checkpoint is stored that way.
///
/// The corruption is invertible. Reading storage that actually holds `Wᵀ` as
/// `[R, C]` yields `S = reshape(flat(Wᵀ), [R, C])`, and the adapter then
/// transposes it to `T = Sᵀ`. Since `flat(Sᵀᵀ) = flat(S) = flat(Wᵀ)`,
/// transposing back and reshaping to the parameter's own shape recovers `Wᵀ`,
/// which is what a row-major [`Linear`](burn::nn::Linear) wants.
///
/// For a square weight the reshape is a no-op and this degenerates to a plain
/// transpose — which is why square projections appear merely "untransposed"
/// while non-square ones come out scrambled.
///
/// The parameter's value and shape are untouched; only what crosses the store
/// boundary is repaired.
///
/// # Example
///
/// ```rust,ignore
/// let mut attn = mha_config.init(device);
/// attn.query.weight = repair_pytorch_strided_weight(attn.query.weight);
/// ```
pub fn repair_pytorch_strided_weight<B: Backend>(
    param: Param<Tensor<B, 2>>
) -> Param<Tensor<B, 2>> {
    param
        // Coming from the store: undo the mangling.
        .load_mapper(|tensor: Tensor<B, 2>| {
            B::sync(&tensor.device()).unwrap();
            let dims = tensor.dims();
            let tensor = tensor.transpose().reshape(dims);
            B::sync(&tensor.device()).unwrap();
            tensor
        })
        // Going back out: re-apply it, so a round trip is the identity.
        .save_mapper(|tensor: Tensor<B, 2>| {
            B::sync(&tensor.device()).unwrap();
            let [rows, cols] = tensor.dims();
            let tensor = tensor.reshape([cols, rows]).transpose();
            B::sync(&tensor.device()).unwrap();
            tensor
        })
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::LinearConfig,
        prelude::TensorData,
        tensor::{
            Tolerance,
            backend::BackendTypes,
        },
    };

    use super::*;
    use crate::{
        burner::tensor::TensorElemOpExt,
        support::testing::CpuBackend,
    };

    type B = CpuBackend;
    type F = <B as BackendTypes>::FloatElem;

    /// The mapper must leave the live parameter alone — it only changes what
    /// crosses the store boundary.
    #[test]
    fn test_repair_preserves_the_live_value() {
        let device = Default::default();

        let linear = LinearConfig::new(3, 5).with_bias(false).init::<B>(&device);
        let before = linear.weight.val();

        let mapped = repair_pytorch_strided_weight(linear.weight);

        assert_eq!(mapped.val().dims(), [3, 5]);
        mapped
            .val()
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&before.to_data_as::<F>(), Tolerance::default());
    }

    /// A round trip through the mappers is the identity, so saving what was
    /// loaded reproduces the store's orientation.
    #[test]
    fn test_load_then_save_round_trips() {
        let device = Default::default();

        // A store-orientation tensor: `[d_output, d_input]`.
        let stored: Tensor<B, 2> = Tensor::from_data(
            TensorData::new((0..15).map(|v| v as f64).collect::<Vec<_>>(), [5, 3]),
            &device,
        );

        let transposed = stored.clone().transpose();
        assert_eq!(transposed.dims(), [3, 5]);

        // Loading transposes into compute orientation, saving transposes back.
        transposed
            .transpose()
            .to_data_as::<F>()
            .assert_approx_eq::<F>(&stored.to_data_as::<F>(), Tolerance::default());
    }
}
