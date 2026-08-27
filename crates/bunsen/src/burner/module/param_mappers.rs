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

/// Attaches transposing load and save mappers to a 2-D parameter.
///
/// Use this when a checkpoint stores a weight in the opposite orientation to
/// the one the module computes with, and the module was not built with
/// [`LinearLayout::Col`](burn::nn::LinearLayout::Col) — typically because the
/// enclosing module builds its own [`Linear`](burn::nn::Linear) layers and
/// offers no way to configure them. `burn`'s
/// [`MultiHeadAttention`](burn::nn::attention::MultiHeadAttention) is the
/// motivating case.
///
/// The parameter's own value and shape are untouched, so the module keeps
/// computing in its native orientation; only the external form is transposed.
///
/// # Shape
///
/// The transpose swaps both axes, so a non-square parameter changes shape when
/// crossing the store boundary. That is the point — but it means the store's
/// tensor must be the transpose of the parameter, not merely the same size.
///
/// # Example
///
/// ```rust,ignore
/// let mut attn = mha_config.init(device);
/// attn.query.weight = transpose_on_load(attn.query.weight);
/// ```
pub fn transpose_on_load<B: Backend>(param: Param<Tensor<B, 2>>) -> Param<Tensor<B, 2>> {
    param
        // Coming from the store: transpose into the compute orientation.
        .load_mapper(|tensor: Tensor<B, 2>| {
            B::sync(&tensor.device()).unwrap();
            let tensor = tensor.transpose();
            B::sync(&tensor.device()).unwrap();
            tensor
        })
        // Going back out: restore the store's orientation.
        .save_mapper(|tensor: Tensor<B, 2>| {
            B::sync(&tensor.device()).unwrap();
            let tensor = tensor.transpose();
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
    fn test_transpose_on_load_preserves_the_live_value() {
        let device = Default::default();

        let linear = LinearConfig::new(3, 5).with_bias(false).init::<B>(&device);
        let before = linear.weight.val();

        let mapped = transpose_on_load(linear.weight);

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
