use burn::{
    Tensor,
    prelude::{
        Backend,
        TensorData,
    },
    tensor::TensorCreationOptions,
};

/// Trait defining an interface for building sampling windows.
pub trait SamplingWindowBuilder {
    /// Materialize a vector window of width `win_len`.
    fn to_vec_window(
        &self,
        size: usize,
    ) -> Vec<f64>;

    /// Materialize a tensor window of width `win_len`.
    fn to_tensor_window<B: Backend>(
        &self,
        size: usize,
        options: impl Into<TensorCreationOptions<B>>,
    ) -> Tensor<B, 1> {
        Tensor::from_data(
            TensorData::from(self.to_vec_window(size).as_slice()),
            options,
        )
    }
}

/// Testing Support.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use burn::{
        prelude::{
            Backend,
            TensorData,
        },
        tensor::{
            DType,
            TensorCreationOptions,
            Tolerance,
        },
    };
    use tracing::debug;

    use crate::{
        ops::signal::SamplingWindowBuilder,
        support::testing::assert_close_to_vec,
    };

    /// [`SamplingWindowBuilder`] test helper.
    pub fn assert_builder_impls_match<B: Backend>(
        builder: &impl SamplingWindowBuilder,
        expected: &[f64],
        options: impl Into<TensorCreationOptions<B>>,
    ) {
        let size = expected.len();
        let options = options.into();

        debug!("checking to_vec_window");
        let vec_win = builder.to_vec_window(size);
        assert_close_to_vec(&vec_win, expected, 0.0001);

        debug!("checking to_tensor_window");
        let ten_win = builder.to_tensor_window(size, options);
        ten_win
            .cast(DType::F64)
            .to_data()
            .assert_approx_eq::<f64>(&TensorData::from(expected), Tolerance::default());
    }
}
