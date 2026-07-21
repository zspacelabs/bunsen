use burn::{
    Tensor,
    prelude::{
        Backend,
        TensorData,
    },
};

/// Trait defining an interface for building sampling windows.
pub trait SamplingWindowBuilder {
    /// Materialize a vector window of width `win_len`.
    fn to_vec_window(
        &self,
        size: usize,
    ) -> Vec<f32>;

    /// Materialize a tensor window of width `win_len`.
    fn to_tensor_window<B: Backend>(
        &self,
        size: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        Tensor::from_data(
            TensorData::from(self.to_vec_window(size).as_slice()),
            device,
        )
    }
}
