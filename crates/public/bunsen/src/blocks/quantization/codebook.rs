//! [`EuclideanCodebook`]: a codebook kept as running sums.

use burn::{
    module::Param,
    prelude::*,
};

use crate::{
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::BunsenResult,
};

/// Common meta for [`EuclideanCodebook`] and [`EuclideanCodebookConfig`].
pub trait EuclideanCodebookMeta {
    /// The codes, `S`.
    fn size(&self) -> usize;

    /// A code's vector's width, `d`.
    fn dim(&self) -> usize;

    /// The floor a code's usage is clamped to before it divides.
    fn epsilon(&self) -> f64;
}

/// Config for [`EuclideanCodebook`].
#[derive(Config, Debug)]
pub struct EuclideanCodebookConfig {
    /// The codes, `S`.
    pub size: usize,

    /// A code's vector's width, `d`.
    pub dim: usize,

    /// The floor a code's usage is clamped to before it divides.
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl EuclideanCodebookMeta for EuclideanCodebookConfig {
    fn size(&self) -> usize {
        self.size
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

/// The codebook: sums of zeros and usages of one (every code the zero
/// vector), as a codebook before training.
impl ModuleInit<EuclideanCodebook> for EuclideanCodebookConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<EuclideanCodebook> {
        Ok(EuclideanCodebook {
            cluster_usage: Param::from_tensor(Tensor::ones([self.size], device)),
            embedding_sum: Param::from_tensor(Tensor::zeros([self.size, self.dim], device)),
            epsilon: self.epsilon,
        })
    }
}

/// A codebook kept as the running sums of an exponential-moving-average
/// k-means (Encodec's, Mimi's and Qwen3-TTS's codecs'): each code's summed
/// vectors, `embedding_sum` `[S, d]`, and its usage, `cluster_usage` `[S]`.
/// A code's vector is their quotient, the usage clamped below at `ε`:
///
/// ```text
/// vector[s] = embedding_sum[s] / max(cluster_usage[s], ε)
/// ```
///
/// The parameters keep the checkpoints' names.
///
/// Built by [`EuclideanCodebookConfig`].
#[derive(Module, Debug)]
pub struct EuclideanCodebook {
    /// Each code's usage, `[S]`.
    pub cluster_usage: Param<Tensor<1>>,

    /// Each code's summed vectors, `[S, d]`.
    pub embedding_sum: Param<Tensor<2>>,

    /// The floor a usage is clamped to before it divides.
    pub epsilon: f64,
}

impl EuclideanCodebookMeta for EuclideanCodebook {
    fn size(&self) -> usize {
        self.embedding_sum.dims()[0]
    }

    fn dim(&self) -> usize {
        self.embedding_sum.dims()[1]
    }

    fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl EuclideanCodebook {
    /// The codes' vectors, `[S, d]`.
    pub fn vectors(&self) -> Tensor<2> {
        let usage = self.cluster_usage.val().clamp_min(self.epsilon);
        self.embedding_sum.val() / usage.unsqueeze_dim(1)
    }

    /// Codes to their vectors.
    ///
    /// # Arguments
    /// - `codes`: `[B, T]`, each in `0..S`.
    ///
    /// # Returns
    /// `[B, d, T]`, channels first.
    pub fn decode(
        &self,
        codes: Tensor<2, Int>,
    ) -> Tensor<3> {
        let [b, t] = unpack_shape_contract!(["B", "T"], &codes.dims(), &["B", "T"]);
        let d = self.dim();
        self.vectors()
            .select(0, codes.reshape([b * t]))
            .reshape([b, t, d])
            .swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use super::*;
    use crate::support::testing::cpu_device;

    /// The config and the module built from it answer the meta alike.
    #[test]
    fn test_meta() {
        let config = EuclideanCodebookConfig::new(5, 3);
        let codebook = config.init(&cpu_device());
        assert_eq!(
            (codebook.size(), codebook.dim(), codebook.epsilon()),
            (config.size(), config.dim(), config.epsilon())
        );
        assert_eq!((config.size, config.dim, config.epsilon), (5, 3, 1e-5));
    }

    /// A code's vector is its sum over its usage, the usage clamped at `ε`.
    #[test]
    fn test_decode() {
        let device = cpu_device();
        let mut codebook = EuclideanCodebookConfig::new(3, 2).init(&device);
        codebook.embedding_sum = Param::from_tensor(Tensor::from_floats(
            [[1.0, 2.0], [3.0, -6.0], [2e-6, 4e-6]],
            &device,
        ));
        codebook.cluster_usage = Param::from_tensor(Tensor::from_floats([0.5, 3.0, 1e-9], &device));
        let codes = Tensor::<2, Int>::from_ints([[2, 0, 1], [1, 1, 0]], &device);
        codebook.decode(codes).into_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [[0.2, 2.0, 1.0], [0.4, 4.0, -2.0]],
                [[1.0, 1.0, 2.0], [-2.0, -2.0, 4.0]],
            ]),
            burn::tensor::Tolerance::rel_abs(1e-6, 1e-6),
        );
    }
}
