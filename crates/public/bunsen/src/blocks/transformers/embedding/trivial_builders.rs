//! Fixture embeddings for tests.

use burn::{
    module::Param,
    nn::Embedding,
    tensor::{
        Int,
        Tensor,
        backend::Backend,
    },
};

/// Builds an iota embedding: a fixture whose lookups can be read by hand.
///
/// `weight[i, j] = i * d + j`, so every row is distinct and the value at any
/// position names the token and column it came from. This is for tests and
/// debugging (checking a gather, a slice, or an embedding-weight mapping), not
/// for training: the weights are a plain counting pattern.
///
/// # Arguments
///
/// - `n`: the number of embeddings (the vocabulary size).
/// - `d`: the embedding width.
/// - `device`: the target device.
///
/// # Returns
///
/// An [`Embedding`] with a `[n, d]` weight.
pub fn iota_embedding<B: Backend>(
    n: usize,
    d: usize,
    device: &B::Device,
) -> Embedding<B> {
    let weight = Tensor::<B, 1, Int>::arange(0..(n * d) as i64, device)
        .float()
        .reshape([n, d]);
    Embedding {
        weight: Param::from_tensor(weight),
    }
}

/// Builds a one-hot passthrough embedding: a fixture for round trips.
///
/// The weight is the `[n, n]` identity, so token `i` embeds as the one-hot
/// vector `e_i`, and [`unembed`] through the same weight gives logits whose
/// `argmax` is the original token. Like [`iota_embedding`], this is for tests
/// and debugging, not training.
///
/// # Arguments
///
/// - `n`: the number of embeddings, which is also the embedding width.
/// - `device`: the target device.
///
/// # Returns
///
/// An [`Embedding`] with a `[n, n]` identity weight.
///
/// [`unembed`]: crate::ops::embedding::unembed
pub fn identity_embedding<B: Backend>(
    n: usize,
    device: &B::Device,
) -> Embedding<B> {
    Embedding {
        weight: Param::from_tensor(Tensor::<B, 2>::eye(n, device)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        PerformanceBackend,
        performance_device,
    };

    #[test]
    #[serial_test::serial]
    fn test_iota_embedding() {
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let n = 3;
        let d = 4;
        let emb = iota_embedding::<PerformanceBackend>(n, d, &device);

        let weight = emb.weight.val();
        let data = weight.to_data();

        // Verify shape
        assert_eq!(data.shape, [n, d].into());

        // Verify values: weight[i, j] = i * d + j
        for i in 0..n {
            for j in 0..d {
                let expected = (i * d + j) as f32;
                let actual = data.iter::<f32>().nth(i * d + j).unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_identity_embedding() {
        type B = PerformanceBackend;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let n = 5;
        let emb = identity_embedding::<PerformanceBackend>(n, &device);

        let weight = emb.weight.val();
        let data = weight.to_data();

        // Verify shape
        assert_eq!(data.shape, [n, n].into());

        // Verify identity matrix: 1.0 on diagonal, 0.0 elsewhere
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = data.iter::<f32>().nth(i * n + j).unwrap();
                assert_eq!(actual, expected);
            }
        }
    }
}
