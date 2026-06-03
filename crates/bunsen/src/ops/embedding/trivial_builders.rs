use burn::{
    module::Param,
    nn::Embedding,
    tensor::{
        Int,
        Tensor,
        backend::Backend,
    },
};

/// Build an iota embedding.
///
/// weight[i, j] = i * d + j  — every row distinct, lookups hand-verifiable.
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

/// Build a one-hot passthrough embedding.
///
/// Square identity (`num_embeddings` == dim == n): embedding acts as a one-hot
/// passthrough.
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
    use crate::support::testing::PerformanceBackend;

    #[test]
    fn test_iota_embedding() {
        let device = Default::default();
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
    fn test_identity_embedding() {
        let device = Default::default();
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
