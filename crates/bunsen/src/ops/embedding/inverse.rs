use burn::{
    Tensor,
    nn::Embedding,
    prelude::Backend,
};

/// Argument to [unembed].
pub trait EmbeddingArg<B: Backend> {
    /// The embedding layer's weight.
    fn weight(&self) -> Tensor<B, 2>;
}

impl<B: Backend> EmbeddingArg<B> for &Embedding<B> {
    fn weight(&self) -> Tensor<B, 2> {
        self.weight.val()
    }
}

impl<B: Backend> EmbeddingArg<B> for Tensor<B, 2> {
    fn weight(&self) -> Tensor<B, 2> {
        self.clone()
    }
}

/// Inverts an Embedding layer to get logits.
///
/// # Arguments
/// * `emb` - The `[n_vocab, d_model]`, embedding layer to invert.
/// * `x` - The `[batch, seq_len, d_model]` tensor to unembed.
///
/// # Returns
/// The `[batch, seq_len, n_vocab]` logits tensor.
pub fn unembed<B: Backend, A: EmbeddingArg<B>>(
    emb: A,
    x: Tensor<B, 3>,
) -> Tensor<B, 3> {
    x.matmul(emb.weight().transpose().unsqueeze::<3>())
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Int,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        ops::embedding::identity_embedding,
        support::testing::PerformanceBackend,
    };

    #[test]
    #[serial]
    fn test_embedding_inverse_to_logits() {
        type B = PerformanceBackend;
        let device = Default::default();

        let n_embedding = 10;

        let embedding = identity_embedding(n_embedding, &device);

        let batch = 2;
        let seq_len = 20;

        let x: Tensor<B, 2, Int> = Tensor::random(
            [batch, seq_len],
            Distribution::Uniform(0.0, n_embedding as f64),
            &device,
        );

        let e = embedding.forward(x.clone());

        let logits = unembed(&embedding, e.clone());
        let logits2 = unembed(embedding.weight.val(), e.clone());
        logits
            .clone()
            .into_data()
            .assert_eq(&logits2.into_data(), true);

        let y: Tensor<B, 2, Int> = logits.argmax(2).squeeze_dim(2);

        y.into_data().assert_eq(&x.into_data(), true);
    }
}
