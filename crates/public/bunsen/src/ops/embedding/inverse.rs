use burn::{
    Tensor,
    nn::Embedding,
};

/// Argument to [unembed].
pub trait EmbeddingArg {
    /// The embedding layer's weight.
    fn weight(&self) -> Tensor<2>;
}

impl EmbeddingArg for &Embedding {
    fn weight(&self) -> Tensor<2> {
        self.weight.val()
    }
}

impl EmbeddingArg for Tensor<2> {
    fn weight(&self) -> Tensor<2> {
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
pub fn unembed<A: EmbeddingArg>(
    emb: A,
    x: Tensor<3>,
) -> Tensor<3> {
    x.matmul(emb.weight().transpose().unsqueeze::<3>())
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        tensor::{
            Distribution,
            Int,
        },
    };
    use serial_test::serial;

    use super::*;
    use crate::support::testing::{
        DeviceMemoryGuard,
        performance_device,
    };

    #[test]
    #[serial]
    fn test_embedding_inverse_to_logits() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let n_embedding = 10;

        // A one-hot passthrough, so each logit row peaks at its own token.
        let embedding = Embedding {
            weight: Param::from_tensor(Tensor::<2>::eye(n_embedding, &device)),
        };

        let batch = 2;
        let seq_len = 20;

        let x: Tensor<2, Int> = Tensor::random(
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

        let y: Tensor<2, Int> = logits.argmax(2).squeeze_dim(2);

        y.into_data().assert_eq(&x.into_data(), true);
    }
}
