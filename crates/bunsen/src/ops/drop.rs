//! # Drop Extensions

use burn::{
    Tensor,
    prelude::Backend,
    tensor::Distribution,
};

/// Applies the random dropout op on the input tensor.
///
/// ```ignore
/// p_keep = 1.0 - p;
/// input * input.random_like(Bernoulli(p_keep)) / p_keep
/// ```
///
/// ## Arguments
/// * `prob` - the drop probability.
/// * `input` - the input tensor.
pub fn dropout<B: Backend, const D: usize>(
    prob: f64,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    if prob == 0.0 {
        return input;
    }
    if !(0.0..=1.0).contains(&prob) {
        panic!("Dropout validators should be between 0 and 1, but got {prob}",);
    }

    let prob_keep = 1.0 - prob;
    let random = input.random_like(Distribution::Bernoulli(prob_keep));
    let x = input * random;

    x * (1.0 / prob_keep)
}

#[cfg(test)]
mod tests {
    use burn::prelude::ElementConversion;

    use super::*;
    use crate::support::testing::PerfTestBackend;

    #[test]
    fn dropout_prob_0_should_return_input() {
        type B = PerfTestBackend;
        let device = Default::default();
        let input = Tensor::<B, 2>::random([10, 3], Distribution::Default, &device);

        let output = dropout(0., input.clone());

        output.to_data().assert_eq(&input.to_data(), true);
    }

    #[test]
    fn dropout_rates_stochastic_test() {
        type B = PerfTestBackend;
        let device = Default::default();
        B::seed(&device, 0);

        let input = Tensor::<B, 2>::ones([10, 10], &device);
        let num_elem = input.shape().num_elements();

        let prob = 0.25;
        let output = dropout(prob, input.clone());

        let prob_keep = 1.0 - prob;
        let keep_value = 1.0 / prob_keep;

        let drop_count: f32 = output
            .clone()
            .equal_elem(0.0)
            .float()
            .sum()
            .into_scalar()
            .elem();
        let keep_count: f32 = output
            .clone()
            .equal_elem(keep_value)
            .float()
            .sum()
            .into_scalar()
            .elem();

        assert_eq!(keep_count, num_elem as f32 - drop_count);

        let drop_rate = drop_count / num_elem as f32;
        assert!((drop_rate - prob as f32).abs() < 0.1);
    }

    #[test]
    #[should_panic = "Dropout validators should be between 0 and 1,"]
    fn dropout_prob_invalid() {
        type B = PerfTestBackend;
        let device = Default::default();

        let input = Tensor::<B, 1>::ones([10], &device);
        let _ = dropout(-10., input);
    }
}
