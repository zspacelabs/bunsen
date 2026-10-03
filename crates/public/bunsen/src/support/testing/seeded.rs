use burn::{
    prelude::{
        Backend,
        Tensor,
        TensorData,
    },
    tensor::Distribution,
};
use rand::{
    SeedableRng,
    rngs::StdRng,
};

/// A float tensor of seeded random values, identical on every backend.
///
/// The values are drawn on the host as `f64` from a [`StdRng`] seeded with
/// `seed`, then uploaded at the backend's float element. Unlike
/// `Tensor::random`, the result does not depend on the backend's RNG or on
/// global seeding.
///
/// Use it for any input that two runs must share: a comparison across
/// backends, a stored baseline, a golden value. A backend whose float element
/// is narrower than `f64` sees the same draws, rounded to its element.
///
/// ```
/// use bunsen::support::testing::{
///     CpuBackend,
///     backend_device,
///     seeded_tensor,
/// };
/// use burn::tensor::Distribution;
///
/// type B = CpuBackend;
/// let device = backend_device::<B>();
/// let a = seeded_tensor::<B, 2>(7, [2, 3], Distribution::Default, &device);
/// let b = seeded_tensor::<B, 2>(7, [2, 3], Distribution::Default, &device);
/// a.into_data().assert_eq(&b.into_data(), true);
/// ```
pub fn seeded_tensor<B: Backend, const D: usize>(
    seed: u64,
    shape: [usize; D],
    distribution: Distribution,
    device: &B::Device,
) -> Tensor<B, D> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data = TensorData::random::<f64, _, _>(shape, distribution, &mut rng);
    Tensor::from_data(data.convert::<B::FloatElem>(), device)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Element,
        backend::BackendTypes,
    };

    use super::*;
    use crate::support::testing::{
        CpuBackend,
        backend_device,
    };

    fn host<const D: usize>(tensor: &Tensor<CpuBackend, D>) -> Vec<f64> {
        tensor.to_data().convert::<f64>().into_vec().unwrap()
    }

    #[test]
    fn test_seeded_tensor_is_deterministic() {
        type B = CpuBackend;
        let device = backend_device::<B>();
        let dist = Distribution::Uniform(-1.0, 1.0);
        let a = seeded_tensor::<B, 3>(7, [2, 3, 4], dist, &device);
        let b = seeded_tensor::<B, 3>(7, [2, 3, 4], dist, &device);
        let c = seeded_tensor::<B, 3>(8, [2, 3, 4], dist, &device);
        assert_eq!(
            a.dtype(),
            <<B as BackendTypes>::FloatElem as Element>::dtype()
        );
        assert_eq!(host(&a), host(&b));
        assert_ne!(host(&a), host(&c));
        assert!(host(&a).iter().all(|v| (-1.0..1.0).contains(v)));
    }
}
