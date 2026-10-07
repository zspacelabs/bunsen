use burn::{
    prelude::{
        Tensor,
        TensorData,
    },
    tensor::{
        Device,
        Distribution,
    },
};
use rand::{
    SeedableRng,
    rngs::StdRng,
};

/// A float tensor of seeded random values, identical on every backend.
///
/// The values are drawn on the host as `f64` from a [`StdRng`] seeded with
/// `seed`, then uploaded at the device's default float dtype. Unlike
/// `Tensor::random`, the result does not depend on the backend's RNG or on
/// global seeding.
///
/// Use it for any input that two runs must share: a comparison across
/// backends, a stored baseline, a golden value. A device whose float dtype is
/// narrower than `f64` sees the same draws, rounded to it.
///
/// ```
/// use bunsen::support::testing::{
///     cpu_device,
///     seeded_tensor,
/// };
/// use burn::tensor::Distribution;
///
/// let device = cpu_device();
/// let a = seeded_tensor::<2>(7, [2, 3], Distribution::Default, &device);
/// let b = seeded_tensor::<2>(7, [2, 3], Distribution::Default, &device);
/// a.into_data().assert_eq(&b.into_data(), true);
/// ```
pub fn seeded_tensor<const D: usize>(
    seed: u64,
    shape: [usize; D],
    distribution: Distribution,
    device: &Device,
) -> Tensor<D> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data = TensorData::random::<f64, _, _>(shape, distribution, &mut rng);
    // `from_data` converts host data to the device's default float dtype.
    Tensor::from_data(data, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        burner::tensor::device_float_dtype,
        support::testing::cpu_device,
    };

    fn host<const D: usize>(tensor: &Tensor<D>) -> Vec<f64> {
        tensor.to_data().try_into_vec_as::<f64>().unwrap()
    }

    #[test]
    fn test_seeded_tensor_is_deterministic() {
        let device = cpu_device();
        let dist = Distribution::Uniform(-1.0, 1.0);
        let a = seeded_tensor::<3>(7, [2, 3, 4], dist, &device);
        let b = seeded_tensor::<3>(7, [2, 3, 4], dist, &device);
        let c = seeded_tensor::<3>(8, [2, 3, 4], dist, &device);
        assert_eq!(a.dtype(), device_float_dtype(&device));
        assert_eq!(host(&a), host(&b));
        assert_ne!(host(&a), host(&c));
        assert!(host(&a).iter().all(|v| (-1.0..1.0).contains(v)));
    }
}
