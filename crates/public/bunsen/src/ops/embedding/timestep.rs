//! Sinusoidal embeddings of a scalar.
use burn::{
    Tensor,
    prelude::TensorData,
    tensor::Device,
};

/// The sinusoidal embedding of one scalar, `t`: the cosines of `t` at every
/// frequency, then their sines.
///
/// Frequency `i`: `cos(t * f_i)` at column `i` and `sin(t * f_i)` at column
/// `channels / 2 + i`, with `f_i = exp(-i * ln(theta) / (channels / 2))`.
/// This is the timestep embedding of diffusion models (`guided-diffusion`'s
/// `timestep_embedding`), and the delay embedding Voxtral Realtime conditions
/// its decoder on (`transformers`' `VoxtralRealtimeTimeEmbedding`). Computed
/// in f32, as both compute it.
///
/// Unlike [`sinusoid_table`](super::sinusoid_table), which tabulates
/// positions, the cosines come first, and the frequencies divide by
/// `channels / 2`, not `channels / 2 - 1`.
///
/// # Returns
/// `[channels]`.
///
/// # Panics
/// If `channels` is odd or 0.
pub fn timestep_embedding(
    t: f32,
    channels: usize,
    theta: f32,
    device: &Device,
) -> Tensor<1> {
    assert!(
        channels.is_multiple_of(2) && channels > 0,
        "timestep_embedding: channels ({channels}) must be even and positive"
    );
    let half = channels / 2;
    let angles: Vec<f32> = (0..half)
        .map(|i| t * (-theta.ln() * i as f32 / half as f32).exp())
        .collect();
    let data: Vec<f32> = angles
        .iter()
        .map(|a| a.cos())
        .chain(angles.iter().map(|a| a.sin()))
        .collect();
    Tensor::from_data(TensorData::new(data, [channels]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_timestep_embedding() {
        let e = timestep_embedding(6.0, 8, 10_000.0, &cpu_device());
        let values: Vec<f32> = e.into_data().try_into_vec_as::<f32>().unwrap();
        let f: Vec<f64> = (0..4)
            .map(|i| (-(10_000f64).ln() * i as f64 / 4.0).exp())
            .collect();
        for i in 0..4 {
            assert!((values[i] as f64 - (6.0 * f[i]).cos()).abs() < 1e-6, "{i}");
            assert!(
                (values[4 + i] as f64 - (6.0 * f[i]).sin()).abs() < 1e-6,
                "{i}"
            );
        }
        // At t = 0 every cosine is 1 and every sine 0.
        let zero: Vec<f32> = timestep_embedding(0.0, 6, 10_000.0, &cpu_device())
            .into_data()
            .try_into_vec_as::<f32>()
            .unwrap();
        assert_eq!(zero, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }
}
