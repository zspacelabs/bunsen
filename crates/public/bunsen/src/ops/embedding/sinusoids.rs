//! Sinusoidal position tables.
use burn::{
    Tensor,
    prelude::TensorData,
    tensor::Device,
};

/// The sinusoidal position table of Whisper's audio encoder and its
/// descendants (Qwen3-ASR's): `length` rows of `channels`, the sines of every
/// frequency followed by their cosines.
///
/// Position `p`, frequency `i`: `sin(p * f_i)` at column `i` and
/// `cos(p * f_i)` at column `channels / 2 + i`, with
/// `f_i = exp(-i * ln(max_timescale) / (channels / 2 - 1))`. Computed in f32,
/// as `openai-whisper`'s `sinusoids` and `transformers`'
/// `SinusoidsPositionEmbedding` compute it.
///
/// # Returns
/// `[length, channels]`.
///
/// # Panics
/// If `channels` is odd or less than 4.
pub fn sinusoid_table(
    length: usize,
    channels: usize,
    max_timescale: f32,
    device: &Device,
) -> Tensor<2> {
    assert!(
        channels.is_multiple_of(2) && channels >= 4,
        "sinusoid_table: channels ({channels}) must be even and at least 4"
    );
    let half = channels / 2;
    let increment = max_timescale.ln() / (half - 1) as f32;
    let inv: Vec<f32> = (0..half).map(|i| (-increment * i as f32).exp()).collect();
    let mut data = Vec::with_capacity(length * channels);
    for p in 0..length {
        let row: Vec<f32> = inv.iter().map(|f| p as f32 * f).collect();
        data.extend(row.iter().map(|a| a.sin()));
        data.extend(row.iter().map(|a| a.cos()));
    }
    Tensor::from_data(TensorData::new(data, [length, channels]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_sinusoid_table() {
        let table = sinusoid_table(3, 6, 10000.0, &cpu_device());
        assert_eq!(table.dims(), [3, 6]);
        let v: Vec<f32> = table.into_data().try_into_vec_as::<f32>().unwrap();
        // Position 0: sines 0, cosines 1.
        assert_eq!(&v[..6], &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        // Frequencies 1, 10000^(-1/2), 10000^-1.
        let expected = [1f32.sin(), 0.01f32.sin(), 0.0001f32.sin()];
        for (a, b) in v[6..9].iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }
}
