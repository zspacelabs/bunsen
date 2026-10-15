//! Relative-position attention operations.
use burn::{
    Tensor,
    prelude::TensorData,
    tensor::Device,
};

/// The sinusoidal relative-position table of Transformer-XL attention.
///
/// Row `r` encodes the relative position `p = (seq_len - 1) - r`, so the
/// rows run from `seq_len - 1` down to `-(seq_len - 1)`. Each row interleaves
/// sines and cosines: element `2i` is `sin(p * f_i)` and element `2i + 1` is
/// `cos(p * f_i)`, with `f_i = 10000^(-2i / d_model)`.
///
/// This is the table `NeMo`'s `RelPositionalEncoding` (and `transformers`'
/// `ParakeetEncoderRelPositionalEncoding`) builds; [`rel_shift`] aligns
/// scores against it.
///
/// # Arguments
/// * `seq_len` - the sequence length `T`.
/// * `d_model` - the table width; even.
/// * `device` - where to build it.
///
/// # Returns
/// `[1, 2 * seq_len - 1, d_model]`.
///
/// # Panics
/// If `d_model` is odd, or `seq_len` is zero.
pub fn rel_position_table(
    seq_len: usize,
    d_model: usize,
    device: &Device,
) -> Tensor<3> {
    assert!(seq_len > 0, "rel_position_table: seq_len is zero");
    assert!(
        d_model.is_multiple_of(2),
        "rel_position_table: d_model ({d_model}) is odd"
    );
    let rows = 2 * seq_len - 1;
    let mut data = Vec::with_capacity(rows * d_model);
    for r in 0..rows {
        let p = (seq_len as f64 - 1.0) - r as f64;
        for i in 0..d_model / 2 {
            // `transformers` computes the frequencies in f32; so does this, so
            // the table is the same to the last bit for large positions.
            let freq = 1.0f32 / 10000f32.powf((2 * i) as f32 / d_model as f32);
            let angle = (p as f32) * freq;
            data.push(angle.sin());
            data.push(angle.cos());
        }
    }
    Tensor::from_data(TensorData::new(data, [1, rows, d_model]), device)
}

/// The relative shift of Transformer-XL attention (its appendix B).
///
/// Takes scores of each query against every relative position of a
/// [`rel_position_table`] and returns them against every key: output
/// `[.., i, j]` is input `[.., i, j + (T - 1) - i]`, the score of query `i`
/// at relative position `i - j`.
///
/// # Arguments
/// * `scores` - `[batch, heads, T, 2 * T - 1]`.
///
/// # Returns
/// `[batch, heads, T, T]`.
///
/// # Panics
/// If the last axis is not `2 * T - 1`.
pub fn rel_shift(scores: Tensor<4>) -> Tensor<4> {
    let [batch, heads, t, p] = scores.dims();
    assert_eq!(p, 2 * t - 1, "rel_shift: positions ({p}) != 2 * {t} - 1");
    // Pad one zero column on the left, read the [t, p + 1] block as
    // [p + 1, t] rows, drop the first row, and read the rest back as
    // [t, p]: each row has slid one place further left than the one above.
    let padded = Tensor::cat(
        vec![
            Tensor::zeros([batch, heads, t, 1], &scores.device()),
            scores,
        ],
        3,
    );
    padded
        .reshape([batch, heads, p + 1, t])
        .slice_dim(2, 1..)
        .reshape([batch, heads, t, p])
        .slice_dim(3, 0..t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_rel_position_table() {
        let device = cpu_device();
        let table = rel_position_table(3, 4, &device);
        assert_eq!(table.dims(), [1, 5, 4]);
        let values: Vec<f32> = table.into_data().try_into_vec_as::<f32>().unwrap();
        // Row 2 is position 0: sin 0, cos 0, sin 0, cos 0.
        assert_eq!(&values[8..12], &[0.0, 1.0, 0.0, 1.0]);
        // Row 0 is position 2; the second frequency is 10000^(-1/2) = 0.01.
        let expected = [2f32.sin(), 2f32.cos(), 0.02f32.sin(), 0.02f32.cos()];
        for (a, b) in values[0..4].iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
        // Row 4 is position -2.
        assert!((values[16] + 2f32.sin()).abs() < 1e-6);
        assert!((values[17] - 2f32.cos()).abs() < 1e-6);
    }

    /// `rel_shift` is the gather `out[i][j] = in[i][j + (T - 1) - i]`.
    #[test]
    fn test_rel_shift_is_the_diagonal_gather() {
        let device = cpu_device();
        let (b, h, t) = (2, 3, 5);
        let p = 2 * t - 1;
        let n = b * h * t * p;
        let input: Vec<f32> = (0..n).map(|k| k as f32).collect();
        let scores = Tensor::<4>::from_data(TensorData::new(input.clone(), [b, h, t, p]), &device);

        let shifted = rel_shift(scores);
        assert_eq!(shifted.dims(), [b, h, t, t]);
        let out: Vec<f32> = shifted.into_data().try_into_vec_as::<f32>().unwrap();
        for bh in 0..b * h {
            for i in 0..t {
                for j in 0..t {
                    let expected = input[(bh * t + i) * p + j + (t - 1) - i];
                    assert_eq!(out[(bh * t + i) * t + j], expected, "({bh}, {i}, {j})");
                }
            }
        }
    }
}
