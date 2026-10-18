//! MLX's affine-quantized tensors.
use alloc::{
    format,
    string::String,
    vec::Vec,
};

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// Dequantizes one MLX affine-quantized matrix: the layout `mlx.core.quantize`
/// writes and `mlx.core.dequantize` reads.
///
/// Each row of `words` is a little-endian bit stream of `cols` codes of
/// `bits` bits (codes straddle word boundaries at 3, 5 and 6 bits). Code `q`
/// in group `g` of a row, `group_size` consecutive codes, has the value
/// `scales[g] * q + biases[g]`, evaluated in f32.
///
/// # Arguments
/// * `words` - `[rows, cols * bits / 32]` packed codes, row-major.
/// * `scales`, `biases` - `[rows, cols / group_size]`, row-major.
///
/// # Returns
/// `[rows, cols]` values, row-major.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when
/// `bits` is not one of 2, 3, 4, 5, 6 or 8, `group_size` does not divide
/// `cols`, or a buffer's length does not fit the shape.
pub fn dequantize_mlx_affine(
    words: &[u32],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    bits: u32,
    group_size: usize,
) -> BunsenResult<Vec<f32>> {
    let bad = |why: String| BunsenError::invalid_resource(format!("an MLX affine tensor: {why}"));
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return Err(bad(format!("{bits}-bit codes")));
    }
    if group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(bad(format!("groups of {group_size} over {cols} columns")));
    }
    let row_bits = cols * bits as usize;
    if !row_bits.is_multiple_of(32) {
        return Err(bad(format!(
            "{cols} codes of {bits} bits do not fill whole words"
        )));
    }
    let row_words = row_bits / 32;
    let groups = cols / group_size;
    if words.len() != rows * row_words {
        return Err(bad(format!(
            "{} words for {rows} rows of {row_words}",
            words.len()
        )));
    }
    if scales.len() != rows * groups || biases.len() != rows * groups {
        return Err(bad(format!(
            "{} scales and {} biases for {rows} rows of {groups} groups",
            scales.len(),
            biases.len()
        )));
    }

    let mask = (1u64 << bits) - 1;
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let row = &words[r * row_words..(r + 1) * row_words];
        for c in 0..cols {
            let bit = c * bits as usize;
            let (word, shift) = (bit / 32, bit % 32);
            let mut stream = row[word] as u64;
            if word + 1 < row.len() {
                stream |= (row[word + 1] as u64) << 32;
            }
            let q = ((stream >> shift) & mask) as f32;
            let g = r * groups + c / group_size;
            out.push(scales[g] * q + biases[g]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hand_packed_codes() {
        // Two rows of 32 four-bit codes: row 0 is 0..16 twice, row 1 is all 15.
        let mut words = Vec::new();
        for _ in 0..2 {
            words.push(0x7654_3210);
            words.push(0xfedc_ba98);
        }
        words.extend([0xffff_ffff; 4]);
        let dense = dequantize_mlx_affine(
            &words,
            &[1.0, 2.0, 0.5, 0.5],
            &[0.0, 1.0, -1.0, 0.0],
            2,
            32,
            4,
            16,
        )
        .unwrap();
        assert_eq!(&dense[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(dense[16], 1.0);
        assert_eq!(dense[17], 3.0);
        assert_eq!(dense[32], 6.5);
        assert_eq!(dense[63], 7.5);
        assert!(dequantize_mlx_affine(&words, &[1.0; 4], &[0.0; 4], 2, 32, 7, 16).is_err());
        assert!(dequantize_mlx_affine(&words, &[1.0; 4], &[0.0; 4], 2, 32, 4, 12).is_err());
    }

    /// Every record of `testdata/fermion/mlx_affine.safetensors` dequantizes
    /// to what `mlx.core.dequantize` gave for it.
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_matches_mlx() {
        use burn::store::{
            ModuleStore,
            SafetensorsStore,
            bridge,
        };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/fermion/mlx_affine.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
        let all = store.get_all_tensors().unwrap().clone();
        let get = |k: &str| bridge::to_data(&all[k]).unwrap();
        for (name, bits, group) in [("b4g64", 4, 64), ("b6g128", 6, 128), ("b8g64", 8, 64)] {
            let words: Vec<u32> = get(&format!("{name}.weight")).try_to_vec::<u32>().unwrap();
            let scales: Vec<f32> = get(&format!("{name}.scales"))
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let biases: Vec<f32> = get(&format!("{name}.biases"))
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let expected = get(&format!("{name}.dense"));
            let [rows, cols] = [expected.shape()[0], expected.shape()[1]];
            let dense =
                dequantize_mlx_affine(&words, &scales, &biases, rows, cols, bits, group).unwrap();
            assert_eq!(dense, expected.try_to_vec::<f32>().unwrap(), "{name}");
        }
    }
}
