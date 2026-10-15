//! The `ten-base5-per-24bit-v1` (quint5) decoder codes.
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The decoder-code format this reads.
pub const QUINT5_FORMAT: &str = "ten-base5-per-24bit-v1";

/// Expands one quint5 matrix: the five-symbol code `FermionResearch`'s
/// Phonon-1 stores its text-decoder linears in.
///
/// Each row of `packed` is groups of 3 bytes, each a little-endian 24-bit
/// payload of ten base-5 digits (digit `k` is `payload / 5^k % 5`), the
/// row's `cols` symbols in order and the last group padded. Symbol `s` of row
/// `r` is the weight
/// `[-α - ρ, -α + ρ, 0, α - ρ, α + ρ][s]`, with `α = alpha[r]` and `ρ` the
/// row's residual (`residual` holds one value for every row, or one per
/// row).
///
/// # Arguments
/// * `packed` - `[rows, ceil(cols / 10) * 3]` bytes, row-major.
/// * `alpha` - `[rows]`.
/// * `residual` - `[1]` or `[rows]`.
///
/// # Returns
/// `[rows, cols]` weights, row-major.
///
/// # Errors
/// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource) when a
/// buffer's length does not fit the shape, or a payload holds a digit
/// sequence past `5^10`.
pub fn decode_quint5(
    packed: &[u8],
    rows: usize,
    cols: usize,
    alpha: &[f32],
    residual: &[f32],
) -> BunsenResult<Vec<f32>> {
    let bad = |why: String| BunsenError::invalid_resource(format!("a quint5 matrix: {why}"));
    let row_bytes = cols.div_ceil(10) * 3;
    if packed.len() != rows * row_bytes {
        return Err(bad(format!(
            "{} bytes for {rows} rows of {row_bytes}",
            packed.len()
        )));
    }
    if alpha.len() != rows {
        return Err(bad(format!("{} alphas for {rows} rows", alpha.len())));
    }
    if residual.len() != 1 && residual.len() != rows {
        return Err(bad(format!("{} residuals for {rows} rows", residual.len())));
    }

    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let (a, rho) = (alpha[r], residual[if residual.len() == 1 { 0 } else { r }]);
        let levels = [-a - rho, -a + rho, 0.0, a - rho, a + rho];
        let row = &packed[r * row_bytes..(r + 1) * row_bytes];
        let mut emitted = 0;
        for group in row.as_chunks::<3>().0 {
            let mut payload = group[0] as u32 | (group[1] as u32) << 8 | (group[2] as u32) << 16;
            if payload >= 9_765_625 {
                return Err(bad(format!("row {r}: payload {payload} past 5^10")));
            }
            for _ in 0..10 {
                if emitted == cols {
                    break;
                }
                out.push(levels[(payload % 5) as usize]);
                payload /= 5;
                emitted += 1;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hand_packed_row() {
        // Symbols 0..5 then 2 (zero) four times: 0 + 1*5 + 2*25 + 3*125 + 4*625
        // + 2*(5^5 + 5^6 + 5^7 + 5^8) + 2*5^9.
        let payload: u32 = [0u32, 1, 2, 3, 4, 2, 2, 2, 2, 2]
            .iter()
            .rev()
            .fold(0, |acc, &d| acc * 5 + d);
        let bytes = payload.to_le_bytes();
        let dense = decode_quint5(&bytes[..3], 1, 7, &[1.0], &[0.25]).unwrap();
        assert_eq!(dense, [-1.25, -0.75, 0.0, 0.75, 1.25, 0.0, 0.0]);
        assert!(decode_quint5(&bytes[..3], 1, 11, &[1.0], &[0.25]).is_err());
        assert!(decode_quint5(&[0xff, 0xff, 0xff], 1, 10, &[1.0], &[0.25]).is_err());
    }

    /// Every record of `testdata/fermion/quint5.safetensors` expands to the
    /// weights the reference reader's symbols select.
    #[cfg(feature = "store_safetensors")]
    #[test]
    fn test_matches_the_reference_reader() {
        use burn::store::{
            ModuleStore,
            SafetensorsStore,
            bridge,
        };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/fermion/quint5.safetensors"
        );
        let mut store = SafetensorsStore::from_file(path);
        let all = store.get_all_tensors().unwrap().clone();
        let get = |k: &str| bridge::to_data(&all[k]).unwrap();
        for name in ["a", "b"] {
            let expected = get(&format!("{name}.dense"));
            let [rows, cols] = [expected.shape()[0], expected.shape()[1]];
            let packed: Vec<u8> = get(&format!("{name}.quint5_q")).try_to_vec::<u8>().unwrap();
            let alpha: Vec<f32> = get(&format!("{name}.base_alpha"))
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let residual: Vec<f32> = get(&format!("{name}.residual_scale"))
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let dense = decode_quint5(&packed, rows, cols, &alpha, &residual).unwrap();
            assert_eq!(dense, expected.try_to_vec::<f32>().unwrap(), "{name}");
        }
    }
}
