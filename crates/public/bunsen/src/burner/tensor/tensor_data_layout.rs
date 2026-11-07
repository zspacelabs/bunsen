//! Rearranging [`TensorData`] on the host, without a device.
use alloc::vec::Vec;

use burn::tensor::TensorData;

/// `data`'s axes in a new order, as
/// [`Tensor::permute`](burn::Tensor::permute) would give them: axis `i` of
/// the result is axis `order[i]` of `data`. Any element type; the bytes are
/// moved, not converted.
///
/// For a weight whose axes follow a layout a checkpoint's module chose
/// differently from the block that loads it, at load time.
///
/// # Panics
/// If `order` is not a permutation of `data`'s axes, or `data` is quantized
/// (its bytes are not whole elements).
pub fn permute_data(
    data: TensorData,
    order: &[usize],
) -> TensorData {
    let shape = data.shape().as_slice().to_vec();
    let rank = shape.len();
    assert_eq!(order.len(), rank, "a permutation of {rank} axes");
    let mut seen = vec![false; rank];
    for &axis in order {
        assert!(axis < rank && !seen[axis], "{order:?} is not a permutation");
        seen[axis] = true;
    }

    let n = shape.iter().product::<usize>();
    let dtype = data.dtype();
    let bytes = data.as_bytes();
    if n == 0 {
        return TensorData::from_bytes_vec(Vec::new(), shape_of(&shape, order), dtype);
    }
    assert_eq!(bytes.len() % n, 0, "whole elements only");
    let elem = bytes.len() / n;

    // Row-major strides of the source, in elements.
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let out_shape = shape_of(&shape, order);
    let mut out = vec![0u8; bytes.len()];
    let mut index = vec![0usize; rank];
    for k in 0..n {
        let src: usize = (0..rank).map(|i| index[i] * strides[order[i]]).sum();
        out[k * elem..(k + 1) * elem].copy_from_slice(&bytes[src * elem..(src + 1) * elem]);
        // Advance the destination's row-major index.
        for i in (0..rank).rev() {
            index[i] += 1;
            if index[i] < out_shape[i] {
                break;
            }
            index[i] = 0;
        }
    }
    TensorData::from_bytes_vec(out, out_shape, dtype)
}

/// `data` with a new shape of the same element count; the bytes are
/// unchanged.
///
/// # Panics
/// If `shape` holds a different number of elements.
pub fn reshape_data(
    data: TensorData,
    shape: impl Into<Vec<usize>>,
) -> TensorData {
    let shape = shape.into();
    assert_eq!(
        shape.iter().product::<usize>(),
        data.shape().num_elements(),
        "{shape:?} holds a different number of elements"
    );
    let (bytes, _, dtype) = data.into_parts();
    TensorData::from_bytes(bytes, shape, dtype)
}

/// `data`'s rows (its first axis) in consecutive runs of `rows[i]`, each
/// with the further axes: the parts a checkpoint fuses into one tensor
/// (a packed `qkv` projection's three, a gated MLP's two). Any element type;
/// the bytes are moved, not converted. [`concat_rows_data`] is its inverse.
///
/// # Panics
/// If `data` has no axes, `rows` does not sum to its rows, or it is
/// quantized (its bytes are not whole elements).
pub fn split_rows_data(
    data: TensorData,
    rows: &[usize],
) -> Vec<TensorData> {
    let shape = data.shape().as_slice().to_vec();
    assert!(!shape.is_empty(), "a split of a scalar");
    assert_eq!(
        rows.iter().sum::<usize>(),
        shape[0],
        "{rows:?} does not split {} rows",
        shape[0]
    );
    let n = shape.iter().product::<usize>();
    let dtype = data.dtype();
    let bytes = data.as_bytes();
    let row_bytes = if n == 0 {
        0
    } else {
        assert_eq!(bytes.len() % n, 0, "whole elements only");
        bytes.len() / shape[0]
    };
    let mut start = 0;
    rows.iter()
        .map(|&r| {
            let mut part_shape = shape.clone();
            part_shape[0] = r;
            let part = bytes[start * row_bytes..(start + r) * row_bytes].to_vec();
            start += r;
            TensorData::from_bytes_vec(part, part_shape, dtype)
        })
        .collect()
}

/// `parts` stacked along their rows (the first axis): the inverse of
/// [`split_rows_data`]. Any element type; the bytes are moved.
///
/// # Panics
/// If `parts` is empty, or the parts differ in dtype or in their further
/// axes.
pub fn concat_rows_data(parts: Vec<TensorData>) -> TensorData {
    let first = parts.first().expect("parts to concatenate");
    let dtype = first.dtype();
    let rest = first.shape().as_slice()[1..].to_vec();
    let mut rows = 0;
    let mut bytes = Vec::new();
    for part in &parts {
        let shape = part.shape().as_slice();
        assert_eq!(part.dtype(), dtype, "parts of one dtype");
        assert_eq!(&shape[1..], rest.as_slice(), "parts of one row shape");
        rows += shape[0];
        bytes.extend_from_slice(part.as_bytes());
    }
    let mut shape = vec![rows];
    shape.extend(rest);
    TensorData::from_bytes_vec(bytes, shape, dtype)
}

fn shape_of(
    shape: &[usize],
    order: &[usize],
) -> Vec<usize> {
    order.iter().map(|&axis| shape[axis]).collect()
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        tensor::Int,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    /// The host permute is the device's, for every order of three axes.
    #[test]
    fn test_permute_matches_the_device() {
        let device = cpu_device();
        let x = Tensor::<1, Int>::arange(0..24, &device).reshape([2, 3, 4]);
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let want = x.clone().permute(order).to_data();
            let got = permute_data(x.to_data(), &order);
            assert_eq!(got.shape(), want.shape(), "{order:?}");
            got.assert_eq(&want, true);
        }

        let f = Tensor::<3>::from_floats([[[1.0, 2.0], [3.0, 4.0]]], &device);
        permute_data(f.to_data(), &[0, 2, 1]).assert_eq(&f.permute([0, 2, 1]).to_data(), true);
    }

    #[test]
    fn test_reshape_keeps_the_bytes() {
        let device = cpu_device();
        let x = Tensor::<1, Int>::arange(0..6, &device);
        let got = reshape_data(x.to_data(), vec![2, 3]);
        got.assert_eq(&x.reshape([2, 3]).to_data(), true);
    }

    /// A split is the device's row slices, and concatenating the parts
    /// gives the input back.
    #[test]
    fn test_split_rows_and_back() {
        let device = cpu_device();
        let x = Tensor::<1, Int>::arange(0..24, &device).reshape([6, 2, 2]);
        let parts = split_rows_data(x.to_data(), &[1, 3, 2]);
        for (part, range) in parts.iter().zip([0..1, 1..4, 4..6]) {
            part.assert_eq(&x.clone().slice(range).to_data(), true);
        }
        concat_rows_data(parts).assert_eq(&x.to_data(), true);

        let f = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
        let parts = split_rows_data(f.to_data(), &[2, 1, 0]);
        assert_eq!(parts[2].shape().as_slice(), [0]);
        concat_rows_data(parts).assert_eq(&f.to_data(), true);
    }

    #[test]
    #[should_panic(expected = "does not split 3 rows")]
    fn test_split_rows_must_cover() {
        let _ = split_rows_data(TensorData::new(vec![0f32; 6], [3, 2]), &[1, 1]);
    }
}
