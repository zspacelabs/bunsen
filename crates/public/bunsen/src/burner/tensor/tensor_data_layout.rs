//! Rearranging [`TensorData`] on the host, without a device.
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
}
