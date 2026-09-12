//! Slice utility functions.
use burn::{
    Tensor,
    prelude::{
        Backend,
        Bool,
        SliceArg,
    },
    tensor::Slice,
};

use crate::prelude::TensorElemOpExt;

/// Range size of a [`Slice`].
pub fn slice_size(slice: &Slice) -> usize {
    (slice.end.unwrap() - slice.start) as usize
}

/// Full shape of a [`Slice`].
pub fn slices_shape(slices: &[Slice; 2]) -> [usize; 2] {
    [slice_size(&slices[0]), slice_size(&slices[1])]
}

/// Read a 2D slice from a tensor.
pub fn read_2d_slice<B: Backend, R>(
    state: Tensor<B, 2, Bool>,
    ranges: R,
) -> Vec<Vec<bool>>
where
    R: SliceArg,
{
    let slices: [Slice; 2] = ranges.into_slices(&state.shape()).try_into().unwrap();
    let [_, w] = slices_shape(&slices);

    state
        .slice(slices)
        .to_data_as::<bool>()
        .to_vec::<bool>()
        .unwrap()
        .chunks(w)
        .map(<[_]>::to_vec)
        .collect()
}
