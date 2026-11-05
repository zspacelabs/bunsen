//! Slice utility functions.
use burn::{
    Tensor,
    prelude::{
        Bool,
        SliceArg,
    },
    tensor::{
        Slice,
        SliceOps,
    },
};

/// Read a 2D slice from a tensor.
pub fn read_2d_slice<R>(
    state: Tensor<2, Bool>,
    ranges: R,
) -> Vec<Vec<bool>>
where
    R: SliceArg,
{
    let slices: [Slice; 2] = ranges.into_slices(&state.shape()).try_into().unwrap();
    let w = state.shape().slice(&slices).unwrap()[1];

    state
        .slice(slices)
        .to_data_as::<bool>()
        .try_to_vec_as::<bool>()
        .unwrap()
        .chunks(w)
        .map(<[_]>::to_vec)
        .collect()
}
