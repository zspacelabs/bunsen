//! # NaN and infinity replacement

use burn::{
    Tensor,
    prelude::Backend,
};

/// Replaces NaN, negative infinity and positive infinity with finite values.
///
/// Like `numpy.nan_to_num`, except that every replacement is explicit: NaN
/// becomes `nan_val`, `-inf` becomes `neg_inf_val`, and `+inf` becomes
/// `pos_inf_val`. Every other value passes through.
pub fn nan_to_num<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    nan_val: f64,
    neg_inf_val: f64,
    pos_inf_val: f64,
) -> Tensor<B, D> {
    let is_nan = tensor.clone().is_nan();
    let is_inf = tensor.clone().is_inf();
    let is_neg = tensor.clone().lower_elem(0.0);

    let pos_inf = is_inf.clone().bool_and(is_neg.clone().bool_not());
    let neg_inf = is_inf.clone().bool_and(is_neg);

    tensor
        .mask_fill(is_nan, nan_val)
        .mask_fill(neg_inf, neg_inf_val)
        .mask_fill(pos_inf, pos_inf_val)
}
