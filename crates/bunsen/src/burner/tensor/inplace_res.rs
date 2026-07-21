use burn::{
    Tensor,
    prelude::Backend,
    tensor::{
        BasicOps,
        Element,
    },
};

/// Value-returning variant of `Tensor::inplace`.
///
/// Executes an operation on the tensor and modifies its value.
///
/// # Notes
///
/// This won't necessarily reuse the same tensor data/buffer, but it should if
/// there is no other reference pointing to the same tensor.
///
/// Wrapping operations with inplace is not an optimization, it's mainly there
/// if you want to mutate a tensor by using owned operations. A plausible usage
/// would be to update the weights of a mutable model reference.
pub fn inplace_res<V, F, B, const D: usize, K>(
    tensor: &mut Tensor<B, D, K>,
    func: F,
) -> V
where
    F: FnOnce(Tensor<B, D, K>) -> (Tensor<B, D, K>, V),
    B: Backend,
    K: BasicOps<B>,
    K::Elem: Element,
{
    let mut tensor_owned = Tensor::empty([0; D], &tensor.device());
    core::mem::swap(&mut tensor_owned, tensor);

    let (mut tensor_new, v) = func(tensor_owned);
    core::mem::swap(&mut tensor_new, tensor);

    v
}
