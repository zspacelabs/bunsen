//! Burn Framework Support

use burn::{
    prelude::{
        Backend,
        Shape,
        Tensor,
    },
    tensor::BasicOps,
};

use crate::shape_view::ShapeView;

impl<'a> From<&'a Shape> for ShapeView<'a> {
    fn from(shape: &'a Shape) -> Self {
        shape.as_slice().into()
    }
}

impl<'a> From<Shape> for ShapeView<'a> {
    fn from(shape: Shape) -> Self {
        shape.to_vec().into()
    }
}

impl<'a, B, const R: usize, K> From<&'a Tensor<B, R, K>> for ShapeView<'a>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn from(tensor: &'a Tensor<B, R, K>) -> Self {
        tensor.shape().into()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::ShapeView;

    #[test]
    #[allow(unused)]
    fn test_burn_shape_views() {
        let expected = vec![2, 3, 4];

        let shape = Shape::from([2, 3, 4]);
        let sv: ShapeView = shape.clone().into();
        assert_eq!(sv.as_ref(), &expected);

        let shape_ref: &Shape = &shape;
        let sv: ShapeView = shape_ref.into();
        assert_eq!(shape_ref.as_ref(), &expected);

        let tensor: Tensor<burn::backend::NdArray, 2> = Tensor::zeros([2, 2], &Default::default());
        let tensor_ref = &tensor;
        let sv: ShapeView = tensor_ref.into();
        assert_eq!(sv.as_ref(), &[2, 2]);
    }
}
