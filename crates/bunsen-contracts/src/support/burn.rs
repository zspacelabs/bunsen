//! Burn Framework Support

use alloc::vec::Vec;

use burn::{
    prelude::{
        Backend,
        Shape,
        Tensor,
    },
    tensor::{
        BasicOps,
        TensorKind,
    },
};

use crate::{
    ShapeArgument,
    shape_view::ShapeView,
};

impl<'a> From<&'a Shape> for ShapeView<'a> {
    fn from(shape: &'a Shape) -> Self {
        ShapeView::new(shape)
    }
}

impl<'a> From<Shape> for ShapeView<'a> {
    fn from(shape: Shape) -> Self {
        shape.to_vec().into()
    }
}

impl<'a> From<&'a Tensor<B, R, K>> for ShapeView<'a> {
    fn from(tensor: &'a Tensor<B, R, K>) -> Self {
        ShapeView::new(tensor.shape())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::ShapeView;

    #[test]
    fn test_burn_shape_views() {
        let expected = vec![2, 3, 4];

        let shape = Shape::from([2, 3, 4]);
        let sv: ShapeView = shape.clone().into();
        assert_eq!(sv.as_ref(), &expected);

        let shape_ref: &Shape = &shape;
        assert_eq!(sv.as_ref(), &expected);

        let tensor: Tensor<burn::backend::NdArray, 2> = Tensor::zeros([2, 2], &Default::default());
        let tensor_ref = &tensor;
        let sv: ShapeView = tensor_ref.into();
        assert_eq!(sv.as_ref(), &[2, 2]);
    }
}
