//! # Clone Box Trait
use std::{
    any::Any,
    fmt::Debug,
};

/// A trait for cloning values into a boxed form.
pub trait CloneBox: 'static + Any + Debug + Send + Sync {
    /// Clones the boxed value into a new boxed value.
    fn clone_box(&self) -> Box<dyn CloneBox>;
}

impl dyn CloneBox {
    /// Downcasts the boxed value to a specific type.
    ///
    /// See: `Any::downcast_ref`.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref::<T>()
    }
}

impl<T: 'static + Any + Debug + Clone + Send + Sync> CloneBox for T {
    fn clone_box(&self) -> Box<dyn CloneBox> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CloneBox> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        backend::Wgpu,
        tensor::Distribution,
    };

    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn test_clone_box_tensor() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> = Tensor::random([2, 3], Distribution::Default, &device);

        let boxed: Box<dyn CloneBox> = Box::new(source.clone());

        assert_send::<Box<dyn CloneBox>>();

        let cloned_box = boxed.clone();

        let clone = cloned_box.downcast_ref::<Tensor<B, 2>>().unwrap();

        clone.to_data().assert_eq(&source.to_data(), true);
    }
}
