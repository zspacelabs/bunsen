use std::{
    collections::HashSet,
    sync::Arc,
};

use burn::prelude::Backend;
use dashmap::DashMap;

use crate::prelude::dynamic::DynTensor;

/// [`DynTensor`] Binding Environment.
///
/// Instances of [`DynTensorEnv`] are shared handles to a common tensor
/// environment; and new handles can be created by [`Clone::clone`].
#[derive(Debug, Clone, Default)]
pub struct DynTensorEnv<B: Backend> {
    map: Arc<DashMap<String, DynTensor<B>>>,
}

impl<B: Backend> DynTensorEnv<B> {
    /// Create a new environment which is a distinct copy.
    pub fn copy(&self) -> Self {
        Self {
            map: Arc::new(self.map.as_ref().clone()),
        }
    }

    /// Get the number of tensors in the environment.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if the environment is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get the keys of the tensors in the environment.
    pub fn keys(&self) -> HashSet<String> {
        self.map.iter().map(|r| r.key().to_string()).collect()
    }

    /// Bind a [`DynTensor`] to the environment.
    pub fn bind(
        &mut self,
        name: impl AsRef<str>,
        tensor: impl Into<DynTensor<B>>,
    ) {
        self.map.insert(name.as_ref().to_string(), tensor.into());
    }

    /// Drop a [`DynTensor`] from the environment.
    ///
    /// # Returns
    ///
    /// The dropped [`DynTensor`], or `None` if the tensor does not exist.
    pub fn drop(
        &mut self,
        name: impl AsRef<str>,
    ) -> Option<DynTensor<B>> {
        self.map.remove(name.as_ref()).map(|(_, v)| v)
    }

    /// Check if a [`DynTensor`] exists in the environment.
    pub fn contains_key(
        &self,
        name: impl AsRef<str>,
    ) -> bool {
        self.map.contains_key(name.as_ref())
    }

    /// Get a reference to a [`DynTensor`] from the environment.
    pub fn get_ref(
        &self,
        name: impl AsRef<str>,
    ) -> Option<dashmap::mapref::one::Ref<'_, String, DynTensor<B>>> {
        self.map.get(name.as_ref())
    }
}

#[cfg(test)]
mod tests {

    use burn::{
        Tensor,
        prelude::{
            Bool,
            Float,
        },
        tensor::Int,
    };

    use super::*;
    use crate::{
        prelude::*,
        support::testing::CpuBackend,
    };

    #[test]
    fn test_dyn_tensor_env() {
        type B = CpuBackend;
        let device = Default::default();

        let mut env = DynTensorEnv::<B>::default();

        let int_tensor: Tensor<B, 2, Int> = Tensor::arange(0..6, &device).reshape([2, 3]);
        let float_tensor: Tensor<B, 3> = Tensor::arange(0..24, &device).reshape([2, 3, 4]).float();
        let bool_tensor: Tensor<B, 1, Bool> = Tensor::zeros([2], &device);

        env.bind("foo", int_tensor.clone());

        let float_dyn_tensor: DynTensor<B> = float_tensor.clone().into();
        env.bind("bar", float_dyn_tensor);

        let mut env2 = env.clone();

        env2.bind("baz", bool_tensor.clone());

        let mut env_keys: HashSet<String> = Default::default();
        env_keys.insert("foo".to_string());
        env_keys.insert("bar".to_string());
        env_keys.insert("baz".to_string());

        assert_eq!(&env.keys(), &env_keys);

        assert_eq!(env.contains_key("foo"), true);
        env2.get_ref("foo")
            .unwrap()
            .downcast_clone::<2, Int>()
            .unwrap()
            .to_data_as::<i32>()
            .assert_eq(&int_tensor.to_data_as::<i32>(), true);

        env.get_ref("bar")
            .unwrap()
            .downcast_clone::<3, Float>()
            .unwrap()
            .to_data_as::<f32>()
            .assert_eq(&float_tensor.to_data_as::<f32>(), true);

        env.get_ref("baz")
            .unwrap()
            .downcast_clone::<1, Bool>()
            .unwrap()
            .to_data_as::<bool>()
            .assert_eq(&bool_tensor.to_data_as::<bool>(), true);

        let mut dup = env.copy();
        dup.drop("baz");

        let mut dup_env_keys: HashSet<String> = Default::default();
        dup_env_keys.insert("foo".to_string());
        dup_env_keys.insert("bar".to_string());
        assert_eq!(&dup.keys(), &dup_env_keys);
        assert_eq!(&env.keys(), &env_keys);

        dup.get_ref("foo")
            .unwrap()
            .downcast_clone::<2, Int>()
            .unwrap()
            .to_data_as::<i32>()
            .assert_eq(&int_tensor.to_data_as::<i32>(), true);

        dup.get_ref("bar")
            .unwrap()
            .downcast_clone::<3, Float>()
            .unwrap()
            .to_data_as::<f32>()
            .assert_eq(&float_tensor.to_data_as::<f32>(), true);
    }
}
