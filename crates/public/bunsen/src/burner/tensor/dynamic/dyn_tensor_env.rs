use std::{
    collections::HashSet,
    sync::Arc,
};

use burn::{
    Tensor,
    prelude::Backend,
    tensor::BasicOps,
};
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
    ///
    /// This is a copy of the keys of the tensors in the environment.
    pub fn keys(&self) -> HashSet<String> {
        self.iter().map(|r| r.key().clone()).collect()
    }

    /// Map over all tensors in the environment.
    ///
    /// # Returns
    ///
    /// A ref-guard iterator, the items of the environment.
    /// Use `r.key()` and `r.value()` to access the key and value of each item.
    pub fn iter(&self) -> dashmap::iter::Iter<'_, String, DynTensor<B>> {
        self.map.iter()
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

    /// Get a clone of a [`DynTensor`] from the environment.
    pub fn get_dyn(
        &self,
        name: impl AsRef<str>,
    ) -> Option<DynTensor<B>> {
        self.get_ref(name).map(|r| r.value().clone())
    }

    /// Get a clone of a [`DynTensor`] from the environment, or panic.
    pub fn expect_dyn(
        &self,
        name: impl AsRef<str>,
    ) -> DynTensor<B> {
        self.get_dyn(name).expect("tensor not found: \"{name:?}\"")
    }

    /// Get a downcast clone of a [`Tensor`] from the environment.
    ///
    /// # Returns
    ///
    /// Either the typed [`Some(Tensor<B, D, K>)`](`burn::tensor::Tensor`);
    /// or `None` if the key isn't bound, or the tensor does not match this
    /// type.
    pub fn get_tensor<const D: usize, K>(
        &self,
        name: impl AsRef<str>,
    ) -> Option<Tensor<B, D, K>>
    where
        K: BasicOps<B> + 'static,
    {
        self.get_dyn(name).and_then(|dt| dt.downcast_clone())
    }

    /// Get a downcast clone of a [`Tensor`] from the environment, or panic.
    pub fn expect_tensor<const D: usize, K>(
        &self,
        name: impl AsRef<str>,
    ) -> Tensor<B, D, K>
    where
        K: BasicOps<B> + 'static,
    {
        self.get_tensor(name)
            .expect("tensor not found: \"{name:?}\"")
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

        // Two handles to the same environment.
        let mut env1_a = DynTensorEnv::<B>::default();
        let mut env1_b = env1_a.clone();

        let int_tensor: Tensor<B, 2, Int> = Tensor::arange(0..6, &device).reshape([2, 3]);
        let float_tensor: Tensor<B, 3> = Tensor::arange(0..24, &device).reshape([2, 3, 4]).float();
        let bool_tensor: Tensor<B, 1, Bool> = Tensor::zeros([2], &device);

        assert_eq!(env1_a.contains_key("foo"), false);
        assert_eq!(env1_b.contains_key("foo"), false);
        env1_a.bind("foo", int_tensor.clone());
        assert_eq!(env1_a.contains_key("foo"), true);
        assert_eq!(env1_b.contains_key("foo"), true);
        env1_a
            .expect_tensor::<2, Int>("foo")
            .to_data_as::<i32>()
            .assert_eq(&int_tensor.to_data_as::<i32>(), true);
        env1_b
            .expect_tensor::<2, Int>("foo")
            .to_data_as::<i32>()
            .assert_eq(&int_tensor.to_data_as::<i32>(), true);

        let float_dyn_tensor: DynTensor<B> = float_tensor.clone().into();
        env1_a.bind("bar", float_dyn_tensor);
        assert_eq!(env1_a.contains_key("bar"), true);
        assert_eq!(env1_b.contains_key("bar"), true);
        env1_a
            .expect_tensor::<3, Float>("bar")
            .to_data_as::<f32>()
            .assert_eq(&float_tensor.to_data_as::<f32>(), true);
        env1_b
            .expect_tensor::<3, Float>("bar")
            .to_data_as::<f32>()
            .assert_eq(&float_tensor.to_data_as::<f32>(), true);

        env1_b.bind("baz", bool_tensor.clone());
        assert_eq!(env1_a.contains_key("baz"), true);
        assert_eq!(env1_b.contains_key("baz"), true);
        env1_a
            .expect_tensor::<1, Bool>("baz")
            .to_data_as::<bool>()
            .assert_eq(&bool_tensor.to_data_as::<bool>(), true);
        env1_b
            .expect_tensor::<1, Bool>("baz")
            .to_data_as::<bool>()
            .assert_eq(&bool_tensor.to_data_as::<bool>(), true);

        let env1_keys: HashSet<String> = ["foo", "bar", "baz"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(&env1_a.keys(), &env1_keys);
        assert_eq!(&env1_b.keys(), &env1_keys);

        // Make a distinct copy of env1.
        let mut env2 = env1_a.copy();
        env2.drop("baz");
        assert_eq!(env2.contains_key("baz"), false);
        let evn2_keys: HashSet<String> = ["foo", "bar"].into_iter().map(String::from).collect();
        assert_eq!(&env2.keys(), &evn2_keys);

        // Dropping "baz" should not affect the original environment's keys
        assert_eq!(&env1_a.keys(), &env1_keys);
        assert_eq!(env1_a.contains_key("baz"), true);
        assert_eq!(env1_b.contains_key("baz"), true);

        // The tensors should remain the same.
        env2.expect_tensor::<2, Int>("foo")
            .to_data_as::<i32>()
            .assert_eq(&int_tensor.to_data_as::<i32>(), true);

        env2.expect_tensor::<3, Float>("bar")
            .to_data_as::<f32>()
            .assert_eq(&float_tensor.to_data_as::<f32>(), true);

        assert!(env2.get_ref("baz").is_none());
    }
}
