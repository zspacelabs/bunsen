use burn::{
    optim::{
        LearningRate,
        Optimizer,
    },
    tensor::{
        Device,
        Tensor,
    },
};

/// An [`Optimizer`] that never moves a parameter: `step` returns the tensor
/// unchanged and keeps no state.
///
/// It is the optimizer of a frozen [`OptimizerGroup`], which
/// [`OptimizerGroup::frozen`] builds: the way to keep parameters fixed on
/// purpose under a [`GroupOptimizerPlan`]. A frozen group's parameters count
/// as assigned, so the plan accepts them; their gradients are dropped. It is
/// also the fallback of a plan built with
/// [`UnknownParamPolicy::Freeze`](crate::burner::optim::UnknownParamPolicy::Freeze).
///
/// Autodiff still computes the gradients of a frozen group's parameters.
/// [`Module::freeze_group`](burn::module::Module::freeze_group) is the cheaper
/// path: it turns their `require_grad` off, so no gradient is computed at all.
/// It also turns off the training flags (dropout and the like) that its group
/// matches.
///
/// [`OptimizerGroup`]: crate::burner::optim::OptimizerGroup
/// [`OptimizerGroup::frozen`]: crate::burner::optim::OptimizerGroup::frozen
/// [`GroupOptimizerPlan`]: crate::burner::optim::GroupOptimizerPlan
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrozenOptimizer;

impl Optimizer for FrozenOptimizer {
    type State<const D: usize> = ();

    fn step<const D: usize>(
        &self,
        _lr: LearningRate,
        tensor: Tensor<D>,
        _grad: Tensor<D>,
        _state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        (tensor, None)
    }

    fn to_device<const D: usize>(
        state: Self::State<D>,
        _device: &Device,
    ) -> Self::State<D> {
        state
    }
}

#[cfg(test)]
mod tests {
    use burn::optim::ModuleOptimizer;

    use super::*;
    use crate::support::testing::cpu_device;

    #[test]
    fn test_frozen_optimizer_leaves_the_tensor_unchanged() {
        let device = cpu_device();
        let tensor = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let grad = Tensor::<2>::ones([2, 2], &device);

        let (stepped, state) = FrozenOptimizer.step(1.0, tensor.clone(), grad, None);

        assert_eq!(stepped.into_data(), tensor.into_data());
        assert!(state.is_none());
    }

    #[test]
    fn test_frozen_optimizer_mounts_as_a_module_optimizer() {
        let optim: ModuleOptimizer = FrozenOptimizer.into();
        assert!(!optim.has_gradient_clipping());
    }
}
