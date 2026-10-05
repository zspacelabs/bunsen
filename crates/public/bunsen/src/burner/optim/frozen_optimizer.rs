use burn::{
    optim::{
        LearningRate,
        SimpleOptimizer,
    },
    prelude::Backend,
    tensor::Tensor,
};

/// A [`SimpleOptimizer`] that never moves a parameter: `step` returns the
/// tensor unchanged and keeps no state.
///
/// It is the optimizer of a frozen [`OptimizerGroup`], which
/// [`OptimizerGroup::frozen`] builds: the way to keep parameters fixed on
/// purpose under a `GroupOptimizerAdaptorN` (e.g.
/// [`GroupOptimizerAdaptor2`]). A frozen group's parameters count as
/// assigned, so `new` accepts them; their gradients are dropped. As an
/// optimizer type of its own, a frozen group takes one of the adaptor's `N`
/// type slots.
///
/// Autodiff still computes the gradients of a frozen group's parameters. To
/// skip that work too, also turn off their `require_grad`.
///
/// [`OptimizerGroup`]: crate::burner::optim::OptimizerGroup
/// [`OptimizerGroup::frozen`]: crate::burner::optim::OptimizerGroup::frozen
/// [`GroupOptimizerAdaptor2`]: crate::burner::optim::GroupOptimizerAdaptor2
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrozenOptimizer;

impl<B: Backend> SimpleOptimizer<B> for FrozenOptimizer {
    type State<const D: usize> = ();

    fn step<const D: usize>(
        &self,
        _lr: LearningRate,
        tensor: Tensor<B, D>,
        _grad: Tensor<B, D>,
        _state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        (tensor, None)
    }

    fn to_device<const D: usize>(
        state: Self::State<D>,
        _device: &B::Device,
    ) -> Self::State<D> {
        state
    }
}
