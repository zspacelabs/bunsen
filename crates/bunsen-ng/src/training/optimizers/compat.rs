//! Cross-burn version compatibility hacks.
use std::marker::PhantomData;

use burn::{
    Tensor,
    grad_clipping::GradientClipping,
    module::{
        AutodiffModule,
        ParamId,
    },
    optim::{
        GradientsParams,
        MultiGradientsParams,
        SimpleOptimizer,
        adaptor::OptimizerAdaptor,
        record::AdaptorRecord,
    },
    prelude::Backend,
    tensor::backend::AutodiffBackend,
};
use hashbrown::HashMap;

/// This is an unsafe shadow of [`OptimizerAdaptor`].
///
/// This should be removed after burn 0.22.0
///
/// See:
/// - <https://github.com/tracel-ai/burn/pull/4818>
///
/// It exists to permit us to clone `optim`.
#[allow(unused)]
struct XOptimizerAdaptor<O, M, B>
where
    O: SimpleOptimizer<B::InnerBackend>,
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    optim: O,
    records: HashMap<ParamId, AdaptorRecord<O, B>>,
    module: PhantomData<M>,
    grad_clipping: Option<GradientClipping>,
}

/// Clone the internal [`SimpleOptimizer`] from an [`OptimizerAdaptor`].
///
/// This should be removed after burn 0.22.0
///
/// See:
/// - <https://github.com/tracel-ai/burn/pull/4818>
pub fn clone_simple_optimizer<O, M, B>(adaptor: &OptimizerAdaptor<O, M, B>) -> O
where
    O: SimpleOptimizer<B::InnerBackend>,
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    unsafe {
        let adaptor: &XOptimizerAdaptor<O, M, B> = std::mem::transmute(adaptor);
        adaptor.optim.clone()
    }
}

/// This is a local public clone of the private burn version.
///
/// See: <https://github.com/tracel-ai/burn/pull/4822>
pub enum GradAdaptor {
    Single(GradientsParams),
    Multi(MultiGradientsParams),
}

impl From<GradientsParams> for GradAdaptor {
    fn from(grads: GradientsParams) -> Self {
        Self::Single(grads)
    }
}

impl From<MultiGradientsParams> for GradAdaptor {
    fn from(grads: MultiGradientsParams) -> Self {
        Self::Multi(grads)
    }
}

impl GradAdaptor {
    pub fn remove<B: Backend, const D: usize>(
        &mut self,
        id: ParamId,
    ) -> Option<(Tensor<B, D>, B::Device)> {
        match self {
            GradAdaptor::Single(grads) => grads.remove(id).map(|t| {
                let device = t.device();
                (t, device)
            }),
            GradAdaptor::Multi(grads) => grads.remove(id),
        }
    }
}
