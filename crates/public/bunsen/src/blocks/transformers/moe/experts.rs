//! # The expert stack

use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::Initializer,
    prelude::Int,
    tensor::Device,
};

use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
    ops::transformers::moe::{
        ExpertDispatchStrategy,
        PackedExperts,
        expert_dispatch,
    },
};

/// Common meta for [`ExpertStack`] and [`ExpertStackConfig`].
pub trait ExpertStackMeta {
    /// The expert count, `E`.
    fn n_experts(&self) -> usize;

    /// The width the experts read and write, `W`.
    fn width(&self) -> usize;

    /// Each expert's hidden width, `H`.
    fn hidden(&self) -> usize;
}

/// Config for [`ExpertStack`].
#[derive(Config, Debug)]
pub struct ExpertStackConfig {
    /// The expert count.
    pub n_experts: usize,

    /// The width the experts read and write: the model's, or a latent's.
    pub width: usize,

    /// Each expert's hidden width.
    pub hidden: usize,

    /// How rows reach their experts.
    #[config(default = "ExpertDispatchStrategy::Auto")]
    pub strategy: ExpertDispatchStrategy,

    /// The weights' initializer.
    #[config(default = "Initializer::Normal { mean: 0.0, std: 0.02 }")]
    pub initializer: Initializer,
}

impl ExpertStackMeta for ExpertStackConfig {
    fn n_experts(&self) -> usize {
        self.n_experts
    }

    fn width(&self) -> usize {
        self.width
    }

    fn hidden(&self) -> usize {
        self.hidden
    }
}

impl ModuleInit<ExpertStack> for ExpertStackConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ExpertStack> {
        let (e, w, h) = (self.n_experts, self.width, self.hidden);
        Ok(ExpertStack {
            gate_up_proj: self
                .initializer
                .init_with([e, 2 * h, w], Some(w), Some(2 * h), device),
            down_proj: self
                .initializer
                .init_with([e, w, h], Some(h), Some(w), device),
            strategy: self.strategy,
        })
    }
}

/// A packed stack of `SwiGLU` experts: `expert_e(x) = down_e · (silu(gate_e ·
/// x) ⊙ (up_e · x))`, all `E` in two tensors, in the layout `transformers`
/// packs its experts in (gate rows first). A checkpoint that stores one
/// tensor per expert stacks them at load.
///
/// Built by [`ExpertStackConfig`].
#[derive(Module, Debug)]
pub struct ExpertStack {
    /// `[E, 2H, W]`: each expert's gate rows, then its up rows.
    pub gate_up_proj: Param<Tensor<3>>,

    /// `[E, W, H]`: each expert's down projection.
    pub down_proj: Param<Tensor<3>>,

    /// How rows reach their experts.
    #[module(skip)]
    pub strategy: ExpertDispatchStrategy,
}

impl ExpertStackMeta for ExpertStack {
    fn n_experts(&self) -> usize {
        self.gate_up_proj.dims()[0]
    }

    fn width(&self) -> usize {
        self.gate_up_proj.dims()[2]
    }

    fn hidden(&self) -> usize {
        self.down_proj.dims()[2]
    }
}

impl ExpertStack {
    /// The experts as the dispatch seam reads them.
    pub fn packed(&self) -> PackedExperts {
        PackedExperts {
            gate_up: self.gate_up_proj.val(),
            down: self.down_proj.val(),
        }
    }

    /// The weighted sum of each row's chosen experts' outputs
    /// ([`expert_dispatch`], by [`strategy`](Self::strategy)).
    ///
    /// # Arguments
    /// - `x`: `[N, W]`, the rows.
    /// - `ids`: `[N, k]`, each row's chosen experts, distinct within a row.
    /// - `weights`: `[N, k]`, their weights.
    ///
    /// # Returns
    /// - `[N, W]`.
    pub fn forward(
        &self,
        x: Tensor<2>,
        ids: Tensor<2, Int>,
        weights: Tensor<2>,
    ) -> Tensor<2> {
        expert_dispatch(x, ids, weights, &self.packed(), self.strategy)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
    };

    /// The config's geometry is the module's; its forward is the seam's on
    /// its packed weights.
    #[test]
    fn test_expert_stack() {
        let device = cpu_device();
        let config = ExpertStackConfig::new(4, 6, 10);
        let stack: ExpertStack = config.init(&device);
        assert_eq!(stack.gate_up_proj.dims(), [4, 20, 6]);
        assert_eq!(stack.down_proj.dims(), [4, 6, 10]);
        for meta in [&config as &dyn ExpertStackMeta, &stack] {
            assert_eq!((meta.n_experts(), meta.width(), meta.hidden()), (4, 6, 10));
        }
        assert_eq!(stack.strategy, ExpertDispatchStrategy::Auto);

        let x = crate::support::testing::seeded_tensor::<2>(
            1,
            [3, 6],
            burn::tensor::Distribution::Default,
            &device,
        );
        let ids = Tensor::<2, Int>::from_data([[0, 3], [1, 2], [3, 1]], &device);
        let weights = Tensor::<2>::from_data([[0.6f32, 0.4], [0.5, 0.5], [0.9, 0.1]], &device);
        assert_tensors_close(
            &stack.forward(x.clone(), ids.clone(), weights.clone()),
            &expert_dispatch(
                x,
                ids,
                weights,
                &stack.packed(),
                ExpertDispatchStrategy::DenseMasked,
            ),
            Tolerance::default(),
        );
    }
}
