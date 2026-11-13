//! # The sparse mixture-of-experts feed-forward

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Linear,
        LinearConfig,
    },
    tensor::Device,
};

use crate::{
    blocks::transformers::moe::{
        experts::{
            ExpertStack,
            ExpertStackConfig,
            ExpertStackMeta,
        },
        router::{
            RouterChoice,
            TopKRouter,
            TopKRouterConfig,
            TopKRouterMeta,
        },
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// A shared expert's config: an always-on feed-forward beside the routed
/// experts (Qwen's `shared_expert`, `DeepSeek`'s `n_shared_experts`). Not
/// built: a [`SparseMoeConfig`] with one is refused as
/// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported).
#[derive(Config, Debug)]
pub struct SharedExpertConfig {
    /// Its hidden width.
    pub hidden: usize,

    /// Whether a sigmoid gate on the layer's input scales it (Qwen's).
    #[config(default = "false")]
    pub gated: bool,
}

/// Config for [`SparseMoe`].
#[derive(Config, Debug)]
pub struct SparseMoeConfig {
    /// The router; its width is the layer's.
    pub router: TopKRouterConfig,

    /// The routed experts; their width is the latent's, or the layer's
    /// without one.
    pub experts: ExpertStackConfig,

    /// A latent the experts run in (ZONOS3's latent MoE): the rows are
    /// projected down to this width before dispatch, and back up after,
    /// each by a linear without a bias. The router reads the full width.
    /// `None`, no latent.
    #[config(default = "None")]
    pub latent: Option<usize>,

    /// A shared expert; `None`, none.
    #[config(default = "None")]
    pub shared: Option<SharedExpertConfig>,
}

impl SparseMoeConfig {
    /// The layer's width: the router's.
    pub fn n_embed(&self) -> usize {
        self.router.width
    }
}

impl ModuleInit<SparseMoe> for SparseMoeConfig {
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// shared expert; [`Illegal`](crate::errors::BunsenErrorKind::Illegal)
    /// when the router and the experts disagree on the expert count, or the
    /// experts' width is not the latent's (the layer's without one); as the
    /// router's own.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<SparseMoe> {
        if self.shared.is_some() {
            return Err(BunsenError::unsupported("a shared expert"));
        }
        let width = self.n_embed();
        let expert_width = self.latent.unwrap_or(width);
        if self.experts.width() != expert_width {
            return Err(BunsenError::illegal(format!(
                "experts {} wide, in a layer {width} wide with latent {:?}",
                self.experts.width(),
                self.latent
            )));
        }
        if self.experts.n_experts() != self.router.n_experts() {
            return Err(BunsenError::illegal(format!(
                "{} experts, routed over {}",
                self.experts.n_experts(),
                self.router.n_experts()
            )));
        }
        let linear = |d_in, d_out| LinearConfig::new(d_in, d_out).with_bias(false).init(device);
        Ok(SparseMoe {
            router: self.router.try_init(device)?,
            experts: self.experts.try_init(device)?,
            latent_down: self.latent.map(|l| linear(width, l)),
            latent_up: self.latent.map(|l| linear(l, width)),
        })
    }
}

/// A sparse mixture-of-experts feed-forward: each position routed to its
/// `k` experts, run through the dispatch seam
/// ([`ExpertStack::forward`]), weighted and summed; in a latent when there
/// is one.
///
/// ```text
/// (ids, w) = router(x)
/// y = up(experts(down(x), ids, w))     (a latent)
/// y = experts(x, ids, w)               (none)
/// ```
///
/// Built by [`SparseMoeConfig`].
#[derive(Module, Debug)]
pub struct SparseMoe {
    /// The router.
    pub router: TopKRouter,

    /// The routed experts.
    pub experts: ExpertStack,

    /// The projection into the latent, `[D, L]`, if there is one.
    pub latent_down: Option<Linear>,

    /// The projection out of the latent, `[L, D]`, if there is one.
    pub latent_up: Option<Linear>,
}

impl SparseMoe {
    /// The layer's width.
    pub fn n_embed(&self) -> usize {
        self.router.width()
    }

    /// The feed-forward, position by position.
    ///
    /// # Arguments
    /// - `x`: `[B, T, D]`.
    ///
    /// # Returns
    /// - `[B, T, D]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        self.forward_routed(x).0
    }

    /// [`forward`](Self::forward), with the router's choice over the
    /// flattened `[B T]` rows.
    pub fn forward_routed(
        &self,
        x: Tensor<3>,
    ) -> (Tensor<3>, RouterChoice) {
        let [b, t, d] = unpack_shape_contract!(
            ["B", "T", "D"],
            &x.dims(),
            &["B", "T", "D"],
            &[("D", self.n_embed())]
        );
        let rows = x.reshape([b * t, d]);
        let choice = self.router.forward(rows.clone());
        let rows = match &self.latent_down {
            Some(down) => down.forward(rows),
            None => rows,
        };
        let y = self
            .experts
            .forward(rows, choice.ids.clone(), choice.weights.clone());
        let y = match &self.latent_up {
            Some(up) => up.forward(y),
            None => y,
        };
        (y.reshape([b, t, d]), choice)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        ops::transformers::moe::{
            ExpertDispatchStrategy,
            expert_dispatch,
        },
        support::testing::{
            assert_tensors_close,
            cpu_device,
            seeded_tensor,
        },
    };

    /// With a latent, the layer is the router's choice over the full-width
    /// rows, dispatched in the latent and projected back; without one, the
    /// dispatch on the rows themselves.
    #[test]
    fn test_sparse_moe_is_its_parts() {
        let device = cpu_device();
        for latent in [Some(6), None] {
            let width = latent.unwrap_or(8);
            let moe: SparseMoe = SparseMoeConfig::new(
                TopKRouterConfig::new(8, 5, 2),
                ExpertStackConfig::new(5, width, 12),
            )
            .with_latent(latent)
            .init(&device);
            assert_eq!(moe.n_embed(), 8);
            assert_eq!(moe.latent_down.is_some(), latent.is_some());

            let x: Tensor<3> = seeded_tensor(2, [2, 3, 8], Distribution::Default, &device);
            let (y, choice) = moe.forward_routed(x.clone());
            assert_eq!(y.dims(), [2, 3, 8]);

            let rows = x.reshape([6, 8]);
            let reference_choice = moe.router.forward(rows.clone());
            assert_eq!(
                choice.ids.clone().into_data(),
                reference_choice.ids.clone().into_data()
            );
            let z = match &moe.latent_down {
                Some(down) => down.forward(rows),
                None => rows,
            };
            let out = expert_dispatch(
                z,
                reference_choice.ids,
                reference_choice.weights,
                &moe.experts.packed(),
                ExpertDispatchStrategy::DenseMasked,
            );
            let out = match &moe.latent_up {
                Some(up) => up.forward(out),
                None => out,
            };
            assert_tensors_close(&y, &out.reshape([2, 3, 8]), Tolerance::default());
        }
    }

    #[test]
    fn test_refusals() {
        let device = cpu_device();
        let base = SparseMoeConfig::new(
            TopKRouterConfig::new(8, 4, 2),
            ExpertStackConfig::new(4, 6, 12),
        )
        .with_latent(Some(6));
        ErrorMatcher::kind(BunsenErrorKind::Unsupported).assert_err(
            &base
                .clone()
                .with_shared(Some(SharedExpertConfig::new(16)))
                .try_init(&device),
        );
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("experts 6 wide")
            .assert_err(&base.clone().with_latent(None).try_init(&device));
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .message_contains("routed over 5")
            .assert_err(
                &SparseMoeConfig {
                    router: TopKRouterConfig::new(8, 5, 2),
                    ..base.clone()
                }
                .try_init(&device),
            );
    }
}
