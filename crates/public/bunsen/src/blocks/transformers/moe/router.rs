//! # The top-k router

use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::Int,
    tensor::{
        DType,
        Device,
        activation::softmax,
    },
};

use crate::{
    blocks::norm::{
        BunsenNormalization,
        BunsenNormalizationConfig,
    },
    burner::module::ModuleInit,
    contracts::unpack_shape_contract,
    errors::{
        BunsenError,
        BunsenResult,
    },
};

/// Common meta for [`TopKRouter`] and [`TopKRouterConfig`].
pub trait TopKRouterMeta {
    /// The width the router reads.
    fn width(&self) -> usize;

    /// The expert count.
    fn n_experts(&self) -> usize;

    /// The experts each row is routed to.
    fn top_k(&self) -> usize;
}

/// How a router turns its logits into scores.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum RouterScore {
    /// A softmax over the experts (Qwen, `OLMoE`, ZONOS3).
    Softmax,

    /// A sigmoid per expert (`DeepSeek` V3). Not built: refused as
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported).
    Sigmoid,
}

/// How a router picks each row's experts.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum RouterSelection {
    /// The `k` best of all experts.
    TopK,

    /// The `k` best within the `top_groups` best of `n_groups` equal groups
    /// (`DeepSeek`'s `n_group` / `topk_group`). Not built: refused as
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported).
    Grouped {
        /// The groups the experts are split into.
        n_groups: usize,

        /// The groups a row's experts may come from.
        top_groups: usize,
    },
}

/// Where a router's [`ChoiceBias`] applies.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum ChoiceBiasSite {
    /// On the raw logits (ZONOS3's `balancing_biases`).
    Logits,

    /// On the scores (`DeepSeek` V3's `e_score_correction_bias`). Not built:
    /// refused as [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported).
    Scores,
}

/// How a router's [`ChoiceBias`] combines with what it biases.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub enum ChoiceBiasMode {
    /// `x + b`.
    Add,

    /// `x - b` (ZONOS3's).
    Subtract,
}

/// A learned per-expert bias on a router's choice alone: the experts are
/// picked by the biased values, and weighted by the unbiased ones (a
/// load-balancing correction). The bias, `[E]`, is kept in f32 and stored
/// with the checkpoint's sign; [`mode`](Self::mode) says how it applies.
#[derive(Config, Copy, Debug, PartialEq, Eq)]
pub struct ChoiceBias {
    /// What it biases.
    pub site: ChoiceBiasSite,

    /// How.
    pub mode: ChoiceBiasMode,
}

/// Config for [`TopKRouter`].
#[derive(Config, Debug)]
pub struct TopKRouterConfig {
    /// The width the router reads.
    pub width: usize,

    /// The expert count.
    pub n_experts: usize,

    /// The experts each row is routed to.
    pub top_k: usize,

    /// How the logits become scores.
    #[config(default = "RouterScore::Softmax")]
    pub score: RouterScore,

    /// How the experts are picked.
    #[config(default = "RouterSelection::TopK")]
    pub selection: RouterSelection,

    /// A bias on the choice alone; `None`, none.
    #[config(default = "None")]
    pub choice_bias: Option<ChoiceBias>,

    /// Whether the chosen experts' weights are renormalized to sum to one
    /// (`transformers`' `norm_topk_prob`). Under a softmax score, the weights
    /// are then the softmax of the chosen logits alone.
    #[config(default = "true")]
    pub norm_top_k: bool,

    /// A factor on the weights (`DeepSeek`'s `routed_scaling_factor`).
    #[config(default = "1.0")]
    pub scale: f64,

    /// A normalization of the router's input, its own (ZONOS3's
    /// `rmsnorm_eda`); `None`, none. Its feature count is set to
    /// [`width`](Self::width).
    #[config(default = "None")]
    pub input_norm: Option<BunsenNormalizationConfig>,
}

impl TopKRouterMeta for TopKRouterConfig {
    fn width(&self) -> usize {
        self.width
    }

    fn n_experts(&self) -> usize {
        self.n_experts
    }

    fn top_k(&self) -> usize {
        self.top_k
    }
}

impl ModuleInit<TopKRouter> for TopKRouterConfig {
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// sigmoid score, a grouped selection or a bias on the scores;
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a `top_k` of
    /// 0 or more than the experts.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<TopKRouter> {
        if self.score != RouterScore::Softmax {
            return Err(BunsenError::unsupported(format!(
                "router score {:?}",
                self.score
            )));
        }
        if self.selection != RouterSelection::TopK {
            return Err(BunsenError::unsupported(format!(
                "router selection {:?}",
                self.selection
            )));
        }
        if let Some(bias) = self.choice_bias
            && bias.site != ChoiceBiasSite::Logits
        {
            return Err(BunsenError::unsupported(format!(
                "a choice bias on the {:?}",
                bias.site
            )));
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return Err(BunsenError::illegal(format!(
                "top_k {} of {} experts",
                self.top_k, self.n_experts
            )));
        }
        Ok(TopKRouter {
            gate: LinearConfig::new(self.width, self.n_experts)
                .with_bias(false)
                .init(device),
            input_norm: self
                .input_norm
                .as_ref()
                .map(|n| n.clone().with_num_features(self.width).init(device)),
            choice_bias: self
                .choice_bias
                .map(|_| Param::from_tensor(Tensor::zeros([self.n_experts], (device, DType::F32)))),
            bias: self.choice_bias,
            top_k: self.top_k,
            norm_top_k: self.norm_top_k,
            scale: self.scale,
        })
    }
}

/// A router's choice for a batch of rows.
#[derive(Clone, Debug)]
pub struct RouterChoice {
    /// `[N, k]`: each row's experts, best first.
    pub ids: Tensor<2, Int>,

    /// `[N, k]`: their weights, in f32.
    pub weights: Tensor<2>,

    /// `[N, E]`: the logits, in f32.
    pub logits: Tensor<2>,
}

/// A mixture of experts' router: a linear score per expert, each row's
/// `k` best experts and their weights.
///
/// Built by [`TopKRouterConfig`].
#[derive(Module, Debug)]
pub struct TopKRouter {
    /// The logits' projection, `[W, E]`, without a bias.
    pub gate: Linear,

    /// The input's normalization, if the router has one.
    pub input_norm: Option<BunsenNormalization>,

    /// The choice bias `[E]`, in f32, if the router has one.
    pub choice_bias: Option<Param<Tensor<1>>>,

    /// Where and how [`choice_bias`](Self::choice_bias) applies.
    #[module(skip)]
    pub bias: Option<ChoiceBias>,

    /// The experts each row is routed to.
    #[module(skip)]
    pub top_k: usize,

    /// Whether the chosen weights are renormalized.
    #[module(skip)]
    pub norm_top_k: bool,

    /// A factor on the weights.
    #[module(skip)]
    pub scale: f64,
}

impl TopKRouterMeta for TopKRouter {
    fn width(&self) -> usize {
        self.gate.weight.dims()[0]
    }

    fn n_experts(&self) -> usize {
        self.gate.weight.dims()[1]
    }

    fn top_k(&self) -> usize {
        self.top_k
    }
}

impl TopKRouter {
    /// Routes each row: its `k` experts by their (biased) logits, and their
    /// weights by the unbiased ones.
    ///
    /// ```text
    /// logits  = gate(input_norm(x))               (in f32)
    /// ids     = topk(logits ± bias)
    /// weights = softmax(logits[ids])              (norm_top_k)
    ///         | softmax(logits)[ids]              (otherwise)
    /// weights = scale · weights
    /// ```
    ///
    /// # Arguments
    /// - `x`: `[N, W]`, the rows.
    ///
    /// # Returns
    /// - the [`RouterChoice`].
    pub fn forward(
        &self,
        x: Tensor<2>,
    ) -> RouterChoice {
        let [n] = unpack_shape_contract!(["N", "W"], &x.dims(), &["N"], &[("W", self.width())]);
        let x = match &self.input_norm {
            Some(norm) => norm.forward(x),
            None => x,
        };
        let logits = self.gate.forward(x).cast(DType::F32);
        let key = match (&self.choice_bias, self.bias) {
            (Some(bias), Some(choice)) => {
                let bias = bias.val().cast(DType::F32).unsqueeze::<2>();
                match choice.mode {
                    ChoiceBiasMode::Add => logits.clone() + bias,
                    ChoiceBiasMode::Subtract => logits.clone() - bias,
                }
            }
            _ => logits.clone(),
        };
        let (_, ids) = key.topk_with_indices(self.top_k, 1);
        let weights = if self.norm_top_k {
            softmax(logits.clone().gather(1, ids.clone()), 1)
        } else {
            softmax(logits.clone(), 1).gather(1, ids.clone())
        };
        let weights = if self.scale == 1.0 {
            weights
        } else {
            weights.mul_scalar(self.scale)
        };
        crate::contracts::assert_shape_contract_periodically!(
            ["N", "k"],
            &ids.dims(),
            &[("N", n), ("k", self.top_k)]
        );
        RouterChoice {
            ids,
            weights,
            logits,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        nn::norm::RmsNormConfig,
        tensor::{
            Distribution,
            Tolerance,
        },
    };

    use super::*;
    use crate::{
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::{
            assert_tensors_close,
            cpu_device,
            seeded_tensor,
        },
    };

    /// A router whose logits are its input: an identity gate over 4 experts.
    fn identity_router(config: TopKRouterConfig) -> TopKRouter {
        let device = cpu_device();
        let mut router: TopKRouter = config.init(&device);
        router.gate.weight = Param::from_tensor(Tensor::eye(4, &device));
        router
    }

    fn ids(choice: &RouterChoice) -> Vec<i64> {
        choice
            .ids
            .clone()
            .into_data()
            .try_into_vec_as::<i64>()
            .unwrap()
    }

    /// The bias moves the choice, not the weights: subtracted, it demotes
    /// the best logit; added, it promotes a poor one. The weights are the
    /// softmax of the chosen raw logits.
    #[test]
    fn test_choice_bias() {
        let device = cpu_device();
        let x = Tensor::<2>::from_data([[3.0f32, 2.0, 1.0, 0.0]], &device);
        let bias = Param::from_tensor(Tensor::<1>::from_data([2.5f32, 0.0, 0.0, 0.0], &device));

        let plain = identity_router(TopKRouterConfig::new(4, 4, 2));
        assert!(plain.choice_bias.is_none());
        assert_eq!(ids(&plain.forward(x.clone())), [0, 1]);

        let mut subtract = identity_router(TopKRouterConfig::new(4, 4, 2).with_choice_bias(Some(
            ChoiceBias::new(ChoiceBiasSite::Logits, ChoiceBiasMode::Subtract),
        )));
        subtract.choice_bias = Some(bias.clone());
        let choice = subtract.forward(x.clone());
        // 3 - 2.5 = 0.5 falls below 2 and 1.
        assert_eq!(ids(&choice), [1, 2]);
        let expected = softmax(Tensor::<2>::from_data([[2.0f32, 1.0]], &device), 1);
        assert_tensors_close(&choice.weights, &expected, Tolerance::default());

        let mut add = identity_router(TopKRouterConfig::new(4, 4, 2).with_choice_bias(Some(
            ChoiceBias::new(ChoiceBiasSite::Logits, ChoiceBiasMode::Add),
        )));
        add.choice_bias = Some(Param::from_tensor(Tensor::<1>::from_data(
            [0.0f32, 0.0, 0.0, 2.5],
            &device,
        )));
        // 0 + 2.5 rises above 2.
        let choice = add.forward(x);
        assert_eq!(ids(&choice), [0, 3]);
        let expected = softmax(Tensor::<2>::from_data([[3.0f32, 0.0]], &device), 1);
        assert_tensors_close(&choice.weights, &expected, Tolerance::default());
    }

    /// Without renormalization the weights are the full softmax's at the
    /// chosen experts; `scale` multiplies them; an input norm is applied
    /// before the gate.
    #[test]
    fn test_weights_and_norm() {
        let device = cpu_device();
        let x = Tensor::<2>::from_data([[3.0f32, 2.0, 1.0, 0.0]], &device);
        let router = identity_router(
            TopKRouterConfig::new(4, 4, 2)
                .with_norm_top_k(false)
                .with_scale(2.0),
        );
        let full = softmax(x.clone(), 1);
        let expected = full.slice_dim(1, 0..2).mul_scalar(2.0);
        assert_tensors_close(
            &router.forward(x.clone()).weights,
            &expected,
            Tolerance::default(),
        );

        let normed = identity_router(TopKRouterConfig::new(4, 4, 2).with_input_norm(Some(
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(0).with_epsilon(1e-6)),
        )));
        let rms = (x.clone().square().mean_dim(1).add_scalar(1e-6)).sqrt();
        assert_tensors_close(
            &normed.forward(x.clone()).logits,
            &(x / rms),
            Tolerance::default(),
        );
        let rows: Tensor<2> = seeded_tensor(1, [5, 4], Distribution::Default, &device);
        let choice = normed.forward(rows);
        assert_eq!(choice.ids.dims(), [5, 2]);
        assert_eq!(choice.weights.dims(), [5, 2]);
    }

    /// The variants designed for other families are refused, not run.
    #[test]
    fn test_refusals() {
        let device = cpu_device();
        let base = TopKRouterConfig::new(4, 4, 2);
        for config in [
            base.clone().with_score(RouterScore::Sigmoid),
            base.clone().with_selection(RouterSelection::Grouped {
                n_groups: 2,
                top_groups: 1,
            }),
            base.clone().with_choice_bias(Some(ChoiceBias::new(
                ChoiceBiasSite::Scores,
                ChoiceBiasMode::Add,
            ))),
        ] {
            ErrorMatcher::kind(BunsenErrorKind::Unsupported).assert_err(&config.try_init(&device));
        }
        for top_k in [0, 5] {
            ErrorMatcher::kind(BunsenErrorKind::Illegal)
                .assert_err(&TopKRouterConfig::new(4, 4, top_k).try_init(&device));
        }
    }
}
