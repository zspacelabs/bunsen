//! # The residual sublayer
//!
//! A transformer layer is residual sublayers in turn: each a branch (the
//! attention, the MLP, a convolution) whose output is added back to the
//! stream, with normalizations around it. [`ResidualSublayer`] owns them: its
//! branch module and its four norm slots, so a block holds one per sublayer,
//! and its record nests each under the sublayer's name (`mlp.pre`,
//! `mlp.branch`, `mlp.post`).
//!
//! In the field's terms, the *residual* is the branch's contribution,
//! `post(f(pre(x)))`, and the *residual update* is the stream after it,
//! `out(x + residual)`, with `x` the stream after the `input` norm.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::Device,
};

use crate::blocks::norm::{
    BunsenNormalization,
    BunsenNormalizationConfig,
};

/// Config for a [`ResidualSublayer`]'s norms: each slot's norm, sized to the
/// sublayer's width at init, or `None` for an empty slot.
#[derive(Config, Debug)]
pub struct ResidualSublayerConfig {
    /// On the stream, before the fork.
    #[config(default = "None")]
    pub input: Option<BunsenNormalizationConfig>,

    /// On the branch's input (a pre-norm block's).
    #[config(default = "None")]
    pub pre: Option<BunsenNormalizationConfig>,

    /// On the branch's output (the sandwich).
    #[config(default = "None")]
    pub post: Option<BunsenNormalizationConfig>,

    /// After the add (a post-norm block's).
    #[config(default = "None")]
    pub out: Option<BunsenNormalizationConfig>,
}

impl ResidualSublayerConfig {
    /// A pre-norm sublayer's: `pre` alone.
    pub fn pre_norm(pre: Option<BunsenNormalizationConfig>) -> Self {
        Self::new().with_pre(pre)
    }

    /// A post-norm sublayer's: `out` alone.
    pub fn post_norm(out: Option<BunsenNormalizationConfig>) -> Self {
        Self::new().with_out(out)
    }

    /// The sublayer around `branch`, its norms `width` wide, on `device`.
    pub fn init<M: Module>(
        &self,
        branch: M,
        width: usize,
        device: &Device,
    ) -> ResidualSublayer<M> {
        let norm = |c: &Option<BunsenNormalizationConfig>| {
            c.as_ref()
                .map(|c| c.clone().with_num_features(width).init(device))
        };
        ResidualSublayer {
            input: norm(&self.input),
            pre: norm(&self.pre),
            branch,
            post: norm(&self.post),
            out: norm(&self.out),
        }
    }
}

/// A residual sublayer: a branch module and the normalizations around it, in
/// four slots:
///
/// 1. `input`, on the stream before the fork: both the skip and the branch see
///    `input(x)`;
/// 2. `pre`, on the branch's input (a pre-norm block's);
/// 3. `post`, on the branch's output (the sandwich: Gemma 2's and 3's
///    post-norms);
/// 4. `out`, after the add (a post-norm block's).
///
/// An empty slot is the identity, and holds no parameters.
///
/// Built by [`ResidualSublayerConfig::init`] around its branch.
#[derive(Module, Debug)]
pub struct ResidualSublayer<M: Module> {
    /// On the stream, before the fork.
    pub input: Option<BunsenNormalization>,

    /// On the branch's input.
    pub pre: Option<BunsenNormalization>,

    /// The branch: the attention, the MLP, a convolution.
    pub branch: M,

    /// On the branch's output.
    pub post: Option<BunsenNormalization>,

    /// After the add.
    pub out: Option<BunsenNormalization>,
}

impl<M: Module> ResidualSublayer<M> {
    /// `x` through `norm`, or as it is in an empty slot.
    fn apply<const D: usize>(
        norm: &Option<BunsenNormalization>,
        x: Tensor<D>,
    ) -> Tensor<D> {
        match norm {
            Some(norm) => norm.forward(x),
            None => x,
        }
    }

    /// The branch's input: `pre(x)`, `x` the stream after `input`.
    pub fn branch_input<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Tensor<D> {
        Self::apply(&self.pre, x)
    }

    /// The residual: `post(f(branch, pre(x)))`, `x` the stream after
    /// `input`; `f` runs the branch with whatever else it takes.
    pub fn residual<const D: usize>(
        &self,
        x: Tensor<D>,
        f: impl FnOnce(&M, Tensor<D>) -> Tensor<D>,
    ) -> Tensor<D> {
        Self::apply(&self.post, f(&self.branch, self.branch_input(x)))
    }

    /// The residual update: `out(x + post(f(branch, pre(x))))`, with `x`
    /// first through `input`.
    pub fn residual_update<const D: usize>(
        &self,
        x: Tensor<D>,
        f: impl FnOnce(&M, Tensor<D>) -> Tensor<D>,
    ) -> Tensor<D> {
        let x = Self::apply(&self.input, x);
        let residual = self.residual(x.clone(), f);
        Self::apply(&self.out, x + residual)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        nn::{
            Linear,
            LinearConfig,
            RmsNormConfig,
        },
        tensor::Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
    };

    fn rms() -> Option<BunsenNormalizationConfig> {
        Some(BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)))
    }

    /// A sublayer over a `Linear`, every slot's norm with a gain of its own.
    fn sublayer(config: ResidualSublayerConfig) -> ResidualSublayer<Linear> {
        let device = cpu_device();
        let mut s = config.init(LinearConfig::new(3, 3).init(&device), 3, &device);
        for (g, norm) in [
            (1.5, &mut s.input),
            (0.5, &mut s.pre),
            (2.0, &mut s.post),
            (3.0, &mut s.out),
        ] {
            if let Some(BunsenNormalization::Rms(r)) = norm {
                r.gamma = Param::from_tensor(Tensor::from_floats([g, g, g], &device));
            }
        }
        s
    }

    fn x() -> Tensor<2> {
        Tensor::from_floats([[1.0, -2.0, 4.0], [0.5, 0.25, -1.0]], &cpu_device())
    }

    fn n(
        norm: &Option<BunsenNormalization>,
        x: Tensor<2>,
    ) -> Tensor<2> {
        norm.as_ref().unwrap().forward(x)
    }

    /// Each slot alone, and all four, are their hand-written forms; an empty
    /// slot holds no parameters.
    #[test]
    fn test_slots() {
        let tol = Tolerance::default();
        let x = x();
        let f = |l: &Linear, z: Tensor<2>| l.forward(z);

        let s = sublayer(ResidualSublayerConfig::new());
        assert_tensors_close(
            &s.residual_update(x.clone(), f),
            &(x.clone() + s.branch.forward(x.clone())),
            tol,
        );
        #[cfg(feature = "store")]
        {
            use burn::store::ModuleSnapshot;
            let names: Vec<String> = s
                .collect(None, None, false)
                .into_iter()
                .map(|t| t.name)
                .collect();
            assert_eq!(names, ["branch.weight", "branch.bias"]);
        }

        let s = sublayer(ResidualSublayerConfig::pre_norm(rms()).with_post(rms()));
        assert_tensors_close(
            &s.residual_update(x.clone(), f),
            &(x.clone() + n(&s.post, s.branch.forward(n(&s.pre, x.clone())))),
            tol,
        );

        let s = sublayer(ResidualSublayerConfig::post_norm(rms()));
        assert_tensors_close(
            &s.residual_update(x.clone(), f),
            &n(&s.out, x.clone() + s.branch.forward(x.clone())),
            tol,
        );

        let s = sublayer(
            ResidualSublayerConfig::new()
                .with_input(rms())
                .with_pre(rms())
                .with_post(rms())
                .with_out(rms()),
        );
        let xi = n(&s.input, x.clone());
        assert_tensors_close(
            &s.residual_update(x.clone(), f),
            &n(
                &s.out,
                xi.clone() + n(&s.post, s.branch.forward(n(&s.pre, xi.clone()))),
            ),
            tol,
        );
    }
}
