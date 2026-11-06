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
//!
//! A branch may return more than its residual (an attention's weights): its
//! output is a [`BranchOutput`], which says where the residual is and what
//! the update gives back, the stream alone or the stream with the rest.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::attention::MhaOutput,
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

    /// The residual update: `out(x + post(r))`, with `x` first through
    /// `input`, and `r` the residual in `f(branch, pre(x))`'s output; the
    /// output's [`Updated`](BranchOutput::Updated) form, the new stream in the
    /// residual's place.
    pub fn residual_update_with<const D: usize, O: BranchOutput<D>>(
        &self,
        x: Tensor<D>,
        f: impl FnOnce(&M, Tensor<D>) -> O,
    ) -> O::Updated {
        let x = Self::apply(&self.input, x);
        f(&self.branch, self.branch_input(x.clone()))
            .into_updated(|residual| Self::apply(&self.out, x + Self::apply(&self.post, residual)))
    }

    /// [`residual_update_with`](Self::residual_update_with), `f` also given
    /// `ctx`: a call's context, for an `f` written once and passed around.
    pub fn residual_update_with_ctx<const D: usize, C, O: BranchOutput<D>>(
        &self,
        x: Tensor<D>,
        ctx: C,
        f: impl FnOnce(&M, Tensor<D>, C) -> O,
    ) -> O::Updated {
        self.residual_update_with(x, |branch, z| f(branch, z, ctx))
    }
}

/// What a [`ResidualSublayer`]'s branch returns: its residual, and anything
/// that comes with it.
///
/// A residual update takes the residual out, and gives back
/// [`Updated`](Self::Updated): the new stream alone for a plain tensor, or
/// the new stream with the rest of the output (a tuple, a struct).
pub trait BranchOutput<const D: usize> {
    /// What the residual update returns.
    type Updated;

    /// This output, its residual `r` replaced by `update(r)`, the new stream.
    fn into_updated(
        self,
        update: impl FnOnce(Tensor<D>) -> Tensor<D>,
    ) -> Self::Updated;
}

/// The residual alone: the update returns the stream.
impl<const D: usize> BranchOutput<D> for Tensor<D> {
    type Updated = Tensor<D>;

    fn into_updated(
        self,
        update: impl FnOnce(Tensor<D>) -> Tensor<D>,
    ) -> Tensor<D> {
        update(self)
    }
}

/// An attention's output: the update returns the stream (in the context's
/// place) and the attention weights, `(stream, weights)`.
impl BranchOutput<3> for MhaOutput {
    type Updated = (Tensor<3>, Tensor<4>);

    fn into_updated(
        self,
        update: impl FnOnce(Tensor<3>) -> Tensor<3>,
    ) -> (Tensor<3>, Tensor<4>) {
        (update(self.context), self.weights)
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
            &s.residual_update_with(x.clone(), f),
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
            &s.residual_update_with(x.clone(), f),
            &(x.clone() + n(&s.post, s.branch.forward(n(&s.pre, x.clone())))),
            tol,
        );

        let s = sublayer(ResidualSublayerConfig::post_norm(rms()));
        assert_tensors_close(
            &s.residual_update_with(x.clone(), f),
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
            &s.residual_update_with(x.clone(), f),
            &n(
                &s.out,
                xi.clone() + n(&s.post, s.branch.forward(n(&s.pre, xi.clone()))),
            ),
            tol,
        );
    }

    /// A branch output with a side output: the stream, and what rides along.
    struct WithAux {
        residual: Tensor<2>,
        aux: f32,
    }

    impl BranchOutput<2> for WithAux {
        type Updated = (Tensor<2>, f32);

        fn into_updated(
            self,
            update: impl FnOnce(Tensor<2>) -> Tensor<2>,
        ) -> (Tensor<2>, f32) {
            (update(self.residual), self.aux)
        }
    }

    /// A side output comes back beside the stream, untouched; the stream is
    /// the plain form's; the context form passes its context to `f`.
    #[test]
    fn test_branch_outputs() {
        let tol = Tolerance::default();
        let x = x();
        let s = sublayer(
            ResidualSublayerConfig::new()
                .with_input(rms())
                .with_pre(rms())
                .with_post(rms())
                .with_out(rms()),
        );
        let plain = s.residual_update_with(x.clone(), |l, z| l.forward(z));

        let (stream, aux) = s.residual_update_with(x.clone(), |l, z| WithAux {
            residual: l.forward(z),
            aux: 7.0,
        });
        assert_tensors_close(&stream, &plain, tol);
        assert_eq!(aux, 7.0);

        let scaled =
            s.residual_update_with_ctx(x.clone(), 2.0, |l, z, k| l.forward(z).mul_scalar(k));
        let expected = s.residual_update_with(x, |l, z| l.forward(z).mul_scalar(2.0));
        assert_tensors_close(&scaled, &expected, tol);
    }
}
