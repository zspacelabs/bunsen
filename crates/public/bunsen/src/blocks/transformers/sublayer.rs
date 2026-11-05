//! # The residual sublayer
//!
//! A transformer layer is residual sublayers in turn: each a branch (the
//! attention, the MLP, a convolution) whose output is added back to the
//! stream, with normalizations around it. [`SublayerNorms`] names those
//! normalizations, borrowed from the block that holds them, and applies them
//! around a branch the block hands it as a closure: the block keeps its own
//! fields, so its record (and every checkpoint map onto it) is unchanged.
//!
//! In the field's terms, the *residual* is the branch's contribution,
//! `post(f(pre(x)))`, and the *residual update* is the stream after it,
//! `out(x + residual)`, with `x` the stream after the `input` norm.

use burn::Tensor;

use crate::blocks::norm::BunsenNormalization;

/// The normalizations of one residual sublayer, borrowed from its block, in
/// their four slots:
///
/// 1. `input`, on the stream before the fork: both the skip and the branch see
///    `input(x)`;
/// 2. `pre`, on the branch's input (a pre-norm block's);
/// 3. `post`, on the branch's output (the sandwich: Gemma 2's and 3's
///    post-norms);
/// 4. `out`, after the add (a post-norm block's).
///
/// An empty slot is the identity: with none, the sublayer is `x + f(x)`.
///
/// ```rust
/// use bunsen::blocks::transformers::sublayer::SublayerNorms;
/// # use bunsen::blocks::norm::BunsenNormalization;
/// # fn sketch(mlp_norm: &BunsenNormalization, x: burn::Tensor<3>) -> burn::Tensor<3> {
/// // A pre-norm MLP sublayer: `x + mlp(norm(x))`.
/// SublayerNorms::pre(Some(mlp_norm)).residual_update(x, |z| z /* the MLP */)
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct SublayerNorms<'a> {
    /// On the stream, before the fork.
    pub input: Option<&'a BunsenNormalization>,

    /// On the branch's input.
    pub pre: Option<&'a BunsenNormalization>,

    /// On the branch's output.
    pub post: Option<&'a BunsenNormalization>,

    /// After the add.
    pub out: Option<&'a BunsenNormalization>,
}

impl<'a> SublayerNorms<'a> {
    /// A pre-norm sublayer's: `pre` alone.
    pub fn pre(pre: Option<&'a BunsenNormalization>) -> Self {
        Self {
            pre,
            ..Self::default()
        }
    }

    /// A post-norm sublayer's: `out` alone, after the add.
    pub fn out(out: Option<&'a BunsenNormalization>) -> Self {
        Self {
            out,
            ..Self::default()
        }
    }

    /// With `post` on the branch's output.
    pub fn with_post(
        self,
        post: Option<&'a BunsenNormalization>,
    ) -> Self {
        Self { post, ..self }
    }

    /// With `input` on the stream, before the fork.
    pub fn with_input(
        self,
        input: Option<&'a BunsenNormalization>,
    ) -> Self {
        Self { input, ..self }
    }

    /// `x` through `norm`, or as it is in an empty slot.
    fn apply<const D: usize>(
        norm: Option<&BunsenNormalization>,
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
        Self::apply(self.pre, x)
    }

    /// The residual: `post(f(pre(x)))`, `x` the stream after `input`.
    pub fn residual<const D: usize>(
        &self,
        x: Tensor<D>,
        f: impl FnOnce(Tensor<D>) -> Tensor<D>,
    ) -> Tensor<D> {
        Self::apply(self.post, f(self.branch_input(x)))
    }

    /// The residual update: `out(x + post(f(pre(x))))`, with `x` first
    /// through `input`.
    pub fn residual_update<const D: usize>(
        &self,
        x: Tensor<D>,
        f: impl FnOnce(Tensor<D>) -> Tensor<D>,
    ) -> Tensor<D> {
        let x = Self::apply(self.input, x);
        let residual = self.residual(x.clone(), f);
        Self::apply(self.out, x + residual)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        module::Param,
        nn::RmsNormConfig,
        tensor::Tolerance,
    };

    use super::*;
    use crate::{
        blocks::norm::BunsenNormalizationConfig,
        support::testing::{
            assert_tensors_close,
            cpu_device,
        },
    };

    /// An RMS norm whose gain is `g`, so each slot's norm is told apart.
    fn norm(g: f32) -> BunsenNormalization {
        let device = cpu_device();
        let mut norm = BunsenNormalizationConfig::Rms(RmsNormConfig::new(3)).init(&device);
        let BunsenNormalization::Rms(rms) = &mut norm else {
            unreachable!()
        };
        rms.gamma = Param::from_tensor(Tensor::from_floats([g, g, g], &device));
        norm
    }

    fn x() -> Tensor<2> {
        Tensor::from_floats([[1.0, -2.0, 4.0], [0.5, 0.25, -1.0]], &cpu_device())
    }

    /// A branch that tells its input apart: `2 z + 1`.
    fn f(z: Tensor<2>) -> Tensor<2> {
        z.mul_scalar(2.0).add_scalar(1.0)
    }

    /// Each slot alone, and all four, are their hand-written forms.
    #[test]
    fn test_slots() {
        let (input, pre, post, out) = (norm(1.5), norm(0.5), norm(2.0), norm(3.0));
        let tol = Tolerance::default();
        let x = x();
        assert_tensors_close(
            &SublayerNorms::default().residual_update(x.clone(), f),
            &(x.clone() + f(x.clone())),
            tol,
        );
        assert_tensors_close(
            &SublayerNorms::pre(Some(&pre)).residual_update(x.clone(), f),
            &(x.clone() + f(pre.forward(x.clone()))),
            tol,
        );
        assert_tensors_close(
            &SublayerNorms::pre(Some(&pre))
                .with_post(Some(&post))
                .residual_update(x.clone(), f),
            &(x.clone() + post.forward(f(pre.forward(x.clone())))),
            tol,
        );
        assert_tensors_close(
            &SublayerNorms::out(Some(&out)).residual_update(x.clone(), f),
            &out.forward(x.clone() + f(x.clone())),
            tol,
        );
        let all = SublayerNorms {
            input: Some(&input),
            pre: Some(&pre),
            post: Some(&post),
            out: Some(&out),
        };
        let xi = input.forward(x.clone());
        assert_tensors_close(
            &all.residual_update(x.clone(), f),
            &out.forward(xi.clone() + post.forward(f(pre.forward(xi.clone())))),
            tol,
        );
        assert_tensors_close(
            &all.residual(xi.clone(), f),
            &post.forward(f(pre.forward(xi))),
            tol,
        );
    }
}
