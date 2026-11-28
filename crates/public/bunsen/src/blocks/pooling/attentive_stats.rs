//! [`AttentiveStatsPooling`]: attentive statistics pooling.

use alloc::vec;

use burn::{
    config::Config,
    module::Module,
    prelude::Tensor,
    tensor::{
        Device,
        activation::softmax,
    },
};

use crate::{
    blocks::{
        activation::BunsenActivationConfig,
        mlp::{
            BunsenMlp,
            BunsenMlpConfig,
            BunsenMlpMeta,
        },
    },
    burner::module::ModuleInit,
    errors::BunsenResult,
};

/// How [`AttentiveStatsPooling`] takes a weighted standard deviation; the
/// variance is floored before the root.
///
/// The two forms are equal in exact arithmetic and differ in rounding, so a
/// port follows its reference's.
#[derive(Config, Copy, Debug, PartialEq)]
pub enum AttentiveStatsDeviation {
    /// `sqrt(max(Σ w x² - μ², floor))`, the raw second moment less the
    /// squared mean.
    RawMoment {
        /// The variance's floor.
        floor: f64,
    },

    /// `sqrt(max(Σ w (x - μ)², floor))`, the centered second moment.
    Centered {
        /// The variance's floor.
        floor: f64,
    },
}

impl AttentiveStatsDeviation {
    /// The weighted mean and standard deviation of `x`, `[B, C, T]`, under
    /// weights `w` that sum to one over time, `[B, C, T]` or `[B, 1, T]`:
    /// each `[B, C, 1]`.
    pub fn weighted_mean_std(
        &self,
        x: Tensor<3>,
        w: Tensor<3>,
    ) -> (Tensor<3>, Tensor<3>) {
        #[cfg(any(debug_assertions, test))]
        let [b, c, t] = {
            let [b, c, t] = crate::contracts::unpack_shape_contract!(["B", "C", "T"], &x.dims());
            crate::contracts::assert_shape_contract_periodically!(
                ["B", _, "T"],
                &w.dims(),
                &[("B", b), ("T", t)]
            );
            [b, c, t]
        };
        let mean = (w.clone() * x.clone()).sum_dim(2);
        let variance = match self {
            Self::RawMoment { .. } => (w * x.clone() * x).sum_dim(2) - mean.clone() * mean.clone(),
            Self::Centered { .. } => {
                let centered = x - mean.clone();
                (w * centered.clone() * centered).sum_dim(2)
            }
        };
        let floor = match self {
            Self::RawMoment { floor } | Self::Centered { floor } => *floor,
        };
        let std = variance.clamp_min(floor).sqrt();
        #[cfg(any(debug_assertions, test))]
        {
            let _ = t;
            for stat in [&mean, &std] {
                crate::contracts::assert_shape_contract_periodically!(
                    ["B", "C", 1],
                    &stat.dims(),
                    &[("B", b), ("C", c)]
                );
            }
        }
        (mean, std)
    }
}

/// Common meta for [`AttentiveStatsPooling`] and
/// [`AttentiveStatsPoolingConfig`].
pub trait AttentiveStatsPoolingMeta {
    /// The channels, `C`.
    fn channels(&self) -> usize;

    /// The attention's hidden width.
    fn bottleneck(&self) -> usize;

    /// Whether each frame's attention also sees the sequence's mean and
    /// deviation.
    fn global_context(&self) -> bool;

    /// The attention's input width: `C`, or `3 C` with the global context.
    fn attention_input(&self) -> usize {
        if self.global_context() {
            3 * self.channels()
        } else {
            self.channels()
        }
    }

    /// The pooled width, `2 C`: the means, then the deviations.
    fn d_output(&self) -> usize {
        2 * self.channels()
    }
}

/// Config for [`AttentiveStatsPooling`].
///
/// The defaults are a tanh between the attention's layers, no global context,
/// and the raw-moment deviation floored at `1e-7`. The other common form is a
/// `ReLU` then a tanh ([`Chain`](BunsenActivationConfig::Chain)), the global
/// context, and the centered deviation floored at `1e-12`.
///
/// Implements [`AttentiveStatsPoolingMeta`].
///
/// ## Discussion
///
/// The defaults are ASTP's as `WeSpeaker` and `AntSpeaker` build it, the
/// [`RawMoment`](AttentiveStatsDeviation::RawMoment) deviation; the other form
/// is ECAPA-TDNN's pooling as `SpeechBrain` builds it, the
/// [`Centered`](AttentiveStatsDeviation::Centered) deviation.
#[derive(Config, Debug)]
pub struct AttentiveStatsPoolingConfig {
    /// The channels, `C`.
    pub channels: usize,

    /// The attention's hidden width.
    pub bottleneck: usize,

    /// The activation between the attention's layers.
    #[config(default = "BunsenActivationConfig::Tanh")]
    pub act: BunsenActivationConfig,

    /// Whether each frame's attention also sees the sequence's mean and
    /// deviation, by the same [`deviation`](Self::deviation) under uniform
    /// weights.
    #[config(default = false)]
    pub global_context: bool,

    /// How the deviations are taken.
    #[config(default = "AttentiveStatsDeviation::RawMoment { floor: 1e-7 }")]
    pub deviation: AttentiveStatsDeviation,
}

impl AttentiveStatsPoolingMeta for AttentiveStatsPoolingConfig {
    fn channels(&self) -> usize {
        self.channels
    }

    fn bottleneck(&self) -> usize {
        self.bottleneck
    }

    fn global_context(&self) -> bool {
        self.global_context
    }
}

impl ModuleInit<AttentiveStatsPooling> for AttentiveStatsPoolingConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<AttentiveStatsPooling> {
        Ok(AttentiveStatsPooling {
            attention: BunsenMlpConfig::new(self.attention_input(), self.bottleneck, self.channels)
                .with_act(self.act.clone())
                .try_init(device)?,
            global_context: self.global_context,
            deviation: self.deviation,
        })
    }
}

/// Attentive statistics pooling: a sequence's channels, `[B, C, T]`, to
/// their attention-weighted means and standard deviations, `[B, 2 C]`
/// (Okabe et al., 2018, "Attentive Statistics Pooling for Deep Speaker
/// Embedding").
///
/// Each channel has its own weights over time: a position-wise MLP over the
/// channels (with the global context, over each frame's channels beside the
/// sequence's mean and deviation) gives a score per channel and frame, and a
/// softmax over time makes them weights.
///
/// Built by [`AttentiveStatsPoolingConfig`]. Implements
/// [`AttentiveStatsPoolingMeta`].
#[derive(Module, Debug)]
pub struct AttentiveStatsPooling {
    /// The scores: `attention_input -> bottleneck -> C`, over each frame.
    pub attention: BunsenMlp,

    /// Whether the attention sees the global context.
    pub global_context: bool,

    /// How the deviations are taken.
    #[module(skip)]
    pub deviation: AttentiveStatsDeviation,
}

impl AttentiveStatsPoolingMeta for AttentiveStatsPooling {
    fn channels(&self) -> usize {
        self.attention.d_output()
    }

    fn bottleneck(&self) -> usize {
        self.attention.d_hidden()
    }

    fn global_context(&self) -> bool {
        self.global_context
    }
}

impl AttentiveStatsPooling {
    /// The pooled statistics of `x`.
    ///
    /// # Arguments
    /// - `x`: `[B, C, T]`, channels first.
    ///
    /// # Returns
    /// `[B, 2 C]`: each channel's weighted mean, then its weighted
    /// deviation.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<2> {
        let [batch, channels, frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "channels", "frames"],
            &x,
            &["batch", "channels", "frames"],
            &[("channels", self.channels())],
        );
        let context = if self.global_context {
            let uniform = Tensor::ones([batch, 1, frames], &x.device()) / frames as f64;
            let (mean, std) = self.deviation.weighted_mean_std(x.clone(), uniform);
            Tensor::cat(
                vec![
                    x.clone(),
                    mean.expand([batch, channels, frames]),
                    std.expand([batch, channels, frames]),
                ],
                1,
            )
        } else {
            x.clone()
        };
        let scores = self
            .attention
            .forward(context.swap_dims(1, 2))
            .swap_dims(1, 2);
        let weights = softmax(scores, 2);
        let (mean, std) = self.deviation.weighted_mean_std(x, weights);
        let pooled = Tensor::cat(vec![mean, std], 1).reshape([batch, 2 * channels]);
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "pooled"],
            &pooled.dims(),
            &[("batch", batch), ("pooled", 2 * channels)]
        );
        pooled
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        REORDER,
        ROUNDING,
        assert_tensor_close_to_vec,
        performance_device,
        seeded_tensor,
        tensor_to_vec_as,
    };

    /// The pooling, transcribed on the host in f64: `x` `[C][T]` for one
    /// batch row; `linear1` `[in][hidden]` and `linear2` `[hidden][C]` as
    /// burn stores them.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f64],
        (c, t): (usize, usize),
        (w1, b1, w2, b2): (&[f64], &[f64], &[f64], &[f64]),
        hidden: usize,
        act: impl Fn(f64) -> f64,
        global: bool,
        deviation: AttentiveStatsDeviation,
    ) -> Vec<f64> {
        let at = |ch: usize, f: usize| x[ch * t + f];
        let stats = |w: &dyn Fn(usize, usize) -> f64| -> (Vec<f64>, Vec<f64>) {
            (0..c)
                .map(|ch| {
                    let mean: f64 = (0..t).map(|f| w(ch, f) * at(ch, f)).sum();
                    let var = match deviation {
                        AttentiveStatsDeviation::RawMoment { floor } => {
                            let m2: f64 = (0..t).map(|f| w(ch, f) * at(ch, f).powi(2)).sum();
                            (m2 - mean * mean).max(floor)
                        }
                        AttentiveStatsDeviation::Centered { floor } => (0..t)
                            .map(|f| w(ch, f) * (at(ch, f) - mean).powi(2))
                            .sum::<f64>()
                            .max(floor),
                    };
                    (mean, var.sqrt())
                })
                .unzip()
        };
        let (gm, gs) = stats(&|_, _| 1.0 / t as f64);
        let input = if global { 3 * c } else { c };
        let mut scores = vec![0.0; c * t];
        for f in 0..t {
            let feature: Vec<f64> = (0..input)
                .map(|i| match i / c {
                    0 => at(i, f),
                    1 => gm[i - c],
                    _ => gs[i - 2 * c],
                })
                .collect();
            let h: Vec<f64> = (0..hidden)
                .map(|j| {
                    act(b1[j]
                        + (0..input)
                            .map(|i| feature[i] * w1[i * hidden + j])
                            .sum::<f64>())
                })
                .collect();
            for ch in 0..c {
                scores[ch * t + f] =
                    b2[ch] + (0..hidden).map(|j| h[j] * w2[j * c + ch]).sum::<f64>();
            }
        }
        let weights: Vec<f64> = (0..c)
            .flat_map(|ch| {
                let row = &scores[ch * t..(ch + 1) * t];
                let max = row.iter().copied().fold(f64::MIN, f64::max);
                let e: Vec<f64> = row.iter().map(|s| (s - max).exp()).collect();
                let z: f64 = e.iter().sum();
                e.into_iter().map(move |v| v / z)
            })
            .collect();
        let (mean, std) = stats(&|ch, f| weights[ch * t + f]);
        mean.into_iter().chain(std).collect()
    }

    fn check(
        config: AttentiveStatsPoolingConfig,
        act: impl Fn(f64) -> f64,
    ) {
        let device = performance_device();
        let (c, t) = (config.channels, 9);
        let mut pool: AttentiveStatsPooling = config.init(&device);
        let normal = Distribution::Normal(0.0, 0.5);
        let mlp = &mut pool.attention;
        let (input, hidden) = (config.attention_input(), config.bottleneck);
        mlp.linear1.weight = mlp
            .linear1
            .weight
            .clone()
            .map(|_| seeded_tensor(1, [input, hidden], normal, &device));
        mlp.linear2.weight = mlp
            .linear2
            .weight
            .clone()
            .map(|_| seeded_tensor(2, [hidden, c], normal, &device));
        mlp.linear1.bias = mlp
            .linear1
            .bias
            .clone()
            .map(|b| b.map(|_| seeded_tensor(3, [hidden], normal, &device)));
        mlp.linear2.bias = mlp
            .linear2
            .bias
            .clone()
            .map(|b| b.map(|_| seeded_tensor(4, [c], normal, &device)));
        let x = seeded_tensor(5, [2, c, t], Distribution::Normal(0.3, 1.0), &device);
        let actual = pool.forward(x.clone());
        let weights = (
            tensor_to_vec_as::<f64, _>(&pool.attention.linear1.weight.val()),
            tensor_to_vec_as::<f64, _>(&pool.attention.linear1.bias.as_ref().unwrap().val()),
            tensor_to_vec_as::<f64, _>(&pool.attention.linear2.weight.val()),
            tensor_to_vec_as::<f64, _>(&pool.attention.linear2.bias.as_ref().unwrap().val()),
        );
        let xs = tensor_to_vec_as::<f64, _>(&x);
        let expected: Vec<f64> = (0..2)
            .flat_map(|b| {
                reference(
                    &xs[b * c * t..(b + 1) * c * t],
                    (c, t),
                    (&weights.0, &weights.1, &weights.2, &weights.3),
                    hidden,
                    &act,
                    config.global_context,
                    config.deviation,
                )
            })
            .collect();
        assert_tensor_close_to_vec(&actual, &expected, REORDER);
    }

    /// ASTP's form: tanh, no context, the raw moment.
    #[test]
    fn test_astp() {
        check(AttentiveStatsPoolingConfig::new(4, 3), f64::tanh);
    }

    /// The other form: `ReLU` then tanh, the global context, the centered
    /// deviation.
    #[test]
    fn test_ecapa() {
        check(
            AttentiveStatsPoolingConfig::new(4, 3)
                .with_act(BunsenActivationConfig::Chain(vec![
                    BunsenActivationConfig::Relu,
                    BunsenActivationConfig::Tanh,
                ]))
                .with_global_context(true)
                .with_deviation(AttentiveStatsDeviation::Centered { floor: 1e-12 }),
            |v| v.max(0.0).tanh(),
        );
    }

    /// The floor holds where the variance is zero: a constant sequence.
    #[test]
    fn test_floor() {
        let device = performance_device();
        let x = Tensor::<3>::ones([1, 2, 5], &device);
        let w = Tensor::<3>::ones([1, 1, 5], &device) / 5.0;
        for (deviation, floor) in [
            (AttentiveStatsDeviation::RawMoment { floor: 1e-4 }, 1e-4f64),
            (AttentiveStatsDeviation::Centered { floor: 1e-6 }, 1e-6),
        ] {
            let (mean, std) = deviation.weighted_mean_std(x.clone(), w.clone());
            assert_tensor_close_to_vec(&mean, &[1.0, 1.0], ROUNDING);
            assert_tensor_close_to_vec(&std, &[floor.sqrt(); 2], Tolerance::rel_abs(1e-3, 1e-6));
        }
    }

    /// A config and the pooling it builds answer the meta alike.
    #[test]
    fn test_meta() {
        let config = AttentiveStatsPoolingConfig::new(6, 5).with_global_context(true);
        let pool: AttentiveStatsPooling = config.init(&performance_device());
        for meta in [&config as &dyn AttentiveStatsPoolingMeta, &pool] {
            assert_eq!(
                (
                    meta.channels(),
                    meta.bottleneck(),
                    meta.global_context(),
                    meta.attention_input(),
                    meta.d_output()
                ),
                (6, 5, true, 18, 12)
            );
        }
        assert!(!AttentiveStatsPoolingConfig::new(6, 5).global_context);
    }
}
