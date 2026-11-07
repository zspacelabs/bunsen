//! The gated `DeltaNet`, a linear-attention token mixer.

use alloc::string::ToString;

use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        Initializer,
        Linear,
        LinearConfig,
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    prelude::s,
    tensor::{
        Device,
        FloatDType,
        activation::{
            sigmoid,
            silu,
            softplus,
        },
    },
};
// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::{
    blocks::norm::{
        GatedNorm,
        GatedNormConfig,
    },
    burner::{
        module::{
            InputDTypes,
            ModuleInit,
            ResultDType,
        },
        tensor::Widening,
    },
    contracts::unpack_shape_contract,
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
        WithOkOrPanic,
    },
    ops::{
        repeat::repeat_interleave,
        transformers::linear_attention::{
            DeltaNetState,
            DeltaNetStateConfig,
            GatedDeltaRule,
            causal_depthwise_conv1d,
        },
    },
};

/// Common meta for [`GatedDeltaNet`] and [`GatedDeltaNetConfig`].
pub trait GatedDeltaNetMeta {
    /// The size of the input and output.
    fn n_embed(&self) -> usize;

    /// The query and key heads.
    fn n_key_heads(&self) -> usize;

    /// The value heads, a multiple of the key heads: the delta rule's heads.
    fn n_value_heads(&self) -> usize;

    /// The width of each query and key head.
    fn key_dim(&self) -> usize;

    /// The width of each value head.
    fn value_dim(&self) -> usize;

    /// The convolution's taps.
    fn conv_kernel(&self) -> usize;

    /// The convolution's channels: the queries', keys' and values' widths,
    /// `2 · H_k · d_k + H_v · d_v`.
    fn conv_channels(&self) -> usize {
        2 * self.n_key_heads() * self.key_dim() + self.n_value_heads() * self.value_dim()
    }

    /// A [`DeltaNetState`] config for `rows` decode rows of this mixer.
    fn state_config(
        &self,
        rows: usize,
    ) -> DeltaNetStateConfig {
        DeltaNetStateConfig::new(
            rows,
            self.conv_channels(),
            self.conv_kernel(),
            self.n_value_heads(),
            self.key_dim(),
            self.value_dim(),
        )
    }
}

/// Config for [`GatedDeltaNet`].
#[derive(Config, Debug)]
pub struct GatedDeltaNetConfig {
    /// The size of the input and output.
    pub n_embed: usize,

    /// The query and key heads.
    pub n_key_heads: usize,

    /// The value heads, a multiple of [`n_key_heads`](Self::n_key_heads).
    pub n_value_heads: usize,

    /// The width of each query and key head.
    pub key_dim: usize,

    /// The width of each value head.
    pub value_dim: usize,

    /// The convolution's taps (Qwen3.5's `linear_conv_kernel_dim`).
    #[config(default = 4)]
    pub conv_kernel: usize,

    /// The gated output norm, over each value head; its features are
    /// [`value_dim`](Self::value_dim)'s, whatever it says.
    #[config(default = "GatedNormConfig::rms(0, 1e-6)")]
    pub norm: GatedNormConfig,

    /// The delta rule's options.
    #[config(default = "GatedDeltaRule::default()")]
    pub rule: GatedDeltaRule,
}

impl GatedDeltaNetMeta for GatedDeltaNetConfig {
    fn n_embed(&self) -> usize {
        self.n_embed
    }

    fn n_key_heads(&self) -> usize {
        self.n_key_heads
    }

    fn n_value_heads(&self) -> usize {
        self.n_value_heads
    }

    fn key_dim(&self) -> usize {
        self.key_dim
    }

    fn value_dim(&self) -> usize {
        self.value_dim
    }

    fn conv_kernel(&self) -> usize {
        self.conv_kernel
    }
}

impl GatedDeltaNetConfig {
    /// Initializes the module at `layer_index`, its slot in a
    /// [`DeltaNetCache`](crate::ops::transformers::linear_attention::DeltaNetCache).
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for value heads that are not a multiple of
    /// the key heads, or a convolution of fewer than two taps.
    pub fn try_init(
        &self,
        layer_index: usize,
        device: &Device,
    ) -> BunsenResult<GatedDeltaNet> {
        if self.n_key_heads == 0 || !self.n_value_heads.is_multiple_of(self.n_key_heads) {
            return Err(ConstraintError::new(
                "GatedDeltaNetConfig",
                "n_value_heads",
                Rule::NotMultiple {
                    value: self.n_value_heads,
                    of: format!("n_key_heads ({})", self.n_key_heads),
                },
            )
            .into());
        }
        if self.conv_kernel < 2 {
            return Err(ConstraintError::new(
                "GatedDeltaNetConfig",
                "conv_kernel",
                Rule::OutOfRange {
                    value: self.conv_kernel.to_string(),
                    range: ">= 2".into(),
                },
            )
            .into());
        }
        let linear = |d_out| {
            LinearConfig::new(self.n_embed, d_out)
                .with_bias(false)
                .init(device)
        };
        let channels = self.conv_channels();
        let heads = self.n_value_heads;
        Ok(GatedDeltaNet {
            layer_index,
            n_key_heads: self.n_key_heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            rule: self.rule,
            in_proj_qkv: linear(channels),
            in_proj_z: linear(heads * self.value_dim),
            in_proj_b: linear(heads),
            in_proj_a: linear(heads),
            conv1d: Conv1dConfig::new(channels, channels, self.conv_kernel)
                .with_groups(channels)
                .with_bias(false)
                .init(device),
            // As upstream's init: a decay rate `A` in [1, 16], a time-step
            // bias of one.
            a_log: Initializer::Uniform {
                min: 0.0,
                max: 16f64.ln(),
            }
            .init([heads], device),
            dt_bias: Initializer::Ones.init([heads], device),
            norm: self
                .norm
                .clone()
                .with_num_features(self.value_dim)
                .try_init(device)?,
            out_proj: LinearConfig::new(heads * self.value_dim, self.n_embed)
                .with_bias(false)
                .init(device),
        })
    }

    /// Initializes the module, or panics.
    pub fn init(
        &self,
        layer_index: usize,
        device: &Device,
    ) -> GatedDeltaNet {
        self.try_init(layer_index, device).ok_or_panic()
    }
}

/// The gated `DeltaNet` (Yang et al., *Gated Delta Networks*), the
/// linear-attention token mixer of Qwen3.5's hybrid layers
/// (`transformers`' `Qwen3_5GatedDeltaNet`).
///
/// ```text
/// q, k, v = split(silu(conv(in_proj_qkv(x))))     a causal depthwise conv
/// β = σ(in_proj_b(x))
/// g = -exp(a_log) · softplus(in_proj_a(x) + dt_bias)
/// o = delta_rule(q, k, v, g, β)                   q, k repeated to the value heads
/// y = out_proj(norm(o, in_proj_z(x)))             the gated norm, per value head
/// ```
///
/// Its fields keep the checkpoints' names, but `A_log`'s, which is
/// [`a_log`](Self::a_log).
///
/// It carries a [`DeltaNetState`] between calls, one per decode, which the
/// caller injects: a prefill without one starts from zeros and keeps
/// nothing. A call of one position steps the rule
/// ([`GatedDeltaRule::recurrent`]), a longer one runs it a chunk at a time
/// ([`GatedDeltaRule::chunked`]); the two agree.
///
/// Built by [`GatedDeltaNetConfig`].
#[derive(Module, Debug)]
pub struct GatedDeltaNet {
    /// Its slot in a
    /// [`DeltaNetCache`](crate::ops::transformers::linear_attention::DeltaNetCache).
    pub layer_index: usize,

    /// The query and key heads.
    pub n_key_heads: usize,

    /// The width of each query and key head.
    pub key_dim: usize,

    /// The width of each value head.
    pub value_dim: usize,

    /// The delta rule's options.
    #[module(skip)]
    pub rule: GatedDeltaRule,

    /// The queries', keys' and values' projection, `[D, 2 H_k d_k + H_v d_v]`.
    pub in_proj_qkv: Linear,

    /// The output gate's projection, `[D, H_v d_v]`.
    pub in_proj_z: Linear,

    /// The write strength's projection, `[D, H_v]`.
    pub in_proj_b: Linear,

    /// The decay's projection, `[D, H_v]`.
    pub in_proj_a: Linear,

    /// The causal depthwise convolution over the projected queries, keys
    /// and values: its weight, `[C, 1, K]`, alone is used.
    pub conv1d: Conv1d,

    /// The log of each value head's decay rate, `[H_v]` (`A_log`).
    pub a_log: Param<Tensor<1>>,

    /// The time-step bias, `[H_v]`.
    pub dt_bias: Param<Tensor<1>>,

    /// The gated output norm, over each value head.
    pub norm: GatedNorm,

    /// The output projection, `[H_v d_v, D]`.
    pub out_proj: Linear,
}

impl GatedDeltaNetMeta for GatedDeltaNet {
    fn n_embed(&self) -> usize {
        self.in_proj_qkv.weight.dims()[0]
    }

    fn n_key_heads(&self) -> usize {
        self.n_key_heads
    }

    fn n_value_heads(&self) -> usize {
        self.in_proj_b.weight.dims()[1]
    }

    fn key_dim(&self) -> usize {
        self.key_dim
    }

    fn value_dim(&self) -> usize {
        self.value_dim
    }

    fn conv_kernel(&self) -> usize {
        self.conv1d.weight.dims()[2]
    }
}

impl InputDTypes for GatedDeltaNet {
    /// Its projections' dtype.
    fn preferred_dtype(&self) -> FloatDType {
        FloatDType::from(self.in_proj_qkv.weight.val().dtype())
    }
}

impl ResultDType for GatedDeltaNet {
    /// Its own: the output projection's.
    fn result_dtype(
        &self,
        _input: FloatDType,
    ) -> FloatDType {
        self.preferred_dtype()
    }
}

impl GatedDeltaNet {
    /// Forward pass, from `state` and advancing it.
    ///
    /// # Arguments
    /// - `x`: a `[B, T, D]` sequence.
    /// - `state`: the decode's state for this layer, at `x`'s first position;
    ///   `None`, a sequence starting at `x`, nothing kept.
    ///
    /// # Returns
    /// - `[B, T, D]`.
    ///
    /// # Panics
    /// If the state's rows are not `B`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        state: Option<&mut DeltaNetState>,
    ) -> Tensor<3> {
        let [b, t] = unpack_shape_contract!(
            ["B", "T", "D"],
            &x.dims(),
            &["B", "T"],
            &[("D", self.n_embed())]
        );
        if let Some(state) = state.as_deref() {
            assert_eq!(state.rows(), b, "a linear-attention state of {b} rows");
        }
        let (hk, hv) = (self.n_key_heads(), self.n_value_heads());
        let (dk, dv) = (self.key_dim, self.value_dim);

        let [channels, _, taps] = self.conv1d.weight.dims();
        let (mixed, next_conv) = causal_depthwise_conv1d(
            self.in_proj_qkv.forward(x.clone()),
            self.conv1d.weight.val().reshape([channels, taps]),
            state.as_deref().map(|s| s.conv.clone()),
        );
        let mixed = silu(mixed);
        let width = hk * dk;
        let repeat = |x: Tensor<4>| match hv / hk {
            1 => x,
            n => repeat_interleave::<4, 5, _>(x, n, 2),
        };
        let q = repeat(
            mixed
                .clone()
                .slice(s![.., .., ..width])
                .reshape([b, t, hk, dk]),
        );
        let k = repeat(
            mixed
                .clone()
                .slice(s![.., .., width..2 * width])
                .reshape([b, t, hk, dk]),
        );
        let v = mixed.slice(s![.., .., 2 * width..]).reshape([b, t, hv, dv]);

        let beta = sigmoid(self.in_proj_b.forward(x.clone()));
        let a = self.in_proj_a.forward(x.clone());
        let w = Widening::F32.scope(a.dtype());
        let g = w.widen(self.a_log.val()).exp().neg().unsqueeze::<3>()
            * softplus(w.widen(a) + w.widen(self.dt_bias.val()).unsqueeze(), 1.0);

        let initial = state.as_deref().map(|s| s.recurrent.clone());
        let (core, recurrent) = if t == 1 {
            self.rule.recurrent(q, k, v, g, beta, initial)
        } else {
            self.rule.chunked(q, k, v, g, beta, initial)
        };
        if let Some(state) = state {
            state.conv = next_conv.cast(FloatDType::F32);
            state.recurrent = recurrent;
        }

        let z = self.in_proj_z.forward(x).reshape([b, t, hv, dv]);
        let y = self.norm.forward(core, z).reshape([b, t, hv * dv]);
        self.out_proj.forward(y)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Distribution,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::{
        assert_tensors_close,
        cpu_device,
        seeded_tensor,
    };

    fn config() -> GatedDeltaNetConfig {
        GatedDeltaNetConfig::new(16, 2, 4, 3, 5).with_conv_kernel(3)
    }

    /// The config's geometry is the module's, and its state's.
    #[test]
    fn test_meta() {
        let device = cpu_device();
        let cfg = config();
        let net = cfg.init(2, &device);
        assert_eq!(net.layer_index, 2);
        for (a, b) in [
            (net.n_embed(), cfg.n_embed()),
            (net.n_key_heads(), cfg.n_key_heads()),
            (net.n_value_heads(), cfg.n_value_heads()),
            (net.key_dim(), cfg.key_dim()),
            (net.value_dim(), cfg.value_dim()),
            (net.conv_kernel(), cfg.conv_kernel()),
        ] {
            assert_eq!(a, b);
        }
        assert_eq!(net.conv_channels(), 2 * 2 * 3 + 4 * 5);
        assert_eq!(net.in_proj_qkv.weight.dims(), [16, 32]);
        assert_eq!(net.conv1d.weight.dims(), [32, 1, 3]);
        let state = net.state_config(3).init(&device);
        assert_eq!(state.conv.dims(), [3, 2, 32]);
        assert_eq!(state.recurrent.dims(), [3, 4, 3, 5]);

        assert!(
            GatedDeltaNetConfig::new(16, 3, 4, 3, 5)
                .try_init(0, &device)
                .is_err(),
            "value heads a multiple of the key heads"
        );
    }

    /// A prefill in parts through a state, a part of one position (a decode
    /// step) included, is the one-pass forward; and the one pass with a
    /// state left it where the parts left it.
    #[test]
    fn test_parts_agree_with_one_pass() {
        let device = cpu_device();
        let net = config().init(0, &device);
        let x = seeded_tensor::<3>(5, [2, 9, 16], Distribution::Default, &device);
        let whole = net.forward(x.clone(), None);
        assert_eq!(whole.dims(), [2, 9, 16]);

        let mut one_pass = net.state_config(2).init(&device);
        let again = net.forward(x.clone(), Some(&mut one_pass));
        assert_tensors_close(&again, &whole, Tolerance::default());

        let mut state = net.state_config(2).init(&device);
        let parts: Vec<_> = [0, 4, 5, 9]
            .windows(2)
            .map(|span| {
                net.forward(
                    x.clone().slice(s![.., span[0]..span[1], ..]),
                    Some(&mut state),
                )
            })
            .collect();
        assert_tensors_close(&Tensor::cat(parts, 1), &whole, Tolerance::default());
        assert_tensors_close(&state.conv, &one_pass.conv, Tolerance::default());
        assert_tensors_close(&state.recurrent, &one_pass.recurrent, Tolerance::default());
    }
}
