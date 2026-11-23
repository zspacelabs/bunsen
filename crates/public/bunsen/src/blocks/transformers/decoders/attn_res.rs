//! # Attention residuals
//!
//! [`AttnResRoutes`]: a [`DecoderStack`]'s learned routes when it has no
//! residual stream. Full Attention Residuals (ZONOS3's `full_attnres`) route
//! each sublayer's input from a bank of the stack's input and every earlier
//! sublayer's output ([`attn_res_route`]), a learned query per route; the
//! sublayers' outputs are their branches' updates alone, never added to
//! anything. A stack with routes
//! ([`DecoderStackStructureConfig::routes`](crate::blocks::transformers::decoders::stack::DecoderStackStructureConfig::routes))
//! runs that topology ([`DecoderStack::forward_bank`]); everything else is
//! the stack's as ever: the layers and their configs, the input and final
//! norms, the rotary tables and the decode's caches (the bank needs none:
//! every source is per position).

use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    prelude::s,
    tensor::Device,
};

#[cfg(doc)]
use crate::blocks::transformers::decoders::stack::DecoderStack;
use crate::ops::transformers::attn_res::attn_res_route;

/// Config for [`AttnResRoutes`].
#[derive(Config, Debug)]
pub struct AttnResRoutesConfig {
    /// The epsilon of the routes' weightless RMS norm on their keys.
    #[config(default = 1e-5)]
    pub key_eps: f64,
}

impl AttnResRoutesConfig {
    /// The routes of a stack of `n_layers` layers of width `d_model`: a zero
    /// query per route (an even mixture), `2 n_layers + 1` of them.
    pub fn init(
        &self,
        n_layers: usize,
        d_model: usize,
        device: &Device,
    ) -> AttnResRoutes {
        AttnResRoutes {
            queries: Param::from_tensor(Tensor::zeros([2 * n_layers + 1, d_model], device)),
            key_eps: self.key_eps,
        }
    }
}

/// A [`DecoderStack`]'s routes, in place of its residual stream (Full
/// Attention Residuals).
///
/// For layer `l`, with `bank` starting as `[x]` (the stack's input after its
/// input norm):
///
/// ```text
/// a = attn_update_l(route(bank, q[2l]));     bank += [a]
/// f = mlp_update_l(route(bank, q[2l + 1]));  bank += [f]
/// ```
///
/// each update its sublayer's `post(branch(pre(·)))` alone
/// ([`attn_residual`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock::attn_residual),
/// [`mlp_residual`](crate::blocks::transformers::decoders::causal::CausalDecoderBlock::mlp_residual)),
/// and the output `norm(route(bank, q[2L]))`.
#[derive(Module, Debug)]
pub struct AttnResRoutes {
    /// The routes' queries, `[2 L + 1, D]`: layer `l`'s attention route at
    /// `2 l`, its feed-forward route at `2 l + 1`, the final route last.
    pub queries: Param<Tensor<2>>,

    /// The epsilon of the routes' key norm.
    #[module(skip)]
    pub key_eps: f64,
}

impl AttnResRoutes {
    /// The bank's size after the last layer: the input and two outputs per
    /// layer, `2 L + 1`.
    pub fn n_sources(&self) -> usize {
        self.queries.dims()[0]
    }

    /// Route `i` over `bank` ([`attn_res_route`]).
    pub fn route(
        &self,
        bank: &[Tensor<3>],
        i: usize,
    ) -> Tensor<3> {
        let d_model = self.queries.dims()[1];
        let query = self.queries.val().slice(s![i..i + 1]).reshape([d_model]);
        attn_res_route(bank, query, self.key_eps as f32)
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
    use serial_test::serial;

    use super::*;
    use crate::{
        blocks::{
            mlp::BunsenGatedMlpConfig,
            norm::BunsenNormalizationConfig,
            transformers::{
                attention::csa::{
                    AttentionSides,
                    CausalSelfAttentionConfig,
                },
                decoders::{
                    causal::CausalDecoderBlockConfig,
                    stack::{
                        DecoderStack,
                        DecoderStackStructureConfig,
                        PatternedStackConfig,
                    },
                },
                embedding::RotaryEmbeddingConfig,
                mixers::GatedDeltaNetConfig,
            },
        },
        burner::module::{
            ModuleInit,
            ToStructureConfig,
        },
        ops::transformers::attention::KVCacheStorage,
        support::testing::{
            DeviceMemoryGuard,
            assert_tensors_close,
            cpu_device,
            performance_device,
            seeded_tensor,
        },
    };

    /// Linear, attention, linear: the shape of ZONOS3's pattern, with routes.
    fn config() -> DecoderStackStructureConfig {
        let attention = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, 16)
                .with_head_dim(Some(8))
                .into(),
            BunsenGatedMlpConfig::new(16, 32).into(),
        );
        let linear = CausalDecoderBlockConfig::new(
            GatedDeltaNetConfig::new(16, 2, 4, 4, 3).into(),
            BunsenGatedMlpConfig::new(16, 32).into(),
        );
        PatternedStackConfig::new(
            vec![linear, attention],
            vec![0, 1, 0],
            BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
            RotaryEmbeddingConfig::new(0, 8),
        )
        .to_structure()
        .with_routes(Some(AttnResRoutesConfig::new()))
    }

    /// A stack with random route queries, so no route is an even mixture.
    fn stack(device: &Device) -> DecoderStack {
        let mut stack: DecoderStack = config().init(device);
        let routes = stack.routes.as_mut().unwrap();
        assert_eq!(routes.queries.val().dims(), [7, 16]);
        routes.queries = Param::from_tensor(seeded_tensor(
            3,
            [7, 16],
            Distribution::Normal(0.0, 1.0),
            device,
        ));
        stack
    }

    /// The bank holds the input and each sublayer's update, and each update
    /// is its sublayer's branch on the route the bank so far gives it: the
    /// stack adds nothing to any of them; its forward is the bank's.
    #[test]
    fn test_bank_is_routed_updates() {
        let device = cpu_device();
        let stack = stack(&device);
        let routes = stack.routes.as_ref().unwrap();
        assert_eq!(routes.n_sources(), 7);
        let x: Tensor<3> = seeded_tensor(1, [2, 5, 16], Distribution::Default, &device);
        let r_emb = stack.rotary(5, &device);
        let (y, bank) = stack.forward_bank(x.clone(), &r_emb, None, None);
        assert_eq!(bank.len(), 7);
        assert_tensors_close(&bank[0], &x, Tolerance::default());

        let q = |i: usize| routes.queries.val().slice(s![i..i + 1]).reshape([16]);
        for (l, layer) in stack.layers.iter().enumerate() {
            let a = layer.attn_residual(
                attn_res_route(&bank[..=2 * l], q(2 * l), 1e-5),
                r_emb.for_layer(l),
                &mut None,
                None,
                AttentionSides::default(),
            );
            assert_tensors_close(&bank[2 * l + 1], &a, Tolerance::default());
            let f = layer.mlp_residual(
                attn_res_route(&bank[..=2 * l + 1], q(2 * l + 1), 1e-5),
                None,
            );
            assert_tensors_close(&bank[2 * l + 2], &f, Tolerance::default());
        }
        let expected = stack.norm.forward(attn_res_route(&bank, q(6), 1e-5));
        assert_tensors_close(&y, &expected, Tolerance::default());
        assert_tensors_close(
            &stack.forward_hidden(x, &r_emb, None),
            &y,
            Tolerance::default(),
        );
    }

    /// A cached decode, a prefill and then a position at a time, is the
    /// uncached pass: the routes are per position, and the mixers carry
    /// their own state.
    #[test]
    #[serial]
    fn test_cached_matches_uncached() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let stack = stack(&device);
        let n = 7;
        let x: Tensor<3> = seeded_tensor(12, [2, n, 16], Distribution::Default, &device);
        let whole = stack
            .forward_bank(x.clone(), &stack.rotary(n, &device), None, None)
            .0;

        let mut cache = stack.decode_cache(2, n, KVCacheStorage::Auto, &device);
        let mut parts = vec![stack.forward_cached(x.clone().slice_dim(1, 0..3), &mut cache)];
        for pos in 3..n {
            parts.push(stack.forward_cached(x.clone().slice_dim(1, pos..pos + 1), &mut cache));
        }
        assert_eq!(cache.pos(), n);
        assert_tensors_close(
            &Tensor::cat(parts, 1),
            &whole,
            Tolerance::rel_abs(1e-4, 1e-4),
        );
    }

    /// A stack without routes keeps no bank.
    #[test]
    #[should_panic(expected = "a stack with attention-residual routes")]
    fn test_no_bank_without_routes() {
        let device = cpu_device();
        let stack: DecoderStack = config().with_routes(None).init(&device);
        let x: Tensor<3> = seeded_tensor(1, [1, 3, 16], Distribution::Default, &device);
        let _ = stack.forward_bank(x, &stack.rotary(3, &device), None, None);
    }
}
