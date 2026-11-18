//! # The attention-residual stack
//!
//! [`AttnResStack`]: a [`DecoderStack`]'s layers without a residual stream.
//! Full Attention Residuals (ZONOS3's `full_attnres`) route each sublayer's
//! input from a bank of the stack's input and every earlier sublayer's
//! output ([`attn_res_route`]), a learned query per route; the sublayers'
//! outputs are their branches' updates alone, never added to anything.
//!
//! Everything else is the decoder stack's: the layers and their configs
//! (a [`PatternedStackConfig`] mixes attention and linear-attention layers),
//! the input and final norms, the rotary tables and the decode's caches. Only
//! the topology between the layers differs.

use alloc::vec::Vec;

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
use crate::blocks::transformers::decoders::stack::PatternedStackConfig;
use crate::{
    blocks::transformers::{
        attention::csa::AttentionSides,
        decoders::stack::{
            DecoderStack,
            DecoderStackCache,
            DecoderStackMeta,
            DecoderStackVariantConfig,
        },
        embedding::RotarySet,
    },
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    errors::BunsenResult,
    ops::transformers::{
        attention::{
            KVCache,
            KVCacheStorage,
        },
        attn_res::attn_res_route,
        linear_attention::DeltaNetCache,
    },
};

/// Config for [`AttnResStack`].
#[derive(Config, Debug)]
pub struct AttnResStackConfig {
    /// The layers, the input and final norms and the rotary embedding, as a
    /// [`DecoderStack`] would have them.
    pub stack: DecoderStackVariantConfig,

    /// The epsilon of the routes' weightless RMS norm on their keys.
    #[config(default = 1e-5)]
    pub key_eps: f64,
}

impl ModuleInit<AttnResStack> for AttnResStackConfig {
    /// The decoder stack, and a zero query per route (an even mixture):
    /// `2 L + 1` of them, `L` the layer count.
    ///
    /// # Errors
    /// As the stack's own.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<AttnResStack> {
        let stack: DecoderStack = self.stack.try_to_structure()?.try_init(device)?;
        let routes = 2 * stack.n_layers() + 1;
        Ok(AttnResStack {
            queries: Param::from_tensor(Tensor::zeros([routes, stack.d_model()], device)),
            stack,
            key_eps: self.key_eps,
        })
    }
}

/// A stack of decoder layers whose sublayers read routed mixtures of the
/// stack's input and every earlier sublayer's output, in place of a
/// residual stream (Full Attention Residuals).
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
///
/// Built by [`AttnResStackConfig`].
#[derive(Module, Debug)]
pub struct AttnResStack {
    /// The layers, norms and rotary embedding; its own residual forward is
    /// not used.
    pub stack: DecoderStack,

    /// The routes' queries, `[2 L + 1, D]`: layer `l`'s attention route at
    /// `2 l`, its feed-forward route at `2 l + 1`, the final route last.
    pub queries: Param<Tensor<2>>,

    /// The epsilon of the routes' key norm.
    #[module(skip)]
    pub key_eps: f64,
}

impl AttnResStack {
    /// The bank's size after the last layer: the input and two outputs per
    /// layer, `2 L + 1`.
    pub fn n_sources(&self) -> usize {
        2 * self.stack.n_layers() + 1
    }

    /// A decode cache, as the decoder stack's
    /// ([`DecoderStack::decode_cache`]): the attention layers' KV cache and
    /// the linear-attention layers' states. The bank needs none: every
    /// source is per position.
    pub fn decode_cache(
        &self,
        rows: usize,
        positions: usize,
        storage: KVCacheStorage,
        device: &Device,
    ) -> DecoderStackCache {
        self.stack.decode_cache(rows, positions, storage, device)
    }

    /// [`forward_with_caches`](Self::forward_with_caches) through `cache`,
    /// at its position, advancing it.
    pub fn forward_cached(
        &self,
        x: Tensor<3>,
        cache: &mut DecoderStackCache,
    ) -> Tensor<3> {
        self.forward_with_caches(
            x,
            &cache.rotary,
            Some(&mut cache.cache),
            Some(&mut cache.linear),
        )
    }

    /// Runs the layers and the final norm over `x`, at positions
    /// `start..start + seq`, through the caches when there are some.
    ///
    /// # Arguments
    /// * `x` - `[batch, seq, d_model]`.
    /// * `r_emb` - rotary tables covering `start + seq` positions
    ///   ([`DecoderStack::rotary`]).
    /// * `kv_cache`, `linear` - the decode's caches, at position `start`;
    ///   `None` means `start` is 0 and nothing is kept.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, normalized.
    pub fn forward_with_caches(
        &self,
        x: Tensor<3>,
        r_emb: &RotarySet,
        kv_cache: Option<&mut KVCache>,
        linear: Option<&mut DeltaNetCache>,
    ) -> Tensor<3> {
        self.forward_bank(x, r_emb, kv_cache, linear).0
    }

    /// [`forward_with_caches`](Self::forward_with_caches), with the bank it
    /// built: the input (after the input norm), then each layer's attention
    /// and feed-forward outputs, `2 L + 1` tensors of `[batch, seq,
    /// d_model]`.
    pub fn forward_bank(
        &self,
        x: Tensor<3>,
        r_emb: &RotarySet,
        mut kv_cache: Option<&mut KVCache>,
        mut linear: Option<&mut DeltaNetCache>,
    ) -> (Tensor<3>, Vec<Tensor<3>>) {
        let d_model = self.stack.d_model();
        let [batch, seq] = crate::contracts::unpack_shape_contract!(
            ["batch", "seq", "d_model"],
            &x.dims(),
            &["batch", "seq"],
            &[("d_model", d_model)]
        );
        let start = kv_cache.as_ref().map(|c| c.pos()).unwrap_or(0);
        let r_emb = r_emb.clip_range(start..start + seq);
        let eps = self.key_eps as f32;
        let queries = self.queries.val();
        let query = |i: usize| queries.clone().slice(s![i..i + 1]).reshape([d_model]);

        let x = match &self.stack.input_norm {
            Some(norm) => norm.forward(x),
            None => x,
        };
        let mut bank = Vec::with_capacity(self.n_sources());
        bank.push(x);
        for (i, layer) in self.stack.layers.iter().enumerate() {
            let h = attn_res_route(&bank, query(2 * i), eps);
            let a = layer.attn_residual(
                h,
                r_emb.for_layer(i),
                &mut kv_cache,
                linear.as_deref_mut(),
                AttentionSides::default(),
            );
            bank.push(a);
            let h = attn_res_route(&bank, query(2 * i + 1), eps);
            bank.push(layer.mlp_residual(h, None));
        }
        let y =
            self.stack
                .norm
                .forward(attn_res_route(&bank, query(2 * self.stack.n_layers()), eps));

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "seq", "d_model"],
            &y.dims(),
            &[("batch", batch), ("seq", seq), ("d_model", d_model)],
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = batch;

        (y, bank)
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
            norm::BunsenNormalizationConfig,
            transformers::{
                attention::csa::CausalSelfAttentionConfig,
                decoders::{
                    causal::CausalDecoderBlockConfig,
                    stack::PatternedStackConfig,
                },
                embedding::RotaryEmbeddingConfig,
                mixers::GatedDeltaNetConfig,
                mlp::TransformerGatedMlpConfig,
            },
        },
        support::testing::{
            DeviceMemoryGuard,
            assert_tensors_close,
            cpu_device,
            performance_device,
            seeded_tensor,
        },
    };

    /// Linear, attention, linear: the shape of ZONOS3's pattern.
    fn config() -> AttnResStackConfig {
        let attention = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(4, 2, 16)
                .with_head_dim(Some(8))
                .into(),
            TransformerGatedMlpConfig::new(16, 32).into(),
        );
        let linear = CausalDecoderBlockConfig::new(
            GatedDeltaNetConfig::new(16, 2, 4, 4, 3).into(),
            TransformerGatedMlpConfig::new(16, 32).into(),
        );
        AttnResStackConfig::new(
            PatternedStackConfig::new(
                vec![linear, attention],
                vec![0, 1, 0],
                BunsenNormalizationConfig::Rms(RmsNormConfig::new(0)),
                RotaryEmbeddingConfig::new(0, 8),
            )
            .into(),
        )
    }

    /// A stack with random route queries, so no route is an even mixture.
    fn stack(device: &Device) -> AttnResStack {
        let mut stack: AttnResStack = config().init(device);
        assert_eq!(stack.queries.val().dims(), [7, 16]);
        stack.queries = Param::from_tensor(seeded_tensor(
            3,
            [7, 16],
            Distribution::Normal(0.0, 1.0),
            device,
        ));
        stack
    }

    /// The bank holds the input and each sublayer's update, and each update
    /// is its sublayer's branch on the route the bank so far gives it: the
    /// stack adds nothing to any of them.
    #[test]
    fn test_bank_is_routed_updates() {
        let device = cpu_device();
        let stack = stack(&device);
        assert_eq!(stack.n_sources(), 7);
        let x: Tensor<3> = seeded_tensor(1, [2, 5, 16], Distribution::Default, &device);
        let r_emb = stack.stack.rotary(5, &device);
        let (y, bank) = stack.forward_bank(x.clone(), &r_emb, None, None);
        assert_eq!(bank.len(), 7);
        assert_tensors_close(&bank[0], &x, Tolerance::default());

        let q = |i: usize| stack.queries.val().slice(s![i..i + 1]).reshape([16]);
        for (l, layer) in stack.stack.layers.iter().enumerate() {
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
        let expected = stack.stack.norm.forward(attn_res_route(&bank, q(6), 1e-5));
        assert_tensors_close(&y, &expected, Tolerance::default());
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
        let whole =
            stack.forward_with_caches(x.clone(), &stack.stack.rotary(n, &device), None, None);

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
}
