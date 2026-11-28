//! # Projected decoder stacks
//!
//! [`ProjectedDecoderStack`]: a decoder stack between a projection in and a
//! projection out, run over a whole sequence with positions from 0 and no
//! cache: a codec's transformer trunk, at a width of its own between the
//! latent's. ZONOS3's codec stages and Qwen3-TTS's codec transformer.

use burn::{
    Tensor,
    config::Config,
    nn::{
        Linear,
        LinearConfig,
    },
    prelude::Device,
};

use crate::{
    blocks::transformers::decoders::stack::{
        DecoderStack,
        DecoderStackMeta,
        DecoderStackStructureConfig,
    },
    burner::module::{
        CanonicalKeys,
        ModuleInit,
        Rekey,
    },
    errors::BunsenResult,
    ops::transformers::attention::KVCache,
};

/// Common meta for [`ProjectedDecoderStack`] and its configs.
pub trait ProjectedDecoderStackMeta {
    /// The width it reads, `C_in`.
    fn input_width(&self) -> usize;

    /// The stack's width, `W`.
    fn d_model(&self) -> usize;

    /// The stack's layers.
    fn n_layers(&self) -> usize;

    /// The width it writes, `C_out`: the output projection's, else the
    /// stack's.
    fn output_width(&self) -> usize;
}

/// Config for [`ProjectedDecoderStack`].
#[derive(Config, Debug)]
pub struct ProjectedDecoderStackConfig {
    /// The width it reads, `C_in`.
    pub input_width: usize,

    /// The stack; its layers' width is the stack's
    /// [`d_model`](ProjectedDecoderStackMeta::d_model).
    pub stack: DecoderStackStructureConfig,

    /// The width it writes; `None`, the stack's (no output projection).
    #[config(default = "None")]
    pub output_width: Option<usize>,

    /// Whether the projections have a bias.
    #[config(default = false)]
    pub bias: bool,
}

impl ProjectedDecoderStackMeta for ProjectedDecoderStackConfig {
    fn input_width(&self) -> usize {
        self.input_width
    }

    fn d_model(&self) -> usize {
        self.stack.d_model()
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers()
    }

    fn output_width(&self) -> usize {
        self.output_width.unwrap_or_else(|| self.d_model())
    }
}

impl ModuleInit<ProjectedDecoderStack> for ProjectedDecoderStackConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<ProjectedDecoderStack> {
        let linear = |i, o| LinearConfig::new(i, o).with_bias(self.bias).init(device);
        let w = self.d_model();
        Ok(ProjectedDecoderStack {
            input_proj: linear(self.input_width, w),
            stack: self.stack.try_init(device)?,
            output_proj: self.output_width.map(|o| linear(w, o)),
        })
    }
}

/// A decoder stack between a projection in and an optional projection out.
#[derive(burn::module::Module, Debug)]
pub struct ProjectedDecoderStack {
    /// `[C_in, W]`.
    pub input_proj: Linear,

    /// The layers, and the stack's final norm.
    pub stack: DecoderStack,

    /// `[W, C_out]`; `None`, it writes the stack's width.
    pub output_proj: Option<Linear>,
}

impl ProjectedDecoderStackMeta for ProjectedDecoderStack {
    fn input_width(&self) -> usize {
        self.input_proj.weight.dims()[0]
    }

    fn d_model(&self) -> usize {
        self.input_proj.weight.dims()[1]
    }

    fn n_layers(&self) -> usize {
        self.stack.n_layers()
    }

    fn output_width(&self) -> usize {
        self.output_proj
            .as_ref()
            .map_or_else(|| self.d_model(), |proj| proj.weight.dims()[1])
    }
}

impl ProjectedDecoderStack {
    /// `[B, T, C_in]` to `[B, T, C_out]`: the whole sequence, positions from
    /// 0, no cache.
    pub fn forward(
        &self,
        x: Tensor<3>,
    ) -> Tensor<3> {
        let [b, t] = self.unpack_input(&x);
        let x = self.input_proj.forward(x);
        let x = self
            .stack
            .forward_hidden(x.clone(), &self.stack.rotary(t, &x.device()), None);
        self.project_out(x, b, t)
    }

    /// The batch and length of `x`, checked against the input width:
    /// `[B, T, C_in]`.
    fn unpack_input(
        &self,
        x: &Tensor<3>,
    ) -> [usize; 2] {
        crate::contracts::unpack_shape_contract!(
            ["B", "T", "C_in"],
            &x.dims(),
            &["B", "T"],
            &[("C_in", self.input_width())]
        )
    }

    /// The stack's output through the output projection, if any: `[B, T,
    /// C_out]`.
    fn project_out(
        &self,
        x: Tensor<3>,
        b: usize,
        t: usize,
    ) -> Tensor<3> {
        let y = match &self.output_proj {
            Some(proj) => proj.forward(x),
            None => x,
        };
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["B", "T", "C_out"],
            &y.dims(),
            &[("B", b), ("T", t), ("C_out", self.output_width())]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (b, t);
        y
    }

    /// `[B, T, C_in]` to `[B, T, C_out]` at positions `start..start + T`,
    /// `start` the cache's [`pos`](KVCache::pos), through the cache: a
    /// stream's next chunk. Its rotary tables cover those positions only
    /// ([`DecoderStack::rotary_range`]), so a stream runs any length; a
    /// windowed stack's cache (from the stack's
    /// [`kv_cache_config`](crate::ops::transformers::attention::KVCacheModel::kv_cache_config)) keeps each
    /// layer's window only.
    pub fn forward_cached(
        &self,
        x: Tensor<3>,
        cache: &mut KVCache,
    ) -> Tensor<3> {
        let [b, t] = self.unpack_input(&x);
        let start = cache.pos();
        let x = self.input_proj.forward(x);
        let r_emb = self.stack.rotary_range(start..start + t, &x.device());
        let x = self.stack.forward_hidden(x, &r_emb, Some(cache));
        self.project_out(x, b, t)
    }
}

impl CanonicalKeys for ProjectedDecoderStack {
    fn canonical_rekey(&self) -> Rekey {
        Rekey::keep_rest().mount_canonical("stack.", &self.stack)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::{
        blocks::{
            mlp::BunsenMlpConfig,
            norm::BunsenNormalizationConfig,
            transformers::{
                attention::csa::CausalSelfAttentionConfig,
                decoders::causal::CausalDecoderBlockConfig,
                embedding::RotaryEmbeddingConfig,
            },
        },
        support::testing::{
            ROUNDING,
            assert_tensors_close,
            performance_device,
            seeded_tensor,
        },
    };

    fn stack(width: usize) -> DecoderStackStructureConfig {
        let layer = CausalDecoderBlockConfig::new(
            CausalSelfAttentionConfig::new(2, 2, width).into(),
            BunsenMlpConfig::expansion(width, 4).into(),
        );
        DecoderStackStructureConfig::new(
            vec![layer; 2],
            BunsenNormalizationConfig::Identity,
            RotaryEmbeddingConfig::new(0, width / 2),
        )
    }

    /// The geometry, with and without an output projection; the forward is
    /// the projections around the stack's, positions from 0.
    #[test]
    fn test_projected_stack() {
        let device = performance_device();
        let config = ProjectedDecoderStackConfig::new(6, stack(8))
            .with_output_width(Some(5))
            .with_bias(true);
        let model = config.try_init(&device).unwrap();
        let meta = |m: &dyn ProjectedDecoderStackMeta| {
            [m.input_width(), m.d_model(), m.n_layers(), m.output_width()]
        };
        assert_eq!(meta(&config), [6, 8, 2, 5]);
        assert_eq!(meta(&model), meta(&config));
        assert!(model.input_proj.bias.is_some());

        let x = seeded_tensor(3, [2, 7, 6], Distribution::Normal(0.0, 1.0), &device);
        let y = model.forward(x.clone());
        assert_eq!(y.dims(), [2, 7, 5]);
        let h = model.input_proj.forward(x);
        let h = model
            .stack
            .forward_hidden(h.clone(), &model.stack.rotary(7, &h.device()), None);
        let expected = model.output_proj.as_ref().unwrap().forward(h);
        assert_tensors_close(&y, &expected, ROUNDING);

        let bare = ProjectedDecoderStackConfig::new(6, stack(8))
            .try_init(&device)
            .unwrap();
        assert!(bare.output_proj.is_none() && bare.input_proj.bias.is_none());
        assert_eq!(meta(&bare), [6, 8, 2, 8]);
        assert_eq!(
            bare.forward(seeded_tensor(
                4,
                [1, 3, 6],
                Distribution::Normal(0.0, 1.0),
                &device
            ))
            .dims(),
            [1, 3, 8]
        );
    }
}
