//! An encoder-decoder's decoder layer: self-attention, cross-attention over
//! the encoder output, and an MLP.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        LayerNormConfig,
        activation::ActivationConfig,
        attention::{
            MultiHeadAttention,
            MultiHeadAttentionConfig,
        },
        norm::{
            Normalization,
            NormalizationConfig,
        },
    },
    prelude::Bool,
    tensor::Device,
};

use crate::{
    blocks::transformers::{
        encoders::residual::ResidualEncoderAttentionBlock,
        mlp::{
            Mlp,
            MlpConfig,
            norm_mlp,
        },
    },
    burner::module::{
        ModuleInit,
        Rekey,
    },
    errors::BunsenResult,
    ops::transformers::attention::{
        KVCache,
        norm_cross_attn,
        norm_cross_attn_w_kv_cache,
        norm_self_attn,
        norm_self_attn_w_kv_cache,
        project_kv,
    },
};

/// Common meta for [`CrossDecoderBlock`] and
/// [`CrossDecoderBlockConfig`].
pub trait CrossDecoderBlockMeta {
    /// Returns the embedding dimensionality.
    fn d_model(&self) -> usize;

    /// Returns the number of heads.
    fn n_heads(&self) -> usize;

    /// Returns the dropout.
    fn dropout(&self) -> f64;
}

/// Config for [`CrossDecoderBlock`].
#[derive(Config, Debug)]
pub struct CrossDecoderBlockConfig {
    /// Returns the embedding dimensionality.
    pub d_model: usize,

    /// Head width.
    #[config(default = "64")]
    pub d_head: usize,

    /// Dropout.
    #[config(default = "0.0")]
    pub dropout: f64,

    /// The normalization before each sublayer, sized to `d_model` whatever
    /// feature count it names: a `LayerNorm` by default, Whisper's.
    #[config(default = "NormalizationConfig::Layer(LayerNormConfig::new(0))")]
    pub norm: NormalizationConfig,
}

impl CrossDecoderBlockMeta for CrossDecoderBlockConfig {
    fn d_model(&self) -> usize {
        self.d_model
    }

    fn n_heads(&self) -> usize {
        self.d_model / self.d_head
    }

    fn dropout(&self) -> f64 {
        self.dropout
    }
}

impl ModuleInit<CrossDecoderBlock> for CrossDecoderBlockConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<CrossDecoderBlock> {
        // Whisper's attention has no key bias; nor, by default, does this.
        let mha_cfg = MultiHeadAttentionConfig::new(self.d_model, self.n_heads())
            .with_dropout(self.dropout)
            .with_key_bias(false);
        let norm = || {
            self.norm
                .clone()
                .with_num_features(self.d_model)
                .init(device)
        };

        let attn = mha_cfg.init(device);
        let cross_attn = mha_cfg.init(device);

        // Whisper's MLP projections carry a bias, and it runs
        // `Linear -> GELU -> Linear`; the `MlpConfig` default is ReLU.
        let mlp: Mlp = MlpConfig::new(self.d_model)
            .with_activation(ActivationConfig::Gelu)
            .with_bias(true)
            .try_init(device)?;

        Ok(CrossDecoderBlock {
            attn_norm: norm(),
            attn,
            cross_attn_norm: norm(),
            cross_attn,
            mlp_norm: norm(),
            mlp,
        })
    }
}

/// An encoder-decoder's decoder layer: pre-norm masked self-attention,
/// pre-norm cross-attention over the encoder output, and a pre-norm MLP, each
/// wrapped in a residual connection.
///
/// Whisper's text decoder is a stack of these. The attention is burn's
/// [`MultiHeadAttention`] (without a key bias), and the norms are any
/// [`Normalization`], `LayerNorm`s by default: what the ported checkpoints
/// hold. The fields are named by
/// role: `attn_norm`, `attn`, `cross_attn_norm`, `cross_attn`, `mlp_norm`,
/// `mlp`.
///
/// [`forward`](Self::forward) runs a whole sequence, and returns its
/// cross-attention weights too ([`CrossDecoderRecord`]), which alignment needs;
/// [`forward_w_kv_cache`](Self::forward_w_kv_cache) runs a decode's new tokens
/// against its [`KVCache`].
///
/// Built by [`CrossDecoderBlockConfig`].
#[derive(Module, Debug)]
pub struct CrossDecoderBlock {
    /// Attention Normalization.
    pub attn_norm: Normalization,

    /// Attention.
    pub attn: MultiHeadAttention,

    /// Cross Attention Normalization.
    pub cross_attn_norm: Normalization,

    /// Cross Attention.
    pub cross_attn: MultiHeadAttention,

    /// MLP Normalization.
    pub mlp_norm: Normalization,

    /// MLP.
    pub mlp: Mlp,
}

impl CrossDecoderBlockMeta for CrossDecoderBlock {
    fn d_model(&self) -> usize {
        self.attn.d_model
    }

    fn n_heads(&self) -> usize {
        self.attn.n_heads
    }

    fn dropout(&self) -> f64 {
        self.attn.dropout.prob
    }
}

/// What [`CrossDecoderBlock::forward`] returns: its output, and its
/// cross-attention weights.
#[derive(Debug, Clone)]
pub struct CrossDecoderRecord {
    /// Block Output: `[batch, seq_len, d_model]`.
    pub output: Tensor<3>,

    /// Cross-Attention Weights: `[batch, n_heads, seq_len, seq_len]`.
    pub ca_weights: Tensor<4>,
}

impl CrossDecoderRecord {
    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.output.shape()[0]
    }

    /// Returns the embedding size.
    pub fn d_model(&self) -> usize {
        self.output.shape()[2]
    }

    /// Returns the sequence length.
    pub fn seq_len(&self) -> usize {
        self.output.shape()[1]
    }
}

impl CrossDecoderBlock {
    /// Forward pass over a whole sequence.
    ///
    /// # Arguments
    /// * `x` : `[batch, seq_len, d_model]` input.
    /// * `xa` : `[batch, seq_len, d_model]` cross-attention input.
    /// * `mask` : `[batch, seq_len, seq_len]` attention mask.
    ///
    /// # Returns
    /// `CrossDecoderRecord` - forward record.
    /// * `fr.output` : `[batch, seq_len, d_model]`.
    /// * `fr.ca_weights` : `[batch, n_heads, seq_len, seq_len]`.
    pub fn forward(
        &self,
        x: Tensor<3>,
        xa: Tensor<3>,
        mask: Option<Tensor<3, Bool>>,
    ) -> CrossDecoderRecord {
        let self_attn = norm_self_attn(&self.attn_norm, &self.attn, x.clone(), mask);
        let x = x + self_attn.context;

        let cross_attn = norm_cross_attn(&self.cross_attn_norm, &self.cross_attn, x.clone(), xa);
        let x = x + cross_attn.context;

        let mlp = norm_mlp(&self.mlp_norm, &self.mlp, x.clone());
        let x = x + mlp;

        CrossDecoderRecord {
            output: x,
            ca_weights: cross_attn.weights,
        }
    }

    /// Forward pass against a decode's [`KVCache`].
    ///
    /// Unlike [`forward`](Self::forward), `x` is only the **new** token(s):
    /// the cache holds the projected keys and values for everything before,
    /// so this projects and scores only what is new.
    ///
    /// Cross-attention does no work beyond the query: its keys and values
    /// are projected once per decode from the encoder output
    /// ([`cross_kv`](Self::cross_kv), set in the cache), which over 1500
    /// encoder frames is the bulk of what caching saves. When the cache's
    /// [`group`](KVCache::group) is above one, `group` rows of `x` share each
    /// encoder row (`row = encoder_row * group + member`): a beam search's
    /// beams attend to their audio's one projection rather than a copy each.
    ///
    /// # Arguments
    /// * `x` : `[batch, seq_new, d_model]`, the new step(s) only.
    /// * `mask` : optional `[batch, seq_new, seq_past + seq_new]`. A single new
    ///   token needs none: it may attend to all of its history.
    /// * `cache`, `layer` : the decode's cache, and this block's layer in it.
    ///
    /// # Returns
    /// `[batch, seq_new, d_model]`.
    pub fn forward_w_kv_cache(
        &self,
        x: Tensor<3>,
        mask: Option<Tensor<3, Bool>>,
        cache: &mut KVCache,
        layer: usize,
    ) -> Tensor<3> {
        let x = x.clone().add(norm_self_attn_w_kv_cache(
            &self.attn_norm,
            &self.attn,
            x,
            mask,
            cache,
            layer,
        ));

        let x = x.clone().add(norm_cross_attn_w_kv_cache(
            &self.cross_attn_norm,
            &self.cross_attn,
            x,
            cache,
            layer,
        ));

        let mlp = norm_mlp(&self.mlp_norm, &self.mlp, x.clone());
        x + mlp
    }

    /// Projects the encoder output into this block's cross-attention keys and
    /// values, for the decode's cache
    /// ([`set_cross_kv`](KVCache::set_cross_kv)).
    ///
    /// Call once per decode; the result is valid for every step against the
    /// same `xa`.
    ///
    /// # Arguments
    /// * `xa` : `[batch, cross_len, d_model]` encoder output.
    ///
    /// # Returns
    /// The `[batch, heads, cross_len, d_k]` keys and values.
    pub fn cross_kv(
        &self,
        xa: Tensor<3>,
    ) -> (Tensor<4>, Tensor<4>) {
        project_kv(&self.cross_attn, xa)
    }

    /// `transformers`' Whisper decoder layer names onto this block's,
    /// relative to it: `encoder_attn`'s `{q,k,v,out}_proj` the
    /// cross-attention and `encoder_attn_layer_norm` its norm; the rest as
    /// an encoder layer's
    /// ([`ResidualEncoderAttentionBlock::whisper_hf_rekey`]).
    pub fn whisper_hf_rekey() -> Rekey {
        Rekey::keep_rest()
            .rename("encoder_attn.q_proj.", "cross_attn.query.")
            .rename("encoder_attn.k_proj.", "cross_attn.key.")
            .rename("encoder_attn.v_proj.", "cross_attn.value.")
            .rename("encoder_attn.out_proj.", "cross_attn.output.")
            .rename("encoder_attn_layer_norm.", "cross_attn_norm.")
            .mount("", "", ResidualEncoderAttentionBlock::whisper_hf_rekey())
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Shape,
        tensor::Distribution,
    };

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::DeviceMemoryGuard,
    };

    #[test]
    #[serial_test::serial]
    fn test_cross_decoder_forward() {
        use crate::support::testing::performance_device;
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

        let d_model = 128;

        let cfg = CrossDecoderBlockConfig::new(d_model);
        let n_heads = cfg.n_heads();

        assert_eq!(cfg.d_model(), d_model);
        assert_eq!(cfg.n_heads(), n_heads);

        let block: CrossDecoderBlock = cfg.init(&device);

        assert_eq!(block.d_model(), d_model);
        assert_eq!(block.n_heads(), n_heads);
        // Whisper's attention has no key bias; every other projection has one.
        for attn in [&block.attn, &block.cross_attn] {
            assert!(attn.key.bias.is_none());
            assert!(attn.query.bias.is_some());
            assert!(attn.value.bias.is_some());
            assert!(attn.output.bias.is_some());
        }

        let batch = 2;
        let seq_len = 10;
        let shape: Shape = [batch, seq_len, d_model].into();

        let x: Tensor<3> = Tensor::random(shape.clone(), Distribution::Default, &device);
        let xa: Tensor<3> = Tensor::random(shape.clone(), Distribution::Default, &device);

        let mask: Tensor<3> =
            Tensor::random([1, seq_len, seq_len], Distribution::Bernoulli(0.5), &device);
        let mask = mask.bool();

        let result = block.forward(x.clone(), xa.clone(), mask.clone().into());

        assert_eq!(result.batch_size(), batch);
        assert_eq!(result.d_model(), d_model);
        assert_eq!(result.seq_len(), seq_len);

        let expected = {
            let self_attn = norm_self_attn(
                &block.attn_norm,
                &block.attn,
                x.clone(),
                mask.clone().into(),
            );
            let x = x + self_attn.context;

            let cross_attn = norm_cross_attn(
                &block.cross_attn_norm,
                &block.cross_attn,
                x.clone(),
                xa.clone(),
            );
            let x = x + cross_attn.context;

            let mlp = norm_mlp(&block.mlp_norm, &block.mlp, x.clone());
            let x = x + mlp;

            CrossDecoderRecord {
                output: x,
                ca_weights: cross_attn.weights,
            }
        };

        result
            .output
            .clone()
            .into_data()
            .assert_approx_eq::<f64>(&expected.output.to_data(), Default::default());
        result
            .ca_weights
            .clone()
            .into_data()
            .assert_approx_eq::<f64>(&expected.ca_weights.to_data(), Default::default());

        assert_shape_contract!(
            ["batch", "seq_len", "d_model"],
            &result.output,
            &[("batch", batch), ("seq_len", seq_len), ("d_model", d_model),],
        );

        assert_shape_contract!(
            ["batch", "n_heads", "seq_len", "seq_len"],
            &result.ca_weights,
            &[("batch", batch), ("n_heads", n_heads), ("seq_len", seq_len)],
        );
    }
}
