//! The token embedding.

use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::Initializer,
    prelude::Int,
    tensor::Device,
};

use crate::{
    burner::{
        module::ModuleInit,
        tensor::round_scalar_to_dtype,
    },
    errors::BunsenResult,
    ops::norm::RmsNormOptions,
};

/// Common meta for [`TokenEmbedding`] and [`TokenEmbeddingConfig`].
pub trait TokenEmbeddingMeta {
    /// The vocabulary: the table's rows.
    fn n_vocab(&self) -> usize;

    /// The embedding's width: the table's columns.
    fn d_model(&self) -> usize;
}

/// Config for [`TokenEmbedding`].
#[derive(Config, Debug)]
pub struct TokenEmbeddingConfig {
    /// The vocabulary: the table's rows.
    pub n_vocab: usize,

    /// The embedding's width.
    pub d_model: usize,

    /// The epsilon of a parameter-free RMS norm on each looked-up row
    /// (nanochat's `norm(wte(idx))`); `None`, no norm.
    #[config(default = "None")]
    pub norm_eps: Option<f64>,

    /// A scale on each looked-up row, after any norm (Gemma's `√d_model`);
    /// `None`, no scale.
    #[config(default = "None")]
    pub scale: Option<f64>,

    /// The table's initializer.
    #[config(default = "Initializer::Normal { mean: 0.0, std: 1.0 }")]
    pub initializer: Initializer,
}

impl TokenEmbeddingMeta for TokenEmbeddingConfig {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn d_model(&self) -> usize {
        self.d_model
    }
}

impl ModuleInit<TokenEmbedding> for TokenEmbeddingConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<TokenEmbedding> {
        Ok(TokenEmbedding {
            weight: self.initializer.init([self.n_vocab, self.d_model], device),
            norm_eps: self.norm_eps,
            scale: self.scale,
        })
    }
}

/// A token embedding: a table's rows looked up by id, then, as configured,
/// normalized (a parameter-free RMS norm) and scaled.
///
/// The table is `weight`, as in burn's `Embedding`, so a checkpoint's
/// `embed_tokens.weight` loads as it would there. The scale is rounded to the
/// table's dtype before it is applied, as `transformers` casts Gemma's: in
/// bf16, `√1152` is `34.0`.
///
/// Built by [`TokenEmbeddingConfig`].
#[derive(Module, Debug)]
pub struct TokenEmbedding {
    /// `[n_vocab, d_model]`: the table.
    pub weight: Param<Tensor<2>>,

    /// The epsilon of the norm on each row, if there is one.
    #[module(skip)]
    pub norm_eps: Option<f64>,

    /// The scale on each row, if there is one.
    #[module(skip)]
    pub scale: Option<f64>,
}

impl TokenEmbeddingMeta for TokenEmbedding {
    fn n_vocab(&self) -> usize {
        self.weight.dims()[0]
    }

    fn d_model(&self) -> usize {
        self.weight.dims()[1]
    }
}

impl TokenEmbedding {
    /// The scale as applied: rounded to the table's dtype.
    pub fn applied_scale(&self) -> Option<f64> {
        let dtype = crate::burner::tensor::param_float_dtype(&self.weight);
        self.scale.map(|scale| round_scalar_to_dtype(scale, dtype))
    }

    /// Embeds tokens.
    ///
    /// # Arguments
    /// * `ids` - `[batch, seq]`.
    ///
    /// # Returns
    /// `[batch, seq, d_model]`, in the table's dtype.
    pub fn forward(
        &self,
        ids: Tensor<2, Int>,
    ) -> Tensor<3> {
        let x = burn::tensor::module::embedding(self.weight.val(), ids);
        let x = match self.norm_eps {
            Some(eps) => RmsNormOptions::default().with_eps(eps as f32).norm(x),
            None => x,
        };
        match self.applied_scale() {
            Some(scale) => x.mul_scalar(scale),
            None => x,
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        TensorData,
        Tolerance,
    };

    use super::*;
    use crate::support::testing::cpu_device;

    fn table(device: &Device) -> TokenEmbedding {
        let mut embedding: TokenEmbedding = TokenEmbeddingConfig::new(3, 2).init(device);
        embedding.weight = Param::from_tensor(Tensor::from_floats(
            [[1.0, 2.0], [3.0, -4.0], [0.5, 0.5]],
            device,
        ));
        embedding
    }

    /// Without a norm or a scale, a row is the table's.
    #[test]
    fn test_lookup() {
        let device = cpu_device();
        let embedding = table(&device);
        assert_eq!((embedding.n_vocab(), embedding.d_model()), (3, 2));
        let ids = Tensor::<2, Int>::from_ints([[1, 0, 1]], &device);
        embedding.forward(ids).into_data().assert_eq(
            &TensorData::from([[[3.0f32, -4.0], [1.0, 2.0], [3.0, -4.0]]]),
            false,
        );
    }

    /// The norm makes each row's mean square one, and the scale follows it.
    #[test]
    fn test_norm_then_scale() {
        let device = cpu_device();
        let mut embedding = table(&device);
        embedding.norm_eps = Some(0.0);
        embedding.scale = Some(2.0);
        let ids = Tensor::<2, Int>::from_ints([[1, 2]], &device);
        // [3, -4]: rms 3.5355; [0.5, 0.5]: rms 0.5.
        let r = (12.5f32).sqrt();
        embedding.forward(ids).into_data().assert_approx_eq::<f32>(
            &TensorData::from([[[6.0 / r, -8.0 / r], [2.0, 2.0]]]),
            Tolerance::default(),
        );
    }

    /// The scale is rounded to the table's dtype
    /// ([`round_scalar_to_dtype`]), as `transformers` rounds Gemma's.
    #[test]
    fn test_scale_rounds_to_the_table_dtype() {
        let s = 1152f64.sqrt();
        let device = cpu_device();
        let mut embedding = table(&device);
        embedding.scale = Some(s);
        assert_eq!(embedding.applied_scale(), Some(s as f32 as f64));
    }
}
