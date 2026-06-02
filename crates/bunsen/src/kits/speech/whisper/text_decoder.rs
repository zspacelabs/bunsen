use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
        LayerNorm,
        LayerNormConfig,
    },
    prelude::{
        Backend,
        s,
    },
    tensor::{
        Bool,
        Int,
    },
};

use crate::{
    blocks::transformers::attention::causal_mask,
    kits::speech::whisper::{
        ResidualDecoderAttentionBlock,
        ResidualDecoderAttentionBlockConfig,
        ResidualDecoderAttentionBlockMeta,
    },
};

/// Common meta for [`TextDecoder`] and [`TextDecoderConfig`].
pub trait TextDecoderMeta {
    /// Return the size of the vocabulary.
    fn vocab_size(&self) -> usize;

    /// Return the max context size.
    fn max_context(&self) -> usize;

    /// The embedding size of the model.
    fn d_model(&self) -> usize;
}

/// Config for [`TextDecoder`].
#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,

    /// Maximum text context size.
    pub max_context: usize,

    /// The embedding size of the model.
    pub d_model: usize,

    /// The number of heads.
    pub n_heads: usize,

    /// The number of layers.
    pub n_layers: usize,

    /// Dropout.
    #[config(default = "0.0")]
    pub block_dropout: f64,
}

impl TextDecoderMeta for TextDecoderConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn d_model(&self) -> usize {
        self.d_model
    }
}

/// Build attention mask for decoder.
pub fn attn_decoder_mask<B: Backend>(
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length], device);

    for i in 0..(seq_length - 1) {
        let values =
            Tensor::<B, 2>::zeros([1, seq_length - (i + 1)], device).add_scalar(f64::NEG_INFINITY);
        mask = mask.slice_assign([i..i + 1, i + 1..seq_length], values);
    }

    mask
}

impl TextDecoderConfig {
    /// Initialize text decoder module.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> TextDecoder<B> {
        TextDecoder {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),

            positional_embedding: EmbeddingConfig::new(self.max_context, self.d_model).init(device),

            blocks: (0..self.n_layers)
                .map(|_| {
                    ResidualDecoderAttentionBlockConfig::new(self.d_model, self.n_heads)
                        .with_dropout(self.block_dropout)
                        .init(device)
                })
                .collect(),

            ln: LayerNormConfig::new(self.d_model).init(device),
            // mask: Param::from_tensor(attn_decoder_mask(self.max_text_context, device)),
        }
    }
}

/// Text decoder module for Whisper speech recognition model.
#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    /// The token embedding.
    pub token_embedding: Embedding<B>,

    /// The positional embedding.
    pub positional_embedding: Embedding<B>,

    /// The decoder blocks.
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,

    /// The output layer norm.
    pub ln: LayerNorm<B>,
    // mask: Param<Tensor<B, 2>>,
}

impl<B: Backend> TextDecoderMeta for TextDecoder<B> {
    fn vocab_size(&self) -> usize {
        self.token_embedding.weight.val().dims()[0]
    }

    fn max_context(&self) -> usize {
        self.positional_embedding.weight.val().dims()[0]
    }

    fn d_model(&self) -> usize {
        self.blocks[0].d_model()
    }
}

impl<B: Backend> TextDecoder<B> {
    /// Run the decoder.
    ///
    /// ## Arguments
    /// * `x`: ``[batch, seq]``.
    /// * `xa`: ``[batch, seq, d_model]``.
    ///
    /// ## Returns
    /// ``[batch, seq, n_vocab]``.
    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        xa: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.max_context(),
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.max_context()
        );

        let x = self.token_embedding.forward(x)
            + self
                .positional_embedding
                .weight
                .val()
                .slice(s![0..seq_len])
                .unsqueeze::<3>();

        //let mask = attn_decoder_mask(seq_len);

        let mask: Option<Tensor<B, 3, Bool>> = causal_mask(seq_len, 0, &x.device()).into();

        let x = self
            .blocks
            .iter()
            .fold(x, |z, b| b.forward(z, xa.clone(), mask.clone()).output);

        let x = self.ln.forward(x);
        x.matmul(
            self.token_embedding
                .weight
                .val()
                .transpose()
                .unsqueeze::<3>(),
        )

        // denorm [batch, seq_len, n_vocab]
        // Needs softmax / beamsearch.
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::PerformanceBackend,
    };

    #[test]
    #[serial]
    fn test_text_decoder_forward() {
        type B = PerformanceBackend;
        let device = Default::default();

        let d_model = 128;
        let vocab_size = 64;
        let max_context = 128;
        let n_heads = 4;
        let n_layers = 2;

        let config = TextDecoderConfig::new(vocab_size, max_context, d_model, n_heads, n_layers);

        assert_eq!(config.vocab_size(), vocab_size);
        assert_eq!(config.max_context(), max_context);
        assert_eq!(config.d_model(), d_model);

        let decoder: TextDecoder<B> = config.init(&device);

        assert_eq!(decoder.vocab_size(), vocab_size);
        assert_eq!(decoder.max_context(), max_context);
        assert_eq!(decoder.d_model(), d_model);

        let batch = 2;
        let seq_len = max_context / 2;

        let x: Tensor<B, 2, Int> = Tensor::zeros([batch, seq_len], &device);
        let xa: Tensor<B, 3> =
            Tensor::random([batch, seq_len, d_model], Default::default(), &device);

        let output = decoder.forward(x.clone(), xa.clone());

        assert_shape_contract!(
            ["batch", "seq", "n_vocab"],
            &output,
            &[("batch", batch), ("seq", seq_len), ("n_vocab", vocab_size),],
        );
    }
}
