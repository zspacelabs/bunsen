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
    fn n_vocab(&self) -> usize;

    /// Return the max text context size.
    fn max_text_context(&self) -> usize;

    /// Return the number of text states.
    fn n_text_state(&self) -> usize;
}

/// Config for [`TextDecoder`].
#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    /// The size of the vocabulary.
    pub n_vocab: usize,

    /// The max text context size.
    pub max_text_context: usize,

    /// The text embedding size.
    pub n_text_state: usize,

    /// The number of decoder heads.
    pub n_text_head: usize,

    /// The number of decoder layers.
    pub n_text_layer: usize,

    /// Dropout.
    #[config(default = "0.0")]
    pub block_dropout: f64,
}

impl TextDecoderMeta for TextDecoderConfig {
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    fn max_text_context(&self) -> usize {
        self.max_text_context
    }

    fn n_text_state(&self) -> usize {
        self.n_text_state
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
            token_embedding: EmbeddingConfig::new(self.n_vocab, self.n_text_state).init(device),

            positional_embedding: EmbeddingConfig::new(self.max_text_context, self.n_text_state)
                .init(device),

            blocks: (0..self.n_text_layer)
                .map(|_| {
                    ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                        .with_dropout(self.block_dropout)
                        .init(device)
                })
                .collect(),

            ln: LayerNormConfig::new(self.n_text_state).init(device),
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
    fn n_vocab(&self) -> usize {
        self.token_embedding.weight.val().dims()[0]
    }

    fn max_text_context(&self) -> usize {
        self.positional_embedding.weight.val().dims()[0]
    }

    fn n_text_state(&self) -> usize {
        self.blocks[0].n_states()
    }
}

impl<B: Backend> TextDecoder<B> {
    /// Run the decoder.
    ///
    /// ## Arguments
    /// * `x`: ``[batch, seq]``.
    /// * `xa`: ``[batch, seq, n_audio_states]``.
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
            seq_len <= self.max_text_context(),
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.max_text_context()
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

        let n_vocab = 64;
        let max_text_context = 128;
        let n_text_head = 4;
        let n_text_state = n_text_head * 32;
        let n_text_layer = 2;

        let config = TextDecoderConfig::new(
            n_vocab,
            max_text_context,
            n_text_state,
            n_text_head,
            n_text_layer,
        );

        assert_eq!(config.n_vocab(), n_vocab);
        assert_eq!(config.max_text_context(), max_text_context);
        assert_eq!(config.n_text_state(), n_text_state);

        let decoder: TextDecoder<B> = config.init(&device);

        assert_eq!(decoder.n_vocab(), n_vocab);
        assert_eq!(decoder.max_text_context(), max_text_context);
        assert_eq!(decoder.n_text_state(), n_text_state);

        let batch = 2;
        let seq_len = max_text_context / 2;

        let x: Tensor<B, 2, Int> = Tensor::zeros([batch, seq_len], &device);
        let xa: Tensor<B, 3> =
            Tensor::random([batch, seq_len, n_text_state], Default::default(), &device);

        let output = decoder.forward(x.clone(), xa.clone());

        assert_shape_contract!(
            ["batch", "seq", "n_vocab"],
            &output,
            &[("batch", batch), ("seq", seq_len), ("n_vocab", n_vocab),],
        );
    }
}
