use burn::{
    Tensor,
    config::Config,
    module::{
        Module,
        Param,
    },
    nn::{
        LayerNorm,
        LayerNormConfig,
    },
    prelude::{
        Backend,
        s,
    },
    tensor::{
        Bool,
        Distribution,
        Int,
        module::embedding,
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
    fn n_text_ctx(&self) -> usize;

    /// Return the number of text states.
    fn n_text_state(&self) -> usize;
}

/// Config for [`TextDecoder`].
#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    /// The size of the vocabulary.
    pub n_vocab: usize,

    /// The max text context size.
    pub n_text_ctx: usize,

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

    fn n_text_ctx(&self) -> usize {
        self.n_text_ctx
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
        // TODO: Use burn::nn::Embedding

        TextDecoder {
            token_embedding: Param::from_tensor(Tensor::random(
                [self.n_vocab, self.n_text_state],
                Distribution::Normal(0.0, 1.0),
                device,
            )),

            positional_embedding: Param::from_tensor(Tensor::random(
                [self.n_text_ctx, self.n_text_state],
                Distribution::Normal(0.0, 1.0),
                device,
            )),

            blocks: (0..self.n_text_layer)
                .map(|_| {
                    ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                        .with_dropout(self.block_dropout)
                        .init(device)
                })
                .collect(),

            ln: LayerNormConfig::new(self.n_text_state).init(device),
            // mask: Param::from_tensor(attn_decoder_mask(self.n_text_ctx, device)),
        }
    }
}

/// Text decoder module for Whisper speech recognition model.
#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    /// The token embedding.
    pub token_embedding: Param<Tensor<B, 2>>,

    /// The positional embedding.
    positional_embedding: Param<Tensor<B, 2>>,

    /// The decoder blocks.
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,

    /// The output layer norm.
    pub ln: LayerNorm<B>,
    // mask: Param<Tensor<B, 2>>,
}

impl<B: Backend> TextDecoderMeta for TextDecoder<B> {
    fn n_vocab(&self) -> usize {
        self.token_embedding.val().dims()[0]
    }

    fn n_text_ctx(&self) -> usize {
        self.positional_embedding.val().dims()[0]
    }

    fn n_text_state(&self) -> usize {
        self.blocks[0].n_states()
    }
}

impl<B: Backend> TextDecoder<B> {
    /// Run the decoder.
    pub fn forward(
        &self,
        x: Tensor<B, 2, Int>,
        xa: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_text_ctx(),
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx()
        );

        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
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
        x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>())
    }
}
