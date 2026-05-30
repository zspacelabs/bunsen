use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn,
    nn::attention::{
        MhaInput,
        MultiHeadAttention,
        MultiHeadAttentionConfig,
    },
    prelude::{
        Backend,
        Bool,
    },
};

use crate::blocks::transformers::mlp::{
    Mlp,
    MlpConfig,
};

/// Common meta for [`ResidualDecoderAttentionBlock`] and
/// [`ResidualDecoderAttentionBlockConfig`].
pub trait ResidualDecoderAttentionBlockMeta {
    /// Return the number of states.
    fn n_state(&self) -> usize;

    /// Return the number of heads.
    fn n_heads(&self) -> usize;
}

/// Config for [`ResidualDecoderAttentionBlock`].
#[derive(Config, Debug)]
pub struct ResidualDecoderAttentionBlockConfig {
    /// Number of States.
    pub n_state: usize,

    /// Number of Heads.
    pub n_head: usize,
}

impl ResidualDecoderAttentionBlockMeta for ResidualDecoderAttentionBlockConfig {
    fn n_state(&self) -> usize {
        self.n_state
    }

    fn n_heads(&self) -> usize {
        self.n_head
    }
}

impl ResidualDecoderAttentionBlockConfig {
    /// Initialize the residual decoder attention block.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ResidualDecoderAttentionBlock<B> {
        let attn = MultiHeadAttentionConfig::new(self.n_state, self.n_head)
            .with_dropout(0.0)
            .init(device);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let cross_attn = MultiHeadAttentionConfig::new(self.n_state, self.n_head)
            .with_dropout(0.0)
            .init(device);
        let cross_attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let mlp = MlpConfig::new(self.n_state).init(device);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

/// Residual Decoder Attention Block for Whisper.
#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    /// Attention.
    pub attn: MultiHeadAttention<B>,

    /// Attention Normalization.
    pub attn_ln: nn::LayerNorm<B>,

    /// Cross Attention.
    pub cross_attn: MultiHeadAttention<B>,

    /// Cross Attention Normalization.
    pub cross_attn_ln: nn::LayerNorm<B>,

    /// MLP.
    pub mlp: Mlp<B>,

    /// MLP Normalization.
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlockMeta for ResidualDecoderAttentionBlock<B> {
    fn n_state(&self) -> usize {
        self.attn.d_model
    }

    fn n_heads(&self) -> usize {
        self.attn.n_heads
    }
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    /// Forward pass of the residual decoder attention block.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Tensor<B, 3> {
        self.forward_with_cross_attention(x, xa, mask).0
    }

    /// Forward pass of the residual decoder attention block with cross
    /// attention.
    pub fn forward_with_cross_attention(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let self_attn_out = self
            .attn
            .forward(MhaInput::self_attn(self.attn_ln.forward(x.clone())).mask_attn(mask))
            .context;
        let x = x + self_attn_out;

        let cross_attn_out = self.cross_attn.forward(MhaInput::new(
            self.cross_attn_ln.forward(x.clone()),
            xa.clone(),
            xa,
        ));
        let x = x + cross_attn_out.context;

        let output = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        (output, cross_attn_out.weights)
    }
}
