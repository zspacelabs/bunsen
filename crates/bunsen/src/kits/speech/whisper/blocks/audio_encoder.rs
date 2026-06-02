use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{
        Embedding,
        EmbeddingConfig,
        LayerNorm,
        LayerNormConfig,
        PaddingConfig1d,
        activation::{
            Activation,
            ActivationConfig,
        },
        conv::{
            Conv1d,
            Conv1dConfig,
        },
    },
    prelude::{
        Backend,
        s,
    },
};

use crate::{
    contracts::unpack_shape_contract,
    kits::speech::whisper::blocks::{
        ResidualEncoderAttentionBlock,
        ResidualEncoderAttentionBlockConfig,
        ResidualEncoderAttentionBlockMeta,
    },
};

/// Common meta for [`AudioEncoder`] and [`AudioEncoderConfig`].
pub trait AudioEncoderMeta {
    /// Return the Mel-scale frequency resolution.
    fn n_mels(&self) -> usize;

    /// Return the max audio context size.
    fn max_context(&self) -> usize;

    /// The embedding size of the model.
    fn d_model(&self) -> usize;

    /// Return the number of heads.
    fn n_heads(&self) -> usize;

    /// Return the number of layers.
    fn n_layers(&self) -> usize;
}

/// Config for [`AudioEncoder`].
#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    /// The Mel-scale frequency resolution.
    pub n_mels: usize,

    /// The embedding size of the model.
    pub d_model: usize,
    /// Max audio context size.
    pub max_context: usize,

    /// Number of Audio Heads.
    pub n_heads: usize,

    /// Number of Audio Layers.
    pub n_layers: usize,

    /// Head Activation.
    #[config(default = "ActivationConfig::Gelu")]
    pub head_activation: ActivationConfig,
}

impl AudioEncoderMeta for AudioEncoderConfig {
    fn n_mels(&self) -> usize {
        self.n_mels
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn max_context(&self) -> usize {
        self.max_context
    }

    fn n_heads(&self) -> usize {
        self.n_heads
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }
}

impl AudioEncoderConfig {
    /// Initialize an `AudioEncoder`.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> AudioEncoder<B> {
        AudioEncoder {
            conv1: Conv1dConfig::new(self.n_mels, self.d_model, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            act1: self.head_activation.init(device),

            conv2: Conv1dConfig::new(self.d_model, self.d_model, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .with_stride(2)
                .init(device),
            act2: self.head_activation.init(device),

            positional_embedding: EmbeddingConfig::new(self.max_context / 2, self.d_model)
                .init(device),

            blocks: (0..self.n_layers)
                .map(|_| {
                    ResidualEncoderAttentionBlockConfig::new(self.d_model, self.n_heads)
                        .init(device)
                })
                .collect(),

            ln_post: LayerNormConfig::new(self.d_model).init(device),
        }
    }
}

/// Whisper Audio Encoder.
#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    conv1: Conv1d<B>,
    act1: Activation<B>,

    conv2: Conv1d<B>,
    act2: Activation<B>,

    positional_embedding: Embedding<B>,

    blocks: Vec<ResidualEncoderAttentionBlock<B>>,

    ln_post: LayerNorm<B>,
}

impl<B: Backend> AudioEncoderMeta for AudioEncoder<B> {
    fn n_mels(&self) -> usize {
        self.conv1.weight.dims()[1]
    }

    fn d_model(&self) -> usize {
        self.blocks[0].d_model()
    }

    fn max_context(&self) -> usize {
        2 * self.positional_embedding.weight.val().dims()[0]
    }

    fn n_heads(&self) -> usize {
        self.blocks[0].n_heads()
    }

    fn n_layers(&self) -> usize {
        self.blocks.len()
    }
}

impl<B: Backend> AudioEncoder<B> {
    /// Forward pass through the audio encoder.
    ///
    /// ## Arguments
    /// * `x`: The input audio spectrogram ``[batch, n_mels, seq]``.
    ///
    /// ## Returns
    /// ``[batch, seq, n_audio_states]``.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [_batch, seq_len] = unpack_shape_contract!(
            ["batch", "n_mels", "seq_len"],
            &x,
            &["batch", "seq_len"],
            &[("n_mels", self.n_mels())],
        );

        assert!(
            seq_len <= self.max_context(),
            "Audio length {} cannot exceed {}.",
            seq_len,
            self.max_context()
        );

        let x = self.act1.forward(self.conv1.forward(x));
        let x = self.act2.forward(self.conv2.forward(x));

        let x = x.swap_dims(1, 2);

        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "len", "n_audio_states"],
            &x,
            &[
                ("batch", _batch),
                ("len", seq_len / 2),
                ("n_audio_states", self.d_model()),
            ],
        );

        let k = x.dims()[1];
        let x = x + self
            .positional_embedding
            .weight
            .val()
            .slice(s![0..k])
            .unsqueeze::<3>();

        let x = self.blocks.iter().fold(x, |z, b| b.forward(z));

        self.ln_post.forward(x)
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
    fn test_audio_encoder_forward() {
        type B = PerformanceBackend;
        let device = Default::default();

        let d_model = 128;
        let n_mels = 256;
        let max_audio_ctx = 100;

        let n_audio_heads = 4;
        let n_audio_layers = 2;

        let config = AudioEncoderConfig::new(
            n_mels,
            d_model,
            max_audio_ctx,
            n_audio_heads,
            n_audio_layers,
        );

        assert_eq!(config.n_mels(), n_mels);
        assert_eq!(config.d_model(), d_model);
        assert_eq!(config.max_context(), max_audio_ctx);
        assert_eq!(config.n_heads(), n_audio_heads);
        assert_eq!(config.n_layers(), n_audio_layers);

        let encoder: AudioEncoder<B> = config.init(&device);

        assert_eq!(encoder.n_mels(), n_mels);
        assert_eq!(encoder.d_model(), d_model);
        assert_eq!(encoder.max_context(), max_audio_ctx);
        assert_eq!(encoder.n_heads(), n_audio_heads);
        assert_eq!(encoder.n_layers(), n_audio_layers);

        let batch = 2;
        let k = max_audio_ctx / 2;
        let x: Tensor<B, 3> = Tensor::random([batch, n_mels, k], Default::default(), &device);

        let output = encoder.forward(x.clone());

        assert_shape_contract!(
            ["batch", "seq", "n_audio_states"],
            &output,
            &[
                ("batch", batch),
                ("seq", k / 2),
                ("n_audio_states", d_model),
            ],
        );
    }
}
