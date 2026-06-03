use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Backend,
        Int,
    },
};

use super::WHISPER_DEFAULT_D_MODEL;
use crate::kits::speech::whisper::blocks::{
    AudioEncoder,
    AudioEncoderConfig,
    AudioEncoderMeta,
    TextDecoder,
    TextDecoderConfig,
    TextDecoderMeta,
};

/// Whisper API config.
#[derive(Config, Debug)]
pub struct WhisperApiConfig {
    /// The Mel-scale frequency resolution.
    pub n_mels: usize,

    /// The size of the vocabulary.
    pub vocab_size: usize,

    /// Embedding Size of the Model.
    pub d_model: usize,

    /// Maximum Audio Encoder Context Size.
    pub max_audio_ctx: usize,

    /// Number Audio Encoder of Layers.
    pub n_encoder_layers: usize,

    /// Maximum Text Decoder Context Size.
    pub max_text_ctx: usize,

    /// Number Text Decoder of Layers.
    pub n_decoder_layers: usize,

    /// Head Dimensionality.
    #[config(defaul_value = "WHISPER_DEFAULT_D_MODEL")]
    pub d_head: usize,
}

impl WhisperApiConfig {
    /// Convert to a [`WhisperStructuralConfig`].
    pub fn to_structure(&self) -> WhisperStructuralConfig {
        WhisperStructuralConfig {
            encoder: AudioEncoderConfig::new(
                self.n_mels,
                self.d_model,
                self.max_audio_ctx,
                self.n_encoder_layers,
            )
            .with_d_head(self.d_head),
            decoder: TextDecoderConfig::new(
                self.vocab_size,
                self.d_model,
                self.max_text_ctx,
                self.n_decoder_layers,
            )
            .with_d_head(self.d_head),
        }
    }
}

/// Common meta for [`Whisper`] and [`WhisperApiConfig`].
pub trait WhisperMeta {
    /// Return the Mel-scale frequency resolution.
    fn n_mels(&self) -> usize {
        self.encoder().n_mels()
    }

    /// Return the vocabulary size.
    fn vocab_size(&self) -> usize {
        self.decoder().vocab_size()
    }

    /// Return the embedding size of the model.
    fn d_model(&self) -> usize {
        self.encoder().d_model()
    }

    /// The max audio context size.
    fn max_audio_ctx(&self) -> usize {
        self.encoder().max_context()
    }

    /// The max text context size.
    fn max_text_ctx(&self) -> usize {
        self.decoder().max_context()
    }

    /// Return the [`AudioEncoder`] meta.
    fn encoder(&self) -> &impl AudioEncoderMeta;

    /// Return the [`TextDecoder`] meta.
    fn decoder(&self) -> &impl TextDecoderMeta;
}

/// [`Whisper`] structural config.
#[derive(Config, Debug)]
pub struct WhisperStructuralConfig {
    /// Encoder config.
    pub encoder: AudioEncoderConfig,

    /// Decoder config.
    pub decoder: TextDecoderConfig,
}

impl WhisperMeta for WhisperStructuralConfig {
    fn encoder(&self) -> &impl AudioEncoderMeta {
        &self.encoder
    }

    fn decoder(&self) -> &impl TextDecoderMeta {
        &self.decoder
    }
}

impl WhisperStructuralConfig {
    /// Initialize the Whisper model with the given configuration and device.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> Whisper<B> {
        let encoder = self.encoder.init(device);
        let decoder = self.decoder.init(device);

        assert_eq!(encoder.d_model(), decoder.d_model());

        Whisper { encoder, decoder }
    }
}

/// Whisper model
#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    /// The [`AudioEncoder`].
    pub encoder: AudioEncoder<B>,

    /// The [`TextDecoder`].
    pub decoder: TextDecoder<B>,
}

impl<B: Backend> WhisperMeta for Whisper<B> {
    fn encoder(&self) -> &impl AudioEncoderMeta {
        &self.encoder
    }

    fn decoder(&self) -> &impl TextDecoderMeta {
        &self.decoder
    }
}

impl<B: Backend> Whisper<B> {
    /// Forward pass through the Whisper model.
    ///
    /// ## Arguments
    /// * `mel`: The input audio spectrogram ``[batch, n_mels, seq]``.
    /// * `tokens`: ``[batch, seq]``.
    ///
    /// ## Returns
    /// ``[batch, seq, vocab_size]``.
    pub fn forward(
        &self,
        mel: Tensor<B, 3>,
        tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        self.forward_decoder(tokens, self.forward_encoder(mel))
    }

    /// Forward pass through the Whisper encoder.
    ///
    /// ## Arguments
    /// * `mel`: The input audio spectrogram ``[batch, n_mels, seq]``.
    ///
    /// ## Returns
    /// ``[batch, seq, n_audio_states]``.
    pub fn forward_encoder(
        &self,
        mel: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    /// Forward pass through the Whisper decoder.
    ///
    /// ## Arguments
    /// * `tokens`: ``[batch, seq]``.
    /// * `encoder_output`: ``[batch, seq, d_model]``.
    ///
    /// ## Returns
    /// ``[batch, seq, vocab_size]``.
    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output)
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
    fn test_whisper_forward() {
        type B = PerformanceBackend;
        let device = Default::default();

        let d_model = 128;
        let n_mels = 80;
        let vocab_size = 64;

        let max_audio_ctx = 128;
        let n_audio_layers = 2;

        let max_text_context = 128;
        let n_text_layers = 2;

        let config = WhisperApiConfig::new(
            n_mels,
            vocab_size,
            d_model,
            max_audio_ctx,
            n_audio_layers,
            max_text_context,
            n_text_layers,
        );

        let structural = config.to_structure();

        let n_audio_heads = structural.encoder.n_heads();
        let n_text_heads = structural.decoder.n_heads();

        assert_eq!(structural.n_mels(), n_mels);
        assert_eq!(structural.vocab_size(), vocab_size);
        assert_eq!(structural.d_model(), d_model);
        assert_eq!(structural.max_audio_ctx(), max_audio_ctx);
        assert_eq!(structural.max_text_ctx(), max_text_context);

        assert_eq!(structural.encoder().n_mels(), n_mels);
        assert_eq!(structural.encoder().d_model(), d_model);
        assert_eq!(structural.encoder().max_context(), max_audio_ctx);
        assert_eq!(structural.encoder().n_heads(), n_audio_heads);
        assert_eq!(structural.encoder().n_layers(), n_audio_layers);

        assert_eq!(structural.decoder().vocab_size(), vocab_size);
        assert_eq!(structural.decoder().d_model(), d_model);
        assert_eq!(structural.decoder().max_context(), max_text_context);
        assert_eq!(structural.decoder().n_heads(), n_text_heads);
        assert_eq!(structural.decoder().n_layers(), n_text_layers);

        let model: Whisper<B> = structural.init(&device);

        assert_eq!(model.n_mels(), n_mels);
        assert_eq!(model.vocab_size(), vocab_size);
        assert_eq!(model.d_model(), d_model);
        assert_eq!(model.max_audio_ctx(), max_audio_ctx);
        assert_eq!(model.max_text_ctx(), max_text_context);

        assert_eq!(model.encoder().n_mels(), n_mels);
        assert_eq!(model.encoder().d_model(), d_model);
        assert_eq!(model.encoder().max_context(), max_audio_ctx);
        assert_eq!(model.encoder().n_heads(), n_audio_heads);
        assert_eq!(model.encoder().n_layers(), n_audio_layers);

        assert_eq!(model.decoder().vocab_size(), vocab_size);
        assert_eq!(model.decoder().d_model(), d_model);
        assert_eq!(model.decoder().max_context(), max_text_context);
        assert_eq!(model.decoder().n_heads(), n_text_heads);
        assert_eq!(model.decoder().n_layers(), n_text_layers);

        let batch = 2;
        let audio_len = max_audio_ctx / 2;
        // The encoder halves the audio sequence (conv stride 2); the decoder's
        // cross-attention expects the token sequence to match that length.
        let token_len = audio_len / 2;

        let mel: Tensor<B, 3> =
            Tensor::random([batch, n_mels, audio_len], Default::default(), &device);
        let tokens: Tensor<B, 2, Int> = Tensor::zeros([batch, token_len], &device);

        // The encoder halves the audio sequence length (conv stride 2).
        let encoder_output = model.forward_encoder(mel.clone());
        assert_shape_contract!(
            ["batch", "seq", "d_model"],
            &encoder_output,
            &[
                ("batch", batch),
                ("seq", audio_len / 2),
                ("d_model", d_model),
            ],
        );

        // The full forward pass produces vocab logits per input token.
        let output = model.forward(mel, tokens);
        assert_shape_contract!(
            ["batch", "seq", "vocab_size"],
            &output,
            &[
                ("batch", batch),
                ("seq", token_len),
                ("vocab_size", vocab_size)
            ],
        );
    }
}
