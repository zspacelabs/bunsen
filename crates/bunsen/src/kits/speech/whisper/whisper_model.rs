use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::{
        Backend,
        Int,
    },
};

use crate::kits::speech::whisper::{
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

    /// Number of Audio Context.
    pub max_audio_ctx: usize,

    /// Embedding Size of the Model.
    pub d_model: usize,

    /// Number of Audio Heads.
    pub n_audio_heads: usize,

    /// Number of Audio Layers.
    pub n_audio_layers: usize,

    /// The size of the vocabulary.
    pub n_vocab: usize,

    /// The max text context size.
    pub max_text_context: usize,

    /// The number of decoder heads.
    pub n_text_head: usize,

    /// The number of decoder layers.
    pub n_text_layer: usize,
}

impl WhisperApiConfig {
    /// Convert to a [`WhisperStructuralConfig`].
    pub fn to_structural_config(self) -> WhisperStructuralConfig {
        WhisperStructuralConfig {
            encoder: AudioEncoderConfig::new(
                self.n_mels,
                self.max_audio_ctx,
                self.d_model,
                self.n_audio_heads,
                self.n_audio_layers,
            ),
            decoder: TextDecoderConfig::new(
                self.n_vocab,
                self.max_text_context,
                self.d_model,
                self.n_text_head,
                self.n_text_layer,
            ),
        }
    }
}

/// Common meta for [`Whisper`] and [`WhisperApiConfig`].
pub trait WhisperMeta {
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

        assert_eq!(encoder.n_audio_states(), decoder.n_text_state());

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
    /// ``[batch, seq, n_vocab]``.
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
    /// * `encoder_output`: ``[batch, seq, n_audio_states]``.
    ///
    /// ## Returns
    /// ``[batch, seq, n_vocab]``.
    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output)
    }

    /// The max audio context size.
    pub fn max_encoder_ctx(&self) -> usize {
        self.encoder.max_audio_ctx()
    }

    /// The max text context size.
    pub fn max_decoder_ctx(&self) -> usize {
        self.decoder.max_text_context()
    }
}
