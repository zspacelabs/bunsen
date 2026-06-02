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

/// [`Whisper`] structural config.
#[derive(Config, Debug)]
pub struct WhisperStructConfig {
    /// Encoder config.
    pub encoder: AudioEncoderConfig,

    /// Decoder config.
    pub decoder: TextDecoderConfig,
}

impl WhisperStructConfig {
    /// Initialize the Whisper model with the given configuration and device.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> Whisper<B> {
        let encoder = self.encoder.init(device);
        let decoder = self.decoder.init(device);

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
