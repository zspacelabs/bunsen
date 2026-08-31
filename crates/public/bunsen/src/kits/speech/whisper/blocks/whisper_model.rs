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
use crate::{
    burner::{
        module::ModuleInit,
        store::FixPytorchLoadMappers,
    },
    kits::speech::whisper::blocks::{
        AudioEncoder,
        AudioEncoderConfig,
        AudioEncoderMeta,
        TextDecoder,
        TextDecoderConfig,
        TextDecoderMeta,
    },
};

/// Whisper API config.
///
/// User-facing configuration for the [`Whisper`] model, exposing the flat
/// hyperparameters (Mel resolution, vocabulary, model/head sizes, context
/// limits, and encoder/decoder layer counts). Expands via
/// [`to_structure`](WhisperApiConfig::to_structure) into a
/// [`WhisperStructuralConfig`], and builds the [`Whisper`] module directly via
/// [`ModuleInit`].
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
    /// Converts to a [`WhisperStructuralConfig`].
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

impl<B: Backend> ModuleInit<B, Whisper<B>> for WhisperApiConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> crate::errors::BunsenResult<Whisper<B>> {
        self.to_structure().try_init(device)
    }
}

/// Common meta for [`Whisper`] and [`WhisperApiConfig`].
pub trait WhisperMeta {
    /// Returns the Mel-scale frequency resolution.
    fn n_mels(&self) -> usize {
        self.encoder().n_mels()
    }

    /// Returns the vocabulary size.
    fn vocab_size(&self) -> usize {
        self.decoder().vocab_size()
    }

    /// Returns the embedding size of the model.
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

    /// Returns the [`AudioEncoder`] meta.
    fn encoder(&self) -> &impl AudioEncoderMeta;

    /// Returns the [`TextDecoder`] meta.
    fn decoder(&self) -> &impl TextDecoderMeta;
}

/// [`Whisper`] structural config.
///
/// The fully-expanded structural configuration for the [`Whisper`] model,
/// pairing an [`AudioEncoderConfig`] with a [`TextDecoderConfig`]. Builds the
/// [`Whisper`] module via [`ModuleInit`].
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

impl<B: Backend> ModuleInit<B, Whisper<B>> for WhisperStructuralConfig {
    fn try_init(
        &self,
        device: &B::Device,
    ) -> crate::errors::BunsenResult<Whisper<B>> {
        let encoder = self.encoder.init(device);
        let decoder = self.decoder.init(device);

        assert_eq!(encoder.d_model(), decoder.d_model());

        Ok(Whisper { encoder, decoder })
    }
}

/// Whisper model
///
/// End-to-end Whisper speech recognition model, composing an [`AudioEncoder`]
/// over the input log-Mel spectrogram with a [`TextDecoder`] that
/// cross-attends to the encoder output to produce vocabulary logits.
///
/// Built by [`WhisperApiConfig`].
#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    /// The [`AudioEncoder`].
    pub encoder: AudioEncoder<B>,

    /// The [`TextDecoder`].
    pub decoder: TextDecoder<B>,
}

impl<B: Backend> FixPytorchLoadMappers for Whisper<B> {
    fn fix_pytorch_load_mappers(mut self) -> Self {
        self.encoder = self.encoder.fix_pytorch_load_mappers();
        self.decoder = self.decoder.fix_pytorch_load_mappers();
        self
    }
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
    /// # Arguments
    /// * `mel`: The input audio spectrogram `[batch, n_mels, seq]`.
    /// * `tokens`: `[batch, seq]`.
    ///
    /// # Returns
    /// `[batch, seq, vocab_size]`.
    pub fn forward(
        &self,
        mel: Tensor<B, 3>,
        tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        self.forward_decoder(tokens, self.forward_encoder(mel))
    }

    /// Forward pass through the Whisper encoder.
    ///
    /// # Arguments
    /// * `mel`: The input audio spectrogram `[batch, n_mels, seq]`.
    ///
    /// # Returns
    /// `[batch, seq, n_audio_states]`.
    pub fn forward_encoder(
        &self,
        mel: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    /// Forward pass through the Whisper decoder.
    ///
    /// # Arguments
    /// * `tokens`: `[batch, seq]`.
    /// * `encoder_output`: `[batch, seq, d_model]`.
    ///
    /// # Returns
    /// `[batch, seq, vocab_size]`.
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
    use burn::{
        module::Param,
        prelude::{
            Device,
            TensorData,
        },
        tensor::DType,
    };
    use serial_test::serial;

    use super::*;
    use crate::{
        contracts::assert_shape_contract,
        support::testing::{
            CpuBackend,
            PerformanceBackend,
            param_load_mapping,
        },
    };

    /// Whether `param` carries a load-path mapping, read through a `[2, 3]`
    /// probe that the repair visibly reorders.
    fn is_mapped(
        param: &Param<Tensor<CpuBackend, 2>>,
        device: &Device<CpuBackend>,
    ) -> bool {
        let probe: Tensor<CpuBackend, 2> = Tensor::from_data(
            TensorData::new((0..6).map(|v| v as f64).collect::<Vec<_>>(), [2, 3]),
            device,
        );

        let flat = |t: Tensor<CpuBackend, 2>| -> Vec<f32> {
            t.cast(DType::F32).to_data().to_vec().unwrap()
        };

        flat(param_load_mapping(param, probe.clone())) != flat(probe)
    }

    /// **The walk's coverage.** Every `Linear` weight in the model must carry
    /// the repair, and nothing else may: a missed projection loads scrambled,
    /// and an extra one — on a parameter the checkpoint stores contiguously —
    /// is a silent transpose.
    #[test]
    fn test_fix_pytorch_load_mappers_covers_the_projections() {
        type B = CpuBackend;

        let device: Device<B> = Default::default();

        // `n_heads` is `d_model / d_head`, and `d_head` defaults to 64.
        let model: Whisper<B> = WhisperApiConfig::new(8, 16, 128, 16, 1, 16, 1)
            .try_init(&device)
            .unwrap();

        // Initialization leaves the parameters alone; the repair is attached
        // only on the way to a PyTorch load.
        assert!(!is_mapped(
            &model.encoder.blocks[0].attn.query.weight,
            &device
        ));

        let model = model.fix_pytorch_load_mappers();

        let encoder = &model.encoder.blocks[0];
        let decoder = &model.decoder.blocks[0];
        for weight in [
            &encoder.attn.query.weight,
            &encoder.attn.key.weight,
            &encoder.attn.value.weight,
            &encoder.attn.output.weight,
            &encoder.mlp.linear1.weight,
            &encoder.mlp.linear2.weight,
            &decoder.attn.query.weight,
            &decoder.attn.key.weight,
            &decoder.attn.value.weight,
            &decoder.attn.output.weight,
            &decoder.cross_attn.query.weight,
            &decoder.cross_attn.key.weight,
            &decoder.cross_attn.value.weight,
            &decoder.cross_attn.output.weight,
            &decoder.mlp.linear1.weight,
            &decoder.mlp.linear2.weight,
        ] {
            assert!(is_mapped(weight, &device));
        }

        // The embeddings are stored contiguously — the conv head's weights
        // are rank-3, which the repair cannot reach at all.
        assert!(!is_mapped(&model.encoder.positional_embedding, &device));
        assert!(!is_mapped(&model.decoder.positional_embedding, &device));
        assert!(!is_mapped(&model.decoder.token_embedding.weight, &device));
    }

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

        let model: Whisper<B> = structural.try_init(&device).unwrap();

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
