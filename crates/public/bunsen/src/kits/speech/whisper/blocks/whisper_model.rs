use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::Int,
    tensor::{
        DType,
        Device,
    },
};

use super::{
    WHISPER_DEFAULT_D_MODEL,
    WhisperFrontEndConfig,
    WhisperTokenLayoutConfig,
};
use crate::{
    burner::module::{
        HasDType,
        ModuleInit,
        ToStructureConfig,
    },
    errors::{
        BunsenResult,
        ConstraintError,
        Rule,
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
/// limits, and encoder/decoder layer counts): the policy of `Whisper`'s
/// Stacked Config. It implements [`ToStructureConfig`], expanding into a
/// [`WhisperStructureConfig`], and gets [`ModuleInit`] from that trait's
/// blanket impl, so `.init(device)` builds the [`Whisper`] module directly.
#[derive(Config, Debug)]
pub struct WhisperApiConfig {
    /// The Mel-scale frequency resolution.
    pub n_mels: usize,

    /// The audio front end the checkpoint's log-mels were computed
    /// with.
    ///
    /// A checkpoint does not record it; it is the convention of the
    /// pipeline that trained it, and the loader declares it. Every
    /// sample-domain quantity of the front end is derived from it.
    #[config(default = "WhisperFrontEndConfig::new()")]
    pub front_end: WhisperFrontEndConfig,

    /// The token layout the checkpoint's vocabulary follows: what its size
    /// does not say &mdash; the language codes, the spellings, the timestamp
    /// grid. Upstream's by default.
    #[config(default = "WhisperTokenLayoutConfig::new()")]
    pub token_layout: WhisperTokenLayoutConfig,

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
    #[config(default = "WHISPER_DEFAULT_D_MODEL")]
    pub d_head: usize,
}

impl ToStructureConfig for WhisperApiConfig {
    type Structure = WhisperStructureConfig;

    fn try_to_structure(&self) -> BunsenResult<WhisperStructureConfig> {
        Ok(WhisperStructureConfig {
            front_end: self.front_end.clone(),
            token_layout: self.token_layout.clone(),
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
        })
    }
}

/// [`Whisper`] Meta: the dimensions a built model and its structure config
/// both answer, so a caller reads them the same way before and after
/// [`ModuleInit`].
///
/// Implemented by:
/// * [`WhisperStructureConfig`]
/// * [`Whisper`]
pub trait WhisperMeta {
    /// The audio front end the model's log-mels are computed with.
    fn front_end(&self) -> &WhisperFrontEndConfig;

    /// The token layout the model's vocabulary follows.
    fn token_layout(&self) -> &WhisperTokenLayoutConfig;

    /// The sample rate the model's log-mels are computed at, in Hz.
    fn sample_rate(&self) -> usize {
        self.front_end().sample_rate
    }

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

/// [`Whisper`] structure config.
///
/// The fully-expanded structural configuration for the [`Whisper`] model,
/// pairing an [`AudioEncoderConfig`] with a [`TextDecoderConfig`].
/// [`WhisperApiConfig`] lowers to it. Builds the [`Whisper`] module via
/// [`ModuleInit`].
///
/// Implements [`WhisperMeta`].
#[derive(Config, Debug)]
pub struct WhisperStructureConfig {
    /// The audio front end the model's log-mels are computed with.
    pub front_end: WhisperFrontEndConfig,

    /// The token layout the model's vocabulary follows.
    pub token_layout: WhisperTokenLayoutConfig,

    /// Encoder config.
    pub encoder: AudioEncoderConfig,

    /// Decoder config.
    pub decoder: TextDecoderConfig,
}

impl WhisperMeta for WhisperStructureConfig {
    fn front_end(&self) -> &WhisperFrontEndConfig {
        &self.front_end
    }

    fn token_layout(&self) -> &WhisperTokenLayoutConfig {
        &self.token_layout
    }

    fn encoder(&self) -> &impl AudioEncoderMeta {
        &self.encoder
    }

    fn decoder(&self) -> &impl TextDecoderMeta {
        &self.decoder
    }
}

impl ModuleInit<Whisper> for WhisperStructureConfig {
    /// Builds the [`Whisper`] module.
    ///
    /// # Errors
    ///
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, when the encoder's and the decoder's
    /// `d_model` differ, which a hand-edited structure can do;
    /// [`WhisperApiConfig`] never lowers to one.
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<Whisper> {
        let (encoder_width, decoder_width) = (self.encoder.d_model(), self.decoder.d_model());
        if encoder_width != decoder_width {
            return Err(ConstraintError::new(
                "WhisperStructureConfig",
                "",
                Rule::Relation {
                    lhs: ("encoder.d_model".into(), encoder_width.to_string()),
                    op: "==",
                    rhs: ("decoder.d_model".into(), decoder_width.to_string()),
                },
            )
            .into());
        }

        let encoder = self.encoder.try_init(device)?;
        let decoder = self.decoder.try_init(device)?;

        Ok(Whisper {
            front_end: self.front_end.clone(),
            token_layout: self.token_layout.clone(),
            encoder,
            decoder,
        })
    }
}

/// Whisper model
///
/// End-to-end Whisper speech recognition model, composing an [`AudioEncoder`]
/// over the input log-Mel spectrogram with a [`TextDecoder`] that
/// cross-attends to the encoder output to produce vocabulary logits.
///
/// Implements [`WhisperMeta`].
///
/// Built by [`WhisperApiConfig`] (high-level) or [`WhisperStructureConfig`].
#[derive(Module, Debug)]
pub struct Whisper {
    /// The audio front end the log-mels are computed with. A
    /// constant of the module, not part of its record: set by the config at
    /// `init`, it survives a checkpoint load, and the config carries it
    /// across a record round trip.
    #[module(skip)]
    front_end: WhisperFrontEndConfig,

    /// The token layout the vocabulary follows. Likewise not part of the
    /// record.
    #[module(skip)]
    token_layout: WhisperTokenLayoutConfig,

    /// The [`AudioEncoder`].
    pub encoder: AudioEncoder,

    /// The [`TextDecoder`].
    pub decoder: TextDecoder,
}

impl WhisperMeta for Whisper {
    fn front_end(&self) -> &WhisperFrontEndConfig {
        &self.front_end
    }

    fn token_layout(&self) -> &WhisperTokenLayoutConfig {
        &self.token_layout
    }

    fn encoder(&self) -> &impl AudioEncoderMeta {
        &self.encoder
    }

    fn decoder(&self) -> &impl TextDecoderMeta {
        &self.decoder
    }
}

impl HasDType for Whisper {
    /// The dtype the model's parameters were loaded in, and so the one it
    /// computes in. `OpenAI`'s checkpoints ship in fp16.
    ///
    /// This is **not** the dtype of the model's interface. Log-mels go in at
    /// whatever float the front end produced and are cast here; logits come
    /// out in the backend's default float
    /// ([`device_float_dtype`](crate::burner::tensor::device_float_dtype)).
    /// What stays at this precision is everything between: the encoder
    /// features, and the cross- and self-attention caches projected from
    /// them.
    ///
    /// # Panics
    /// If the encoder and the decoder are at different precisions. No
    /// checkpoint ships that and no loader here produces it; only mapping
    /// one half on its own can.
    fn dtype(&self) -> DType {
        let (encoder, decoder) = (self.encoder.dtype(), self.decoder.dtype());
        assert_eq!(
            encoder, decoder,
            "the encoder and the decoder are at different precisions",
        );
        encoder
    }
}

impl Whisper {
    /// Forward pass through the Whisper model.
    ///
    /// # Arguments
    /// * `mel`: The input audio spectrogram `[batch, n_mels, seq]`, in any
    ///   float dtype.
    /// * `tokens`: `[batch, seq]`.
    ///
    /// # Returns
    /// `[batch, seq, vocab_size]` logits, in the backend's default float.
    pub fn forward(
        &self,
        mel: Tensor<3>,
        tokens: Tensor<2, Int>,
    ) -> Tensor<3> {
        self.forward_decoder(tokens, self.forward_encoder(mel))
    }

    /// Forward pass through the Whisper encoder.
    ///
    /// # Arguments
    /// * `mel`: The input audio spectrogram `[batch, n_mels, seq]`, in any
    ///   float dtype; the mel front end works in the backend's default.
    ///
    /// # Returns
    /// `[batch, seq, n_audio_states]`, in the model's [`dtype`](Self::dtype)
    /// — these features feed the decoder and its cross-attention cache, and
    /// are not an interface value. See [`AudioEncoder::forward`].
    pub fn forward_encoder(
        &self,
        mel: Tensor<3>,
    ) -> Tensor<3> {
        self.encoder.forward(mel)
    }

    /// Forward pass through the Whisper decoder.
    ///
    /// # Arguments
    /// * `tokens`: `[batch, seq]`.
    /// * `encoder_output`: `[batch, seq, d_model]`, in any float dtype.
    ///
    /// # Returns
    /// `[batch, seq, vocab_size]` logits, in the backend's default float.
    /// See [`TextDecoder::forward`].
    pub fn forward_decoder(
        &self,
        tokens: Tensor<2, Int>,
        encoder_output: Tensor<3>,
    ) -> Tensor<3> {
        self.decoder.forward(tokens, encoder_output)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use serial_test::serial;

    use super::*;
    use crate::{
        burner::{
            module::DTypeMapper,
            tensor::device_float_dtype,
        },
        contracts::assert_shape_contract,
        errors::{
            BunsenErrorKind,
            testing::ErrorMatcher,
        },
        support::testing::{
            DeviceMemoryGuard,
            cpu_device,
            performance_device,
        },
    };

    /// The front end and the token layout default to upstream's and ride
    /// config -> structure -> module.
    #[test]
    fn test_front_end_and_layout_propagate() {
        let device = cpu_device();

        let config = WhisperApiConfig::new(8, 16, 128, 16, 1, 16, 1);
        assert_eq!(config.front_end, WhisperFrontEndConfig::new());
        assert_eq!(config.token_layout, WhisperTokenLayoutConfig::new());
        assert_eq!(config.to_structure().sample_rate(), 16_000);
        let model: Whisper = config.try_init(&device).unwrap();
        assert_eq!(model.sample_rate(), 16_000);
        assert_eq!(model.token_layout().languages.len(), 100);

        let config = config
            .with_front_end(WhisperFrontEndConfig::new().with_sample_rate(8_000))
            .with_token_layout(WhisperTokenLayoutConfig::new().with_timestamp_tokens(751));
        assert_eq!(config.to_structure().sample_rate(), 8_000);
        let model: Whisper = config.try_init(&device).unwrap();
        assert_eq!(model.sample_rate(), 8_000);
        assert_eq!(model.front_end().hop(), 80);
        assert_eq!(model.token_layout().timestamp_tokens, 751);
    }

    /// Asserts that `a` and `b` answer every [`WhisperMeta`] method alike,
    /// down through the encoder's and the decoder's meta.
    fn assert_meta_agrees(
        a: &impl WhisperMeta,
        b: &impl WhisperMeta,
    ) {
        assert_eq!(a.front_end(), b.front_end());
        assert_eq!(a.token_layout(), b.token_layout());
        assert_eq!(a.sample_rate(), b.sample_rate());
        assert_eq!(a.n_mels(), b.n_mels());
        assert_eq!(a.vocab_size(), b.vocab_size());
        assert_eq!(a.d_model(), b.d_model());
        assert_eq!(a.max_audio_ctx(), b.max_audio_ctx());
        assert_eq!(a.max_text_ctx(), b.max_text_ctx());

        let (a_enc, b_enc) = (a.encoder(), b.encoder());
        assert_eq!(a_enc.n_mels(), b_enc.n_mels());
        assert_eq!(a_enc.max_context(), b_enc.max_context());
        assert_eq!(a_enc.d_model(), b_enc.d_model());
        assert_eq!(a_enc.n_heads(), b_enc.n_heads());
        assert_eq!(a_enc.n_layers(), b_enc.n_layers());

        let (a_dec, b_dec) = (a.decoder(), b.decoder());
        assert_eq!(a_dec.vocab_size(), b_dec.vocab_size());
        assert_eq!(a_dec.d_model(), b_dec.d_model());
        assert_eq!(a_dec.max_context(), b_dec.max_context());
        assert_eq!(a_dec.n_heads(), b_dec.n_heads());
        assert_eq!(a_dec.n_layers(), b_dec.n_layers());
    }

    /// The policy builds the same `Whisper` through its structure as through
    /// the blanket `init`.
    #[test]
    fn test_policy_pathways_agree() {
        let device = cpu_device();

        let policy = WhisperApiConfig::new(8, 16, 64, 16, 1, 12, 2)
            .with_d_head(16)
            .with_front_end(WhisperFrontEndConfig::new().with_sample_rate(8_000))
            .with_token_layout(WhisperTokenLayoutConfig::new().with_timestamp_tokens(751));

        let structure = policy.to_structure();
        assert_eq!(structure.encoder().n_heads(), 4);
        assert_eq!(structure.decoder().n_layers(), 2);

        let lowered: Whisper = structure.init(&device);
        let direct: Whisper = policy.init(&device);

        assert_meta_agrees(&direct, &lowered);
        assert_meta_agrees(&direct, &structure);
    }

    /// A structure whose encoder and decoder widths differ is an error from
    /// `try_init`, not a panic. `WhisperApiConfig` never lowers to one, but a
    /// hand-edited structure can be one.
    #[test]
    fn test_try_init_rejects_mismatched_widths() {
        let device = cpu_device();

        let mut structure = WhisperApiConfig::new(8, 16, 64, 16, 1, 12, 1)
            .with_d_head(16)
            .to_structure();
        structure.decoder = TextDecoderConfig::new(16, 32, 12, 1).with_d_head(16);

        let bad: BunsenResult<Whisper> = structure.try_init(&device);
        ErrorMatcher::kind(BunsenErrorKind::Illegal)
            .has_cause::<ConstraintError>()
            .assert_err(&bad);
    }

    /// **The dtype boundary.** A checkpoint at a precision the caller does
    /// not share must not push that precision onto the caller, and must not
    /// be widened to meet them: mels go in at the backend's float, logits
    /// come back at it, and the model computes in its own throughout.
    ///
    /// This is what lets a fp16 checkpoint be used straight off
    /// `pretrained::default_whisper_factory().load(..)`, with the mel front
    /// end left in f32.
    #[test]
    #[serial]
    fn test_dtype_is_cast_at_the_interface() {
        let device = cpu_device();

        let float = device_float_dtype(&device);
        let model: Whisper = WhisperApiConfig::new(8, 32, 64, 16, 1, 12, 1)
            .try_init(&device)
            .unwrap();
        assert_eq!(model.dtype(), float, "init builds at the backend's float");

        let half = model.map(&mut DTypeMapper::new(DType::F16));
        assert_eq!(half.dtype(), DType::F16);

        // The mel front end's output: the backend's float, not the model's.
        let mel: Tensor<3> = Tensor::random([1, 8, 16], Distribution::Default, &device);
        let tokens: Tensor<2, Int> = Tensor::zeros([1, 4], &device);
        assert_eq!(mel.dtype(), float);

        // In at the backend's float, out at it, with the model's own
        // precision in between.
        let xa = half.forward_encoder(mel.clone());
        assert_eq!(xa.dtype(), DType::F16, "features stay in the model's dtype");
        assert_eq!(
            half.forward_decoder(tokens.clone(), xa.clone()).dtype(),
            float
        );
        assert_eq!(half.forward(mel, tokens.clone()).dtype(), float);

        // Both caches are the model's, including from features a caller
        // widened behind the model's back.
        let cache = half.decoder.new_cache(xa.clone());
        assert_eq!(cache.dtype(), DType::F16, "cross-attention cache");
        let widened = half.decoder.new_cache(xa.cast(float));
        assert_eq!(widened.dtype(), DType::F16, "and from f32 features");

        let mut cache = cache;
        assert_eq!(
            half.decoder.forward_cached(tokens, &mut cache).dtype(),
            float,
        );
    }

    /// The cast is a *conversion*, not a reinterpretation: the same weights
    /// at two precisions take the same audio to the same logits.
    ///
    /// This is the failure the boundary casts exist to prevent, and it is a
    /// silent one — f32 mels fed to f16 weights do not error, they just
    /// return numbers that are wrong. Wrong by the width of the signal:
    /// when this pairing was broken during development it disagreed by ~29
    /// on logits of magnitude ~48, against the 0.013 it agrees to now.
    #[test]
    #[serial]
    fn test_half_precision_tracks_full() {
        use burn::tensor::Tolerance;

        type F = f32;

        let device = cpu_device();

        // `half` is a clone of `full`, cast to f16. Cloning a lazily
        // initialized `Param` shares its initialization, so the clone holds
        // the same weights: one model at two precisions, not two models.
        let full: Whisper = WhisperApiConfig::new(8, 32, 64, 16, 1, 12, 1)
            .try_init(&device)
            .unwrap()
            .map(&mut DTypeMapper::new(device_float_dtype(&device)));
        let half = full.clone().map(&mut DTypeMapper::new(DType::F16));

        let mel: Tensor<3> = Tensor::random([1, 8, 16], Distribution::Default, &device);
        let tokens: Tensor<2, Int> = Tensor::zeros([1, 4], &device);

        // fp16 carries a bit over three decimal digits, and this is a whole
        // encoder and decoder deep: the agreement is precision-limited.
        // Measured at 1.3e-2 absolute on a magnitude of 48; set with
        // headroom over that, and still an order of magnitude tighter than
        // any real defect.
        full.forward(mel.clone(), tokens.clone())
            .to_data_as::<F>()
            .assert_approx_eq::<F>(
                &half.forward(mel, tokens).to_data_as::<F>(),
                Tolerance::rel_abs(1e-2, 5e-2),
            );
    }

    #[test]
    #[serial]
    fn test_whisper_forward() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);

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

        let model: Whisper = structural.try_init(&device).unwrap();

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

        let mel: Tensor<3> =
            Tensor::random([batch, n_mels, audio_len], Default::default(), &device);
        let tokens: Tensor<2, Int> = Tensor::zeros([batch, token_len], &device);

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
