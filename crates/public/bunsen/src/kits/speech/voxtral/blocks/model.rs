//! The Voxtral Realtime model.
use burn::{
    Tensor,
    config::Config,
    module::Module,
    prelude::Int,
    tensor::Device,
};

use super::{
    VOXTRAL_STEM_STRIDE,
    VoxtralAudioEncoder,
    VoxtralAudioEncoderConfig,
    VoxtralAudioEncoderMeta,
    VoxtralProjector,
    VoxtralProjectorConfig,
    VoxtralProjectorMeta,
    VoxtralTextConfig,
    VoxtralTextMeta,
    VoxtralTextModel,
};
use crate::{
    burner::module::ModuleInit,
    errors::{
        BunsenResult,
        ConstraintError,
    },
};

/// Common meta for [`VoxtralRealtime`] and [`VoxtralRealtimeConfig`].
pub trait VoxtralRealtimeMeta {
    /// The mel channel count.
    fn n_mels(&self) -> usize;

    /// The audio encoder's width.
    fn d_audio(&self) -> usize;

    /// The text model's width.
    fn d_text(&self) -> usize;

    /// The token count.
    fn n_vocab(&self) -> usize;

    /// The encoder frames stacked into one decoder position.
    fn downsample(&self) -> usize;

    /// The mel frames one decoder position covers: the stem's stride times
    /// the stacking, 8 (80 ms) for the published model.
    fn mel_frames_per_position(&self) -> usize {
        VOXTRAL_STEM_STRIDE * self.downsample()
    }
}

/// Config for [`VoxtralRealtime`]: `transformers`' `VoxtralRealtimeConfig`.
///
/// Implements [`VoxtralRealtimeMeta`].
#[derive(Config, Debug)]
pub struct VoxtralRealtimeConfig {
    /// The audio encoder.
    pub audio: VoxtralAudioEncoderConfig,

    /// The text model.
    pub text: VoxtralTextConfig,

    /// The encoder frames stacked into one decoder position.
    #[config(default = "4")]
    pub downsample: usize,
}

impl VoxtralRealtimeConfig {
    /// `Voxtral-Mini-4B-Realtime-2602`'s.
    pub fn voxtral_mini_4b_realtime() -> Self {
        Self::new(
            VoxtralAudioEncoderConfig::voxtral_mini_4b_realtime(),
            VoxtralTextConfig::voxtral_mini_4b_realtime(),
        )
    }

    /// The projector between the two.
    pub fn projector(&self) -> VoxtralProjectorConfig {
        VoxtralProjectorConfig::new(self.audio.d_model(), self.text.d_model())
            .with_downsample(self.downsample)
    }
}

impl VoxtralRealtimeMeta for VoxtralRealtimeConfig {
    fn n_mels(&self) -> usize {
        self.audio.n_mels()
    }

    fn d_audio(&self) -> usize {
        self.audio.d_model()
    }

    fn d_text(&self) -> usize {
        self.text.d_model()
    }

    fn n_vocab(&self) -> usize {
        self.text.n_vocab()
    }

    fn downsample(&self) -> usize {
        self.downsample
    }
}

impl ModuleInit<VoxtralRealtime> for VoxtralRealtimeConfig {
    fn try_init(
        &self,
        device: &Device,
    ) -> BunsenResult<VoxtralRealtime> {
        if self.downsample == 0 {
            return Err(ConstraintError::out_of_range(
                "VoxtralRealtimeConfig",
                "downsample",
                "0",
                "at least 1",
            )
            .into());
        }
        Ok(VoxtralRealtime {
            audio_tower: self.audio.try_init(device)?,
            multi_modal_projector: self.projector().try_init(device)?,
            language_model: self.text.try_init(device)?,
        })
    }
}

/// Mistral's Voxtral Realtime (`Voxtral-Mini-4B-Realtime-2602`), streaming
/// speech to text: a causal audio encoder whose frames, four at a time, are
/// added to the token embeddings of a Mistral text model, one decoder
/// position per 80 ms of audio.
///
/// At position `i` the text model reads `embed(id_i) + audio_i`, where
/// `audio_i` is the [projected](VoxtralProjector) `i`-th group of encoder
/// frames, and predicts `id_{i+1}`: a text token, a word marker or a pad. It
/// is told how far its text runs behind the audio (the delay, in positions)
/// through its layers' adaptive scales. Everything is causal, so the model
/// streams: decoder position `i` reads only the audio up to its own group.
///
/// The fields carry the names `transformers`' `VoxtralRealtimeModel` gives
/// them; [`weights::hf_rekey`](crate::kits::speech::voxtral::weights::hf_rekey)
/// maps the checkpoint's.
///
/// Built by [`VoxtralRealtimeConfig`].
/// Implements [`VoxtralRealtimeMeta`].
#[derive(Module, Debug)]
pub struct VoxtralRealtime {
    /// The audio encoder.
    pub audio_tower: VoxtralAudioEncoder,

    /// Encoder frames to text embeddings.
    pub multi_modal_projector: VoxtralProjector,

    /// The text model.
    pub language_model: VoxtralTextModel,
}

impl VoxtralRealtimeMeta for VoxtralRealtime {
    fn n_mels(&self) -> usize {
        self.audio_tower.n_mels()
    }

    fn d_audio(&self) -> usize {
        self.audio_tower.d_model()
    }

    fn d_text(&self) -> usize {
        self.language_model.d_model()
    }

    fn n_vocab(&self) -> usize {
        self.language_model.n_vocab()
    }

    fn downsample(&self) -> usize {
        self.multi_modal_projector.downsample()
    }
}

impl VoxtralRealtime {
    /// The audio embeddings of a whole utterance: one per decoder position.
    ///
    /// # Arguments
    /// * `features` - `[batch, n_mels, frames]` log-mels, `frames` a multiple
    ///   of [`mel_frames_per_position`](VoxtralRealtimeMeta::mel_frames_per_position).
    ///
    /// # Returns
    /// `[batch, frames/8, d_text]`.
    pub fn audio_embeds(
        &self,
        features: Tensor<3>,
    ) -> Tensor<3> {
        let [batch, frames] = crate::contracts::unpack_shape_contract!(
            ["batch", "n_mels", "frames"],
            &features.dims(),
            &["batch", "frames"],
            &[("n_mels", self.n_mels())]
        );
        let y = self
            .multi_modal_projector
            .forward(self.audio_tower.forward(features));
        #[cfg(any(debug_assertions, test))]
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "positions", "d_text"],
            &y.dims(),
            &[
                ("batch", batch),
                ("positions", frames / self.mel_frames_per_position()),
                ("d_text", self.d_text())
            ]
        );
        #[cfg(not(any(debug_assertions, test)))]
        let _ = (batch, frames);
        y
    }

    /// The logits of a whole sequence, uncached: each position's token and
    /// audio embeddings summed, under a delay of `delay_tokens`.
    ///
    /// # Arguments
    /// * `ids` - `[batch, positions]`.
    /// * `audio` - `[batch, positions, d_text]`, from
    ///   [`audio_embeds`](Self::audio_embeds).
    ///
    /// # Returns
    /// `[batch, positions, n_vocab]`.
    pub fn logits(
        &self,
        ids: Tensor<2, Int>,
        audio: Tensor<3>,
        delay_tokens: usize,
    ) -> Tensor<3> {
        let [batch, positions] = crate::contracts::unpack_shape_contract!(
            ["batch", "positions"],
            &ids.dims(),
            &["batch", "positions"]
        );
        crate::contracts::assert_shape_contract_periodically!(
            ["batch", "positions", "d_text"],
            &audio.dims(),
            &[
                ("batch", batch),
                ("positions", positions),
                ("d_text", self.d_text())
            ]
        );
        let device = audio.device();
        let lm = &self.language_model;
        let embeds = lm.embed(ids) + audio;
        let hidden = lm.forward_hidden(
            embeds,
            &lm.rotary(positions, &device),
            &lm.delay_condition(delay_tokens, &device),
            None,
        );
        lm.logits(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    fn toy() -> VoxtralRealtimeConfig {
        VoxtralRealtimeConfig::new(
            VoxtralAudioEncoderConfig::new(6, 8, 2, 4, 16, 1).with_sliding_window(Some(3)),
            VoxtralTextConfig::new(30, 12, 1, 3, 1, 4, 20),
        )
        .with_downsample(2)
    }

    /// A config non-default in every field, and the model built from it,
    /// answer the meta alike, and run end to end.
    #[test]
    fn test_meta_and_shapes() {
        let device = cpu_device();
        let config = toy();
        let model: VoxtralRealtime = config.init(&device);
        assert_eq!(model.n_mels(), config.n_mels());
        assert_eq!(model.d_audio(), config.d_audio());
        assert_eq!(model.d_text(), config.d_text());
        assert_eq!(model.n_vocab(), config.n_vocab());
        assert_eq!(model.downsample(), config.downsample());
        assert_eq!(config.mel_frames_per_position(), 4);
        assert_eq!(
            VoxtralRealtimeConfig::voxtral_mini_4b_realtime().mel_frames_per_position(),
            8
        );

        let audio = model.audio_embeds(Tensor::zeros([1, 6, 12], &device));
        assert_eq!(audio.dims(), [1, 3, 12]);
        let ids = Tensor::<2, Int>::from_data([[1, 2, 3]], &device);
        assert_eq!(model.logits(ids, audio, 6).dims(), [1, 3, 30]);
    }
}
