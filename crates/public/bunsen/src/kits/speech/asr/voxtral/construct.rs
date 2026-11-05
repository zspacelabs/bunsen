//! A loaded Voxtral Realtime model, with what it needs to transcribe.
use burn::{
    Tensor,
    module::Module,
};
use serde::{
    Deserialize,
    Serialize,
};

use super::{
    VoxtralFrontEnd,
    VoxtralFrontEndConfig,
    VoxtralFrontEndMeta,
    VoxtralGreedyDecoder,
    VoxtralTokenLayout,
    VoxtralTokenLayoutConfig,
    VoxtralTranscript,
    blocks::{
        VoxtralRealtime,
        VoxtralRealtimeMeta,
    },
};
use crate::{
    burner::module::ModuleInit,
    data::pretrained::PresetSelection,
    errors::BunsenResult,
    kits::support::tokens::TekkenVocabulary,
};

/// How a Voxtral pretrained transcribes a stream: the config its lookup's
/// patch sets (`{delay_ms: 960, window_seconds: 20}`). It has one known
/// policy, this default, so no named presets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoxtralPolicy {
    /// The longest window, in seconds, that the
    /// [`VoxtralEngine`](super::driver::VoxtralEngine) decodes at once: by
    /// default twice the audio encoder's attention window, so its sliding
    /// window is in play.
    pub window_seconds: f64,

    /// How far behind the audio the model writes, in milliseconds; the
    /// model's own default (480 ms) when `None`
    /// ([`VoxtralConstruct::with_delay_ms`]).
    pub delay_ms: Option<usize>,
}

impl Default for VoxtralPolicy {
    fn default() -> Self {
        Self {
            window_seconds: 30.0,
            delay_ms: None,
        }
    }
}

/// A Voxtral Realtime model, its front end, its vocabulary, its stream
/// layout and its decoder: what a pretrained loads to, and everything a
/// transcription needs.
///
/// Cloning shares the model's parameters.
#[derive(Clone, Debug)]
pub struct VoxtralConstruct {
    /// The model.
    pub model: VoxtralRealtime,

    /// The feature extractor the model was trained behind.
    pub front_end: VoxtralFrontEnd,

    /// The bytes the model's token ids spell.
    pub vocab: TekkenVocabulary,

    /// The stream's special tokens, delay and padding.
    pub layout: VoxtralTokenLayout,

    /// The decode.
    pub decoder: VoxtralGreedyDecoder,

    /// How it transcribes a stream: what its presets resolved to.
    pub policy: VoxtralPolicy,

    /// The presets applied, and the lookup's patch.
    pub presets: PresetSelection,
}

impl VoxtralConstruct {
    /// A construct of `model`, `vocab` and `layout`, behind the default front
    /// end and decoder.
    ///
    /// # Errors
    /// As [`VoxtralFrontEndConfig`]'s `try_init`.
    pub fn new(
        model: VoxtralRealtime,
        vocab: TekkenVocabulary,
        layout: VoxtralTokenLayout,
    ) -> BunsenResult<Self> {
        let device = model.devices()[0].clone();
        let front_end = VoxtralFrontEndConfig::new()
            .with_n_mels(model.n_mels())
            .try_init(&device)?;
        Ok(Self {
            model,
            front_end,
            vocab,
            layout,
            decoder: VoxtralGreedyDecoder::default(),
            policy: VoxtralPolicy::default(),
            presets: PresetSelection::default(),
        })
    }

    /// Takes on `policy`, and the presets that resolved it: its delay, when
    /// it names one ([`with_delay_ms`](Self::with_delay_ms)), and its window
    /// for an engine.
    ///
    /// # Errors
    /// As [`with_delay_ms`](Self::with_delay_ms).
    pub fn with_policy(
        self,
        policy: VoxtralPolicy,
        presets: PresetSelection,
    ) -> BunsenResult<Self> {
        let mut construct = match policy.delay_ms {
            Some(ms) => self.with_delay_ms(ms)?,
            None => self,
        };
        construct.policy = policy;
        construct.presets = presets;
        Ok(construct)
    }

    /// The construct under another delay: the same model, told to write
    /// `delay_ms` behind the audio.
    ///
    /// # Errors
    /// As [`VoxtralTokenLayoutConfig::validate`]: a delay the model does not
    /// take.
    pub fn with_delay_ms(
        mut self,
        delay_ms: usize,
    ) -> BunsenResult<Self> {
        let config: VoxtralTokenLayoutConfig = self.layout.config().clone().with_delay_ms(delay_ms);
        self.layout = config.try_init()?;
        Ok(self)
    }

    /// The audio one decoder position covers, in samples: 1280 (80 ms) for
    /// the published model.
    pub fn samples_per_position(&self) -> usize {
        self.front_end.hop() * self.model.mel_frames_per_position()
    }

    /// Transcribes packaged features: padded with
    /// [`silence`](VoxtralFrontEnd::silence) as [`VoxtralTokenLayout::pad`]
    /// pads samples, encoded whole, then decoded greedily.
    ///
    /// # Arguments
    /// * `features` - `[1, n_mels, frames]`, packaged by the front end.
    ///
    /// # Errors
    /// As [`VoxtralTranscript::parse`].
    pub fn transcribe_features(
        &self,
        features: Tensor<3>,
    ) -> BunsenResult<VoxtralTranscript> {
        let [frames] = crate::contracts::unpack_shape_contract!(
            [1, "n_mels", "frames"],
            &features.dims(),
            &["frames"],
            &[("n_mels", self.front_end.n_mels())]
        );
        let device = features.device();
        let per = self.model.mel_frames_per_position();
        let left = self.layout.config().left_pad_positions * per;
        let right = frames.next_multiple_of(per) - frames + self.layout.right_pad_positions() * per;
        let padded = Tensor::cat(
            vec![
                self.front_end.silence(left, &device),
                features,
                self.front_end.silence(right, &device),
            ],
            2,
        );
        let audio = self.model.audio_embeds(padded);
        let ids = self.decoder.decode(&self.model, &self.layout, audio);
        VoxtralTranscript::parse(ids, &self.layout, &self.vocab)
    }

    /// Transcribes one utterance, as `transformers`' `generate` transcribes
    /// it offline: the audio padded as [`VoxtralTokenLayout::pad`] lays it
    /// out, encoded whole, then decoded greedily, one id per position.
    ///
    /// # Arguments
    /// * `samples` - mono audio at the front end's sample rate.
    ///
    /// # Errors
    /// As the front end, and as [`VoxtralTranscript::parse`].
    pub fn transcribe(
        &self,
        samples: &[f32],
    ) -> BunsenResult<VoxtralTranscript> {
        let device = self.model.devices()[0].clone();
        let padded = self.layout.pad(samples, self.samples_per_position());
        let features = self.front_end.forward(&padded, &device)?;
        let audio = self.model.audio_embeds(features);
        let ids = self.decoder.decode(&self.model, &self.layout, audio);
        VoxtralTranscript::parse(ids, &self.layout, &self.vocab)
    }
}
