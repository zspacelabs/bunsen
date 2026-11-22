//! [`Zonos3Engine`] as a [`TtsEngine`].

use alloc::{
    collections::BTreeMap,
    format,
    string::ToString,
    vec,
    vec::Vec,
};

use burn::tensor::DType;

use super::{
    ZONOS3_EMBEDDING_MODE,
    ZONOS3_EMBEDDING_TENSOR,
    ZONOS3_VOICE_FAMILY,
    Zonos3Codes,
    Zonos3Engine,
    Zonos3Request,
    Zonos3Stop,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::{
        speech::tts::{
            driver::{
                TtsCapabilities,
                TtsEngine,
                TtsSamplingOverrides,
                TtsStop,
                TtsTakeInfo,
                TtsTakeRequest,
                TtsVoiceChoice,
                TtsVoiceIdentity,
                TtsVoiceInfo,
                TtsVoicePrint,
                TtsVoiceReference,
                TtsVoiceTensor,
            },
            zonos3::{
                Zonos3SpeakerPart,
                prompt::Zonos3Speaker,
                voices::Zonos3Voice,
            },
        },
        support::{
            text::BunsenLanguageTag,
            tokens::Tokenizer,
        },
    },
};

/// What the driver says of ZONOS3's stock voices: each one's gender and
/// style, and its language as the release names it (`English`), as its
/// description.
pub fn zonos3_voice_list(voices: &BTreeMap<String, Zonos3Voice>) -> Vec<TtsVoiceInfo> {
    voices
        .iter()
        .map(|(name, voice)| {
            voice
                .info
                .clone()
                .unwrap_or_else(|| TtsVoiceInfo::stock(name.clone()))
        })
        .collect()
}

impl<T: Tokenizer + Send + Sync> Zonos3Engine<T> {
    /// The speaker encoder.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a
    /// release without one.
    pub fn speaker_part(&self) -> BunsenResult<&Zonos3SpeakerPart> {
        self.construct()
            .speaker
            .as_ref()
            .ok_or_else(|| BunsenError::unsupported("this ZONOS3 release ships no speaker encoder"))
    }

    /// A voice file as a speaker: its embedding, when this build made it.
    ///
    /// # Errors
    /// As [`speaker_part`](Self::speaker_part);
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a voice of
    /// another build or mode;
    /// [`InvalidResource`](crate::errors::BunsenErrorKind::InvalidResource)
    /// for one without an `[S]` f32 embedding.
    pub fn voice_speaker(
        &self,
        voice: &TtsVoicePrint,
    ) -> BunsenResult<Zonos3Speaker> {
        let part = self.speaker_part()?;
        voice.identity.check(ZONOS3_VOICE_FAMILY, &part.build)?;
        if voice.identity.mode != ZONOS3_EMBEDDING_MODE {
            return Err(BunsenError::illegal(format!(
                "a ZONOS3 voice of mode {}; this speaks in {ZONOS3_EMBEDDING_MODE} voices",
                voice.identity.mode
            )));
        }
        let width = part.config.embed_dim;
        Ok(Zonos3Speaker::Embedding(
            voice.f32_vector(ZONOS3_EMBEDDING_TENSOR, width)?.to_vec(),
        ))
    }
}

/// ZONOS3 through the TTS driver.
///
/// - A voice is a stock voice, spoken in the config's
///   [`speaker_mode`](super::Zonos3EngineConfig::speaker_mode): its reference
///   codes (the release's default) or its speaker embedding; the default voice
///   is the config's
///   [`default_voice`](super::Zonos3EngineConfig::default_voice).
/// - A language is a request code of the release's `languages` table, or a
///   language tag; none is the config's
///   [`default_language`](super::Zonos3EngineConfig::default_language).
/// - A voice file this build made ([`TtsVoiceChoice::Print`]) speaks through
///   its embedding; [`build_voice`](TtsEngine::build_voice) makes one from a
///   reference, in clone mode `embedding`, when the release ships a speaker
///   encoder.
/// - Sampling overrides apply to every codebook's sampler.
/// - [`max_seconds`](TtsTakeRequest::max_seconds) bounds the take's rows, at
///   the codec's frame rate.
impl<T: Tokenizer + Send + Sync> TtsEngine for Zonos3Engine<T> {
    type Take = Zonos3Codes;
    type Voice = Zonos3Speaker;

    fn capabilities(&self) -> &TtsCapabilities {
        &self.capabilities
    }

    fn sample_rate(&self) -> u32 {
        self.construct().sample_rate
    }

    fn voices(&self) -> Vec<TtsVoiceInfo> {
        zonos3_voice_list(self.voices())
    }

    fn resolve_voice(
        &self,
        choice: &TtsVoiceChoice,
    ) -> BunsenResult<Zonos3Speaker> {
        match choice {
            TtsVoiceChoice::Default => self.stock_speaker(&self.config().default_voice),
            TtsVoiceChoice::Stock(name) => self.stock_speaker(name),
            TtsVoiceChoice::Print(voice) => self.voice_speaker(voice),
            other => Err(BunsenError::unsupported(format!(
                "ZONOS3 speaks in its stock voices and in voices cloned for it; not {other:?}"
            ))),
        }
    }

    fn language_tag(
        &self,
        language: Option<&str>,
        _voice: &Zonos3Speaker,
    ) -> BunsenResult<Option<BunsenLanguageTag>> {
        let key = language.unwrap_or(&self.config().default_language);
        Ok(Some(self.construct().language_tag(key)?))
    }

    fn generate(
        &self,
        take: &TtsTakeRequest<'_, Zonos3Speaker>,
    ) -> BunsenResult<Zonos3Codes> {
        let mut request = Zonos3Request::new(take.text.to_string(), take.voice.clone());
        request.condition_buckets = self.condition_buckets.clone();
        request.seed = take.seed;
        if let Some(seconds) = take.max_seconds {
            let rate = f64::from(self.sample_rate()) / self.construct().codec_config.hop as f64;
            request.max_frames = Some((seconds * rate).ceil().max(1.0) as usize);
        }
        if *take.sampling == TtsSamplingOverrides::default() {
            self.generate_codes(&request)
        } else {
            self.generate_codes_with(&request, take.sampling)
        }
    }

    fn render(
        &self,
        take: &Zonos3Codes,
    ) -> BunsenResult<Vec<f32>> {
        Zonos3Engine::render(self, take)
    }

    /// The embedding voice built from `reference`: its clips, each mixed to
    /// mono and prepared for the encoder (resampled to its rate, scaled),
    /// joined in order and cut to its longest clip, through the speaker
    /// encoder to a raw `[S]` embedding, in a voice file of this build.
    fn build_voice(
        &self,
        reference: &TtsVoiceReference,
        mode: Option<&str>,
    ) -> BunsenResult<TtsVoicePrint> {
        self.capabilities.clone_mode(reference, mode)?;
        let part = self.speaker_part()?;
        let encoder = &part.encoder;
        let mut prepared = Vec::new();
        for clip in &reference.clips {
            prepared.extend(encoder.prepare(&clip.mono(), clip.sample_rate() as usize));
        }
        prepared.truncate(encoder.max_samples);
        let embedding: Vec<f32> = encoder
            .forward(encoder.features(&prepared)?)
            .cast(DType::F32)
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|e| BunsenError::invalid_resource(format!("the embedding: {e:?}")))?;
        Ok(TtsVoicePrint {
            identity: TtsVoiceIdentity {
                family: ZONOS3_VOICE_FAMILY.into(),
                encoder: part.build.clone(),
                mode: ZONOS3_EMBEDDING_MODE.into(),
                reference: reference.digest.clone(),
            },
            tensors: BTreeMap::from([(
                ZONOS3_EMBEDDING_TENSOR.to_string(),
                TtsVoiceTensor::F32 {
                    shape: vec![embedding.len()],
                    values: embedding,
                },
            )]),
        })
    }

    fn take_info(
        &self,
        take: &Zonos3Codes,
    ) -> TtsTakeInfo {
        TtsTakeInfo {
            frames: take.frames.len(),
            steps: take.rows,
            stop: match take.stop {
                Zonos3Stop::EndOfAudio => TtsStop::End,
                Zonos3Stop::Budget => TtsStop::Budget,
            },
        }
    }
}
