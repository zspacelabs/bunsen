//! [`Zonos3Engine`] as a [`TtsEngine`].

use alloc::{
    boxed::Box,
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
    Zonos3TakeRun,
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
                TtsChunkPolicy,
                TtsEngine,
                TtsSamplingOverrides,
                TtsStreamStep,
                TtsTakeInfo,
                TtsTakeRequest,
                TtsTakeStream,
                TtsVoiceChoice,
                TtsVoiceIdentity,
                TtsVoiceInfo,
                TtsVoicePrint,
                TtsVoiceReference,
                TtsVoiceTensor,
            },
            zonos3::{
                Zonos3SpeakerPart,
                blocks::Zonos3CodecStreamContext,
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
        let request = self.take_request(take);
        if *take.sampling == TtsSamplingOverrides::default() {
            self.generate_codes(&request)
        } else {
            self.generate_codes_with(&request, take.sampling)
        }
    }

    fn start_stream<'a>(
        &'a self,
        take: &TtsTakeRequest<'_, Zonos3Speaker>,
        chunks: TtsChunkPolicy,
    ) -> BunsenResult<Box<dyn TtsTakeStream + 'a>> {
        let request = self.take_request(take);
        let run = self.start_take(&request, self.try_sampler_with(take.sampling)?)?;
        let p = run.silence.len();
        Ok(Box::new(Zonos3TakeStream {
            engine: self,
            codec: self.construct().codec.new_stream_context(1, &self.device()),
            next: p - self.config().overlap_frames.min(p),
            start: p - self.config().overlap_frames.min(p),
            covered: false,
            pending: Vec::new(),
            decoded: 0,
            chunks,
            run,
        }))
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
            stop: take.stop,
        }
    }
}

impl<T: Tokenizer + Send + Sync> Zonos3Engine<T> {
    /// A take's request as the engine generates it: its text and voice, the
    /// engine's condition buckets, its seed, and its length in frames.
    fn take_request(
        &self,
        take: &TtsTakeRequest<'_, Zonos3Speaker>,
    ) -> Zonos3Request {
        let mut request = Zonos3Request::new(take.text.to_string(), take.voice.clone());
        request.condition_buckets = self.condition_buckets.clone();
        request.seed = take.seed;
        let rate = f64::from(self.sample_rate()) / self.construct().codec_config.hop as f64;
        if let Some(frames) = take.max_frames(rate) {
            request.max_frames = Some(frames);
        }
        request
    }
}

/// A ZONOS3 take as a stream ([`TtsEngine::start_stream`]): a row a step.
/// A frame is final once its last codebook's row is sampled and no end of
/// audio sampled yet precedes it
/// ([`final_end`](super::Zonos3TakeRows::final_end)), `n - 1` rows behind
/// the model; the final frames, from the whole decode's start in the silence
/// prefill, go through the codec's stream as the chunk policy groups them,
/// and through the output stage.
struct Zonos3TakeStream<'a, T: Tokenizer + Send + Sync> {
    engine: &'a Zonos3Engine<T>,
    run: Zonos3TakeRun,
    codec: Zonos3CodecStreamContext,
    /// The whole decode's first frame, in the stream's frame indices.
    start: usize,
    /// The next frame to take as final.
    next: usize,
    /// Whether a generated frame has been final yet.
    covered: bool,
    /// The final frames not yet decoded.
    pending: Vec<Vec<u32>>,
    /// The frames decoded.
    decoded: usize,
    chunks: TtsChunkPolicy,
}

impl<T: Tokenizer + Send + Sync> TtsTakeStream for Zonos3TakeStream<'_, T> {
    fn step(&mut self) -> BunsenResult<TtsStreamStep> {
        let engine = self.engine;
        let stop = engine.step_take(&mut self.run);
        if let Some(end) = self.run.take.final_end(self.run.silence.len()) {
            self.covered = true;
            for f in self.next..end {
                self.pending.push(self.run.take.frame(&self.run.silence, f));
            }
            self.next = self.next.max(end);
        }
        let codec = &engine.construct().codec;
        let mut parts = Vec::new();
        if self
            .chunks
            .decode_now(self.pending.len(), self.decoded, stop.is_some())
        {
            let ids = engine.frame_ids(&self.pending);
            self.decoded += self.pending.len();
            self.pending.clear();
            parts.extend(codec.forward_stream(ids, &mut self.codec));
        }
        if stop.is_some() && self.decoded > 0 {
            parts.extend(codec.finish_stream(&mut self.codec));
        }
        let samples = if parts.is_empty() {
            Vec::new()
        } else {
            engine.output_samples(burn::Tensor::cat(parts, 1))?
        };
        Ok(TtsStreamStep {
            samples,
            end: stop.map(|stop| TtsTakeInfo {
                frames: if self.covered {
                    self.next - self.start
                } else {
                    0
                },
                steps: self.run.take.rows().len(),
                stop,
            }),
        })
    }
}
