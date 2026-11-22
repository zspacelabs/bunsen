//! [`Zonos3Engine`] as a [`TtsEngine`].

use alloc::{
    collections::BTreeMap,
    format,
    string::ToString,
    vec::Vec,
};

use super::{
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
                TtsVoiceInfo,
            },
            zonos3::{
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
            other => Err(BunsenError::unsupported(format!(
                "ZONOS3 speaks in its stock voices; not {other:?}"
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
