//! A loaded Qwen3-ASR model, with what it needs to transcribe.
use burn::tensor::Device;

use super::{
    Qwen3AsrFrontEnd,
    Qwen3AsrFrontEndConfig,
    Qwen3AsrFrontEndMeta,
    Qwen3AsrGreedyDecoderConfig,
    Qwen3AsrPromptConfig,
    Qwen3AsrTranscript,
    blocks::{
        Qwen3Asr,
        Qwen3AsrAudioEncoderMeta,
        Qwen3AsrMeta,
    },
};
use crate::{
    burner::module::ModuleInit,
    errors::BunsenResult,
    kits::tokens::ByteLevelVocabulary,
};

/// A Qwen3-ASR model, its front end, its vocabulary, its prompt and its
/// decoder: what a pretrained loads to, and everything a transcription
/// needs.
#[derive(Debug)]
pub struct Qwen3AsrBundle {
    /// The model.
    pub model: Qwen3Asr,

    /// The feature extractor the model was trained behind.
    pub front_end: Qwen3AsrFrontEnd,

    /// The bytes the model's token ids spell.
    pub vocab: ByteLevelVocabulary,

    /// The chat template around the audio.
    pub prompt: Qwen3AsrPromptConfig,

    /// The decode's budget and end tokens.
    pub decoder: Qwen3AsrGreedyDecoderConfig,
}

impl Qwen3AsrBundle {
    /// The longest window [`transcribe_long`](Self::transcribe_long) is
    /// meant to take, in seconds: Qwen3-ASR was trained on utterances of at
    /// most 30 s.
    pub const MAX_WINDOW_SECONDS: usize = 30;

    /// A bundle of `model` and `vocab`, behind the default front end, prompt
    /// and decoder.
    ///
    /// # Errors
    /// As [`Qwen3AsrFrontEndConfig`]'s `try_init`.
    pub fn new(
        model: Qwen3Asr,
        vocab: ByteLevelVocabulary,
        device: &Device,
    ) -> BunsenResult<Self> {
        Ok(Self {
            model,
            front_end: Qwen3AsrFrontEndConfig::new().try_init(device)?,
            vocab,
            prompt: Qwen3AsrPromptConfig::new(),
            decoder: Qwen3AsrGreedyDecoderConfig::new(),
        })
    }

    /// Transcribes one utterance, greedily.
    ///
    /// # Arguments
    /// * `samples` - mono audio at the front end's
    ///   [`sample_rate`](Qwen3AsrFrontEndMeta::sample_rate); at most
    ///   [`MAX_WINDOW_SECONDS`](Self::MAX_WINDOW_SECONDS) is the model's range.
    ///
    /// # Errors
    /// As [`Qwen3AsrFrontEnd::forward`]; [`InvalidResource`] if a decoded
    /// token is outside the vocabulary.
    ///
    /// [`InvalidResource`]: crate::errors::BunsenErrorKind::InvalidResource
    pub fn transcribe(
        &self,
        samples: &[f32],
        device: &Device,
    ) -> BunsenResult<Qwen3AsrTranscript> {
        let (features, valid) = self.front_end.forward(samples, device)?;
        let audio = self.model.encode_audio(features, valid);
        let n_audio = self.model.audio_tower.output_len(valid);
        let ids = self.prompt.prompt(n_audio, self.model.audio_token());
        let embeds = self.model.prompt_embeds(&ids, audio)?;
        let tokens = self.decoder.init().decode(&self.model, embeds);
        self.prompt.parse(tokens, &self.decoder.eos, &self.vocab)
    }

    /// Transcribes one waveform of any length, in windows of at most
    /// `max_window` samples.
    ///
    /// A recording longer than the model's range is cut into windows, each
    /// cut at the quietest mel frame of the window's last sixth
    /// ([`quiet_cuts`](crate::ops::signal::quiet_cuts)), and each window is
    /// transcribed alone. The texts are joined with spaces; the language is
    /// the first window's that names one.
    ///
    /// # Errors
    /// As [`transcribe`](Self::transcribe).
    ///
    /// # Panics
    /// If `max_window` is shorter than six mel frames.
    pub fn transcribe_long(
        &self,
        samples: &[f32],
        max_window: usize,
        device: &Device,
    ) -> BunsenResult<Qwen3AsrTranscript> {
        let windows = crate::ops::signal::quiet_cuts(samples, max_window, self.front_end.hop());
        let mut joined = Qwen3AsrTranscript {
            tokens: Vec::new(),
            language: None,
            text: String::new(),
        };
        let mut texts = Vec::new();
        for (a, b) in windows {
            let t = self.transcribe(&samples[a..b], device)?;
            joined.tokens.extend(t.tokens);
            joined.language = joined.language.or(t.language);
            if !t.text.is_empty() {
                texts.push(t.text);
            }
        }
        joined.text = texts.join(" ");
        Ok(joined)
    }

    /// [`MAX_WINDOW_SECONDS`](Self::MAX_WINDOW_SECONDS), in samples.
    pub fn max_window(&self) -> usize {
        Self::MAX_WINDOW_SECONDS * self.front_end.sample_rate()
    }
}
