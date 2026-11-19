//! The engine seam: what a model provides a TTS driver.

use alloc::{
    string::String,
    vec::Vec,
};
use core::fmt::Debug;

use super::{
    TtsTakeRequest,
    TtsVoiceChoice,
};
use crate::{
    errors::BunsenResult,
    kits::support::text::BunsenLanguageTag,
};

/// One language an engine speaks.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsLanguage {
    /// The engine's own key (`en_us`, `sichuan_dialect`), as a request
    /// names it.
    pub key: String,

    /// The well-known tag it stands for, which a text normalizer reads.
    pub tag: BunsenLanguageTag,
}

/// What an engine can do: what a request may ask of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TtsCapabilities {
    /// The languages, by the engine's keys.
    pub languages: Vec<TtsLanguage>,

    /// The voice a request speaks in when it names none.
    pub default_voice: Option<String>,

    /// Whether a request may name a stock voice.
    pub stock_voices: bool,

    /// Whether a request may describe a voice in words.
    pub described_voices: bool,

    /// Whether a request may carry a style instruction.
    pub instruction: bool,
}

/// What kind of voice a [`TtsVoiceInfo`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TtsVoiceKind {
    /// One the release ships.
    Stock,
}

/// What an engine says of one of its voices.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsVoiceInfo {
    /// Its name, as a request names it.
    pub name: String,

    /// What kind it is.
    pub kind: TtsVoiceKind,

    /// Its native language, as an engine key.
    pub default_language: Option<String>,

    /// Its gender, as the release says it.
    pub gender: Option<String>,

    /// Its style, as the release says it.
    pub style: Option<String>,

    /// What the release says of it otherwise.
    pub description: Option<String>,
}

impl TtsVoiceInfo {
    /// A stock voice named `name`, of which nothing else is known.
    pub fn stock(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: TtsVoiceKind::Stock,
            default_language: None,
            gender: None,
            style: None,
            description: None,
        }
    }
}

/// Why a take ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtsStop {
    /// The model ended it.
    End,

    /// It ran out of frames: the budget, the request's
    /// [`max_seconds`](super::TtsRequest::max_seconds) or the model's
    /// context.
    Budget,
}

/// What a take was.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsTakeInfo {
    /// The frames of codes it rendered.
    pub frames: usize,

    /// The model's generation steps.
    pub steps: usize,

    /// Why it ended.
    pub stop: TtsStop,
}

/// A text-to-speech model, as a [`TtsDriver`](super::TtsDriver) runs it.
///
/// The driver owns what does not depend on the model: the request's stages
/// (text normalization, instruction stripping), voice and language checks,
/// and the result. The engine owns what does: its prompt, what a voice is to
/// it, the generation loop and its stop rule, its samplers, and the codec
/// render.
pub trait TtsEngine: Debug + Send + Sync {
    /// What a voice choice resolves to for this model.
    type Voice: Clone + Debug + Send + Sync;

    /// A take's codes, as the engine's render reads them.
    type Take: Debug + Send;

    /// What a request may ask of it.
    fn capabilities(&self) -> &TtsCapabilities;

    /// The output's sample rate, in Hz.
    fn sample_rate(&self) -> u32;

    /// Its stock voices.
    fn voices(&self) -> Vec<TtsVoiceInfo>;

    /// The most characters one take can speak, when the model bounds it
    /// below the driver's
    /// [`max_take_chars`](super::TtsDriverConfig::max_take_chars); `None`,
    /// no bound of its own.
    fn max_take_chars(&self) -> Option<usize> {
        None
    }

    /// `choice` as this model's voice.
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) for a stock voice
    /// it does not have;
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for a kind
    /// of voice it cannot speak in.
    fn resolve_voice(
        &self,
        choice: &TtsVoiceChoice,
    ) -> BunsenResult<Self::Voice>;

    /// The tag a request's language stands for in `voice`, which text
    /// normalizers read by; `None` for none known (a language the voice
    /// chooses, with no tag of its own).
    ///
    /// # Errors
    /// [`Lookup`](crate::errors::BunsenErrorKind::Lookup) or
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a key that
    /// is not one of its languages.
    fn language_tag(
        &self,
        language: Option<&str>,
        voice: &Self::Voice,
    ) -> BunsenResult<Option<BunsenLanguageTag>>;

    /// One take's codes.
    ///
    /// # Errors
    /// The engine's, building its prompt or generating.
    fn generate(
        &self,
        take: &TtsTakeRequest<'_, Self::Voice>,
    ) -> BunsenResult<Self::Take>;

    /// A take's samples, mono, in `[-1, 1]`, at
    /// [`sample_rate`](Self::sample_rate).
    ///
    /// # Errors
    /// The engine's, decoding.
    fn render(
        &self,
        take: &Self::Take,
    ) -> BunsenResult<Vec<f32>>;

    /// What a take was.
    fn take_info(
        &self,
        take: &Self::Take,
    ) -> TtsTakeInfo;
}
