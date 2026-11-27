//! The engine seam: what a model provides a TTS driver.

use alloc::{
    boxed::Box,
    string::String,
    vec::Vec,
};
use core::fmt::Debug;

use super::{
    TtsChunkPolicy,
    TtsTakeRequest,
    TtsTakeStream,
    TtsVoiceChoice,
    TtsVoicePrint,
    TtsVoiceReference,
    stream::TtsWholeTakeStream,
};
use crate::{
    errors::{
        BunsenError,
        BunsenResult,
        LookupError,
    },
    kits::{
        speech::tts::TtsStop,
        support::text::BunsenLanguageTag,
    },
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

    /// The modes it clones a voice in, its default first; none, it clones
    /// none.
    pub clone_modes: Vec<TtsCloneMode>,
}

impl TtsCapabilities {
    /// The mode to build a voice from `reference` in: `asked`, when it names
    /// one; else the first that `reference` satisfies.
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported) for an
    /// engine that clones none; [`Lookup`](crate::errors::BunsenErrorKind::Lookup)
    /// for a mode it does not have; [`Illegal`](crate::errors::BunsenErrorKind::Illegal)
    /// for one `reference` does not satisfy, or none it does, naming the need.
    pub fn clone_mode(
        &self,
        reference: &TtsVoiceReference,
        asked: Option<&str>,
    ) -> BunsenResult<&TtsCloneMode> {
        if self.clone_modes.is_empty() {
            return Err(BunsenError::unsupported("this model clones no voices"));
        }
        match asked {
            Some(name) => {
                let mode = self
                    .clone_modes
                    .iter()
                    .find(|m| m.name == name)
                    .ok_or_else(|| {
                        BunsenError::lookup(
                            LookupError::missing("clone mode", name)
                                .with_candidates(self.clone_modes.iter().map(|m| &m.name)),
                        )
                    })?;
                mode.check(reference)?;
                Ok(mode)
            }
            None => self
                .clone_modes
                .iter()
                .find(|m| m.check(reference).is_ok())
                .ok_or_else(|| {
                    let first = &self.clone_modes[0];
                    first.check(reference).unwrap_err()
                }),
        }
    }
}

/// A mode an engine clones a voice in, and what it needs of a reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TtsCloneMode {
    /// Its name (`embedding`, `xvector`, `icl`).
    pub name: String,

    /// Whether it needs the reference's transcript.
    pub needs_transcript: bool,
}

impl TtsCloneMode {
    /// A mode named `name` that needs a clip alone.
    pub fn clip(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            needs_transcript: false,
        }
    }

    /// Refuses a reference this mode cannot build a voice from.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) for a reference of
    /// no clips, or of no transcript where it needs one, naming the mode.
    pub fn check(
        &self,
        reference: &TtsVoiceReference,
    ) -> BunsenResult<()> {
        if reference.clips.is_empty() {
            return Err(BunsenError::illegal(format!(
                "a voice reference of no clips, for clone mode {}",
                self.name
            )));
        }
        if self.needs_transcript && reference.transcript().is_none_or(str::is_empty) {
            return Err(BunsenError::illegal(format!(
                "clone mode {} needs the reference's transcript: give one, or transcribe the clips",
                self.name
            )));
        }
        Ok(())
    }
}

/// What kind of voice a [`TtsVoiceInfo`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TtsVoiceKind {
    /// One the release ships.
    Stock,

    /// One built from a voice reference ([`TtsVoicePrint`]).
    User,
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

/// A take's codec audio as host samples: `audio`, any shape (`[.., N]`),
/// read back as f32.
///
/// # Errors
/// [`Internal`](crate::errors::BunsenErrorKind::Internal) when it does not
/// read back, or is not finite.
pub fn tts_samples<const D: usize>(audio: burn::Tensor<D>) -> BunsenResult<Vec<f32>> {
    let samples: Vec<f32> = audio
        .cast(burn::tensor::DType::F32)
        .into_data()
        .try_into_vec_as::<f32>()
        .map_err(|e| crate::errors::BunsenError::internal(format!("the codec's audio: {e:?}")))?;
    if samples.iter().any(|x| !x.is_finite()) {
        return Err(crate::errors::BunsenError::internal(
            "the codec's audio is not finite",
        ));
    }
    Ok(samples)
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

    /// One take as a stream ([`TtsTakeStream`]): generated a step at a time,
    /// its final frames decoded as `chunks` groups them, its samples
    /// returned as they become final. Joined, they are
    /// [`render`](Self::render)'s of [`generate`](Self::generate)'s take.
    /// The stream copies what it needs of `take`.
    ///
    /// The default generates the take whole here and renders it whole at
    /// the stream's one step: a correct stream, not an early one.
    ///
    /// # Errors
    /// As [`generate`](Self::generate) (the default's), or the engine's
    /// own, starting the take.
    fn start_stream<'a>(
        &'a self,
        take: &TtsTakeRequest<'_, Self::Voice>,
        chunks: TtsChunkPolicy,
    ) -> BunsenResult<Box<dyn TtsTakeStream + 'a>>
    where
        Self: Sized,
    {
        let _ = chunks;
        Ok(Box::new(TtsWholeTakeStream {
            engine: self,
            take: Some(self.generate(take)?),
        }))
    }

    /// The voice built from `reference` for this engine's build, in `mode` (the
    /// [default](TtsCapabilities::clone_mode) when `None`): its clips
    /// prepared for the encoder, its transcript tokenized where the mode
    /// reads one, encoded, to a voice of the engine's own tensors.
    ///
    /// The default clones none.
    ///
    /// # Errors
    /// As [`TtsCapabilities::clone_mode`]; the encoder's.
    fn build_voice(
        &self,
        reference: &TtsVoiceReference,
        mode: Option<&str>,
    ) -> BunsenResult<TtsVoicePrint> {
        self.capabilities().clone_mode(reference, mode)?;
        Err(BunsenError::unsupported("this model clones no voices"))
    }
}
