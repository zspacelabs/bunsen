//! What a TTS driver is asked to speak.

use alloc::string::String;

/// Which voice a request speaks in.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TtsVoiceChoice {
    /// The engine's default voice
    /// ([`TtsCapabilities::default_voice`](super::TtsCapabilities::default_voice)).
    Default,

    /// A stock voice, by name
    /// ([`TtsEngine::voices`](super::TtsEngine::voices)).
    Stock(String),

    /// A voice described in words, for an engine that designs one
    /// ([`TtsCapabilities::described_voices`](super::TtsCapabilities::described_voices)).
    Described(String),

    /// A voice built for this engine's build
    /// ([`TtsEngine::build_voice`](super::TtsEngine::build_voice)).
    Print(super::TtsVoicePrint),
}

/// Sampling settings over an engine's release defaults: each `None` keeps
/// the engine's. An engine applies them to the sampler that chooses its
/// first codes, and says how in its docs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TtsSamplingOverrides {
    /// The temperature; 0 is greedy.
    pub temperature: Option<f64>,

    /// The most probable codes kept.
    pub top_k: Option<usize>,

    /// The probability mass kept.
    pub top_p: Option<f64>,

    /// The least probability kept, relative to the most probable code's.
    pub min_p: Option<f64>,

    /// The repetition penalty.
    pub repetition_penalty: Option<f64>,
}

/// A request to a [`TtsDriver`](super::TtsDriver): a text to speak, and how.
#[derive(Clone, Debug, PartialEq)]
pub struct TtsRequest {
    /// The text, as typed: the driver's request stages normalize it.
    pub text: String,

    /// The voice.
    pub voice: TtsVoiceChoice,

    /// The language, as one of the engine's own keys
    /// ([`TtsCapabilities::languages`](super::TtsCapabilities::languages));
    /// `None`, the engine's default for the voice.
    pub language: Option<String>,

    /// A style instruction, for an engine that takes one
    /// ([`TtsCapabilities::instruction`](super::TtsCapabilities::instruction)).
    /// Given to one that does not, it is refused unless the driver strips it
    /// ([`TtsDriverConfig::strip_instruction`](super::TtsDriverConfig::strip_instruction)).
    pub instruction: Option<String>,

    /// Seeds the take's random numbers; `None` leaves them as they are.
    pub seed: Option<u64>,

    /// The longest take, in seconds; `None`, as long as the engine allows.
    pub max_seconds: Option<f64>,

    /// Sampling settings over the engine's.
    pub sampling: TtsSamplingOverrides,
}

impl TtsRequest {
    /// `text` in the default voice and language, unseeded, at the engine's
    /// settings.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            voice: TtsVoiceChoice::Default,
            language: None,
            instruction: None,
            seed: None,
            max_seconds: None,
            sampling: TtsSamplingOverrides::default(),
        }
    }

    /// In `voice`.
    pub fn with_voice(
        mut self,
        voice: TtsVoiceChoice,
    ) -> Self {
        self.voice = voice;
        self
    }

    /// In the engine's `language` key.
    pub fn with_language(
        mut self,
        language: impl Into<String>,
    ) -> Self {
        self.language = Some(language.into());
        self
    }

    /// With a style instruction.
    pub fn with_instruction(
        mut self,
        instruction: impl Into<String>,
    ) -> Self {
        self.instruction = Some(instruction.into());
        self
    }

    /// Seeded.
    pub fn with_seed(
        mut self,
        seed: u64,
    ) -> Self {
        self.seed = Some(seed);
        self
    }

    /// At most `seconds` long.
    pub fn with_max_seconds(
        mut self,
        seconds: f64,
    ) -> Self {
        self.max_seconds = Some(seconds);
        self
    }

    /// With `sampling` over the engine's settings.
    pub fn with_sampling(
        mut self,
        sampling: TtsSamplingOverrides,
    ) -> Self {
        self.sampling = sampling;
        self
    }
}

/// One take, as the driver hands it to an engine: the request's text after
/// its stages, and the voice resolved.
#[derive(Debug)]
pub struct TtsTakeRequest<'a, V> {
    /// The text, normalized.
    pub text: &'a str,

    /// The voice, as the engine resolved it.
    pub voice: &'a V,

    /// The language key; `None`, the engine's default for the voice.
    pub language: Option<&'a str>,

    /// The style instruction, when the engine takes one and it was given.
    pub instruction: Option<&'a str>,

    /// Seeds the take.
    pub seed: Option<u64>,

    /// The longest take, in seconds.
    pub max_seconds: Option<f64>,

    /// Sampling settings over the engine's.
    pub sampling: &'a TtsSamplingOverrides,
}
