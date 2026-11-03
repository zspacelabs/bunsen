//! # How a Whisper pretrained decodes: the config its presets patch

use serde::{
    Deserialize,
    Serialize,
};

use crate::kits::speech::whisper::driver::WhisperTask;

/// How a Whisper pretrained decodes: what its presets, and a lookup's patch,
/// set; a [`WhisperConstruct`](super::WhisperConstruct) carries the one they
/// resolved to, and
/// [`WhisperStreamDriverConfig::with_policy`](super::WhisperStreamDriverConfig::with_policy)
/// applies it to a driver.
///
/// The multilingual rows have a `task` axis (`[transcribe]`, the default,
/// and `[translate]`); every other field is set by a patch:
/// `'openai/small[translate]{beam_size: 1}'`.
///
/// The defaults are `openai-whisper`'s `transcribe` command's: a beam of 5,
/// up to 224 tokens a window, each window prompted with the transcript so
/// far, and upstream's temperature ladder when a window's decode fails its
/// thresholds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WhisperPolicy {
    /// Transcribe, or translate to English.
    pub task: WhisperTask,

    /// A Whisper language code (`en`, `de`, ...); detected from the first
    /// window when `None`.
    pub language: Option<String>,

    /// Emit timestamp tokens and split segments on them.
    pub timestamps: bool,

    /// Beams in the search; 1 is greedy.
    pub beam_size: usize,

    /// The most tokens a window generates.
    pub max_tokens: usize,

    /// Prompt each window with the transcript so far.
    pub condition_on_previous_text: bool,

    /// Climb upstream's temperature ladder when a window's decode fails its
    /// thresholds; temperature zero alone otherwise.
    pub fallback: bool,
}

impl Default for WhisperPolicy {
    fn default() -> Self {
        Self {
            task: WhisperTask::Transcribe,
            language: None,
            timestamps: false,
            beam_size: 5,
            max_tokens: 224,
            condition_on_previous_text: true,
            fallback: true,
        }
    }
}
