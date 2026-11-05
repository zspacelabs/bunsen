//! What a Silero pretrained is built into: its models and its policy.

use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    data::pretrained::PresetSelection,
    errors::BunsenResult,
    kits::speech::{
        asr::driver::VoiceActivityFilterConfig,
        vad::silero_vad::{
            SileroVad,
            SileroVadCollection,
        },
    },
};

/// The policy a Silero VAD runs under: how speech regions are cut from its
/// probabilities. The config a Silero pretrained's presets patch.
///
/// Its default is the [`VoiceActivityFilterConfig`] default; a row's presets
/// name the published policies (`silero/vad-16k`'s `vad-policy` axis:
/// `[silero]`, upstream's `get_speech_timestamps` defaults, and
/// `[faster-whisper]`, `faster-whisper`'s `VadOptions`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SileroPolicy {
    /// How the filter turns probabilities into speech regions.
    pub vad: VoiceActivityFilterConfig,
}

/// A Silero pretrained, built: its sample-rate models, and the policy its
/// presets resolved to.
///
/// The Silero kit's construct, what
/// [`SileroConstructConfig`](crate::kits::speech::vad::silero_vad::pretrained::SileroConstructConfig)
/// builds. A stream picks its model by rate
/// ([`try_branch`](Self::try_branch), [`expect_branch`](Self::expect_branch))
/// and cuts regions with [`vad_filter`](Self::vad_filter).
#[derive(Debug)]
pub struct SileroConstruct {
    /// The models, one per sample rate.
    pub vads: SileroVadCollection,

    /// The policy the pretrained's presets resolved to.
    pub policy: SileroPolicy,

    /// The presets applied, and the lookup's patch.
    pub presets: PresetSelection,
}

impl SileroConstruct {
    /// `vads`, under `policy`, with no presets applied.
    pub fn new(
        vads: SileroVadCollection,
        policy: SileroPolicy,
    ) -> Self {
        Self {
            vads,
            policy,
            presets: PresetSelection::default(),
        }
    }

    /// The model for `sample_rate`: [`SileroVadCollection::try_branch`].
    ///
    /// # Errors
    /// As [`SileroVadCollection::try_branch`].
    pub fn try_branch(
        &self,
        sample_rate: usize,
    ) -> BunsenResult<&SileroVad> {
        self.vads.try_branch(sample_rate)
    }

    /// The model for `sample_rate`: [`SileroVadCollection::expect_branch`].
    ///
    /// # Panics
    /// As [`SileroVadCollection::expect_branch`].
    pub fn expect_branch(
        &self,
        sample_rate: usize,
    ) -> &SileroVad {
        self.vads.expect_branch(sample_rate)
    }

    /// The filter its policy cuts speech regions with.
    pub fn vad_filter(&self) -> &VoiceActivityFilterConfig {
        &self.policy.vad
    }
}
