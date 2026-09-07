//! # Emission: what a stream says, and when.
//!
//! Two axes and two output variants. **A `window_full` or `endpoint` decode
//! commits, per the commit rule; an `interval` decode drafts.** Everything the
//! three deployment targets differ by falls out of those two sentences, and
//! the three presets on [`EmissionPolicy`] are those targets.
//!
//! A [`Draft`](TranscriptEvent::Draft) always covers *all* audio after the last
//! commit and supersedes the previous draft entirely, so there is no
//! retraction protocol, no sequence numbers, and no way to hold two drafts at
//! once. Under [`offline`](EmissionPolicy::offline) and
//! [`conservative`](EmissionPolicy::conservative) the variant is never
//! constructed.

use std::time::Duration;

use burn::config::Config;

/// Enum of common [`EmissionPolicy`] presets.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    strum::EnumString,
    strum::Display,
)]
pub enum PresetEmissionPolicy {
    /// Decode each full window and commit all of it.
    Offline,

    /// Decode at the end of each speech region as well; every emission is
    /// final. Needs the bundled VAD.
    Conservative,

    /// Conservative, plus a draft every 600 ms of speech.
    /// Needs the bundled VAD.
    Responsive,
}

impl From<PresetEmissionPolicy> for EmissionPolicy {
    fn from(policy: PresetEmissionPolicy) -> Self {
        match policy {
            PresetEmissionPolicy::Offline => EmissionPolicy::offline(),
            PresetEmissionPolicy::Conservative => EmissionPolicy::conservative(),
            PresetEmissionPolicy::Responsive => EmissionPolicy::responsive(),
        }
    }
}

/// When a decode is run.
#[derive(Config, Debug, PartialEq, Eq)]
pub struct DecodeTriggers {
    /// Decode when a full window of audio has accumulated past the seek
    /// pointer. The only trigger that needs no voice activity.
    #[config(default = "true")]
    pub window_full: bool,

    /// Decode when the voice-activity gate closes a region.
    #[config(default = "false")]
    pub endpoint: bool,

    /// Decode every so often while speech is in progress, as a draft.
    #[config(default = "None")]
    pub interval: Option<Duration>,
}

/// When a decode's output becomes final.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum CommitRule {
    /// Everything decoded is committed. Right when a window is the last
    /// thing that will ever be said about its audio.
    Complete,

    /// Commit up to the last timestamp the decode emitted, and carry the
    /// rest forward; the seek pointer advances to that timestamp.
    LastTimestamp,

    /// Commit a prefix once `runs` consecutive decodes agree on it. The one
    /// rule under which a provisional decode becomes load-bearing.
    Agreement {
        /// Consecutive decodes that must agree.
        runs: usize,
    },
}

/// The emission policy: triggers and a commit rule.
///
/// Held by the driver, configured, never forked. The presets are the three
/// deployment targets; anything else is a custom pairing.
#[derive(Config, Debug, PartialEq, Eq)]
pub struct EmissionPolicy {
    /// When to decode.
    pub triggers: DecodeTriggers,

    /// When decoded output is final.
    pub commit: CommitRule,
}

impl EmissionPolicy {
    /// Server-batch offline inference: decode each full window, commit all
    /// of it. Nothing is emitted until a window fills or the stream is
    /// flushed.
    pub fn offline() -> Self {
        Self {
            triggers: DecodeTriggers::new(),
            commit: CommitRule::Complete,
        }
    }

    /// Conservative real time, for a programmatic consumer: decode at the
    /// end of each speech region as well, commit up to the last timestamp.
    /// Every emission is final.
    pub fn conservative() -> Self {
        Self {
            triggers: DecodeTriggers::new().with_endpoint(true),
            commit: CommitRule::LastTimestamp,
        }
    }

    /// Best-effort real time, for a human reading during the utterance:
    /// as [`conservative`](Self::conservative), plus a draft every 600 ms.
    pub fn responsive() -> Self {
        Self {
            triggers: DecodeTriggers::new()
                .with_endpoint(true)
                .with_interval(Some(Duration::from_millis(600))),
            commit: CommitRule::LastTimestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The presets are the three deployment targets, and differ only by
    /// the two axes.
    #[test]
    fn test_presets() {
        let offline = EmissionPolicy::offline();
        assert!(offline.triggers.window_full);
        assert!(!offline.triggers.endpoint);
        assert_eq!(offline.triggers.interval, None);
        assert_eq!(offline.commit, CommitRule::Complete);

        let conservative = EmissionPolicy::conservative();
        assert!(conservative.triggers.window_full);
        assert!(conservative.triggers.endpoint);
        assert_eq!(conservative.triggers.interval, None);
        assert_eq!(conservative.commit, CommitRule::LastTimestamp);

        let responsive = EmissionPolicy::responsive();
        assert_eq!(
            responsive.triggers,
            DecodeTriggers::new()
                .with_endpoint(true)
                .with_interval(Some(Duration::from_millis(600)))
        );
        assert_eq!(responsive.commit, conservative.commit);
    }
}
