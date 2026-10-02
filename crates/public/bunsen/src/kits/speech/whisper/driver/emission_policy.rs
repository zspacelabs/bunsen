//! # Emission: what a stream says, and when.

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
    /// final, except a decode's unfinished tail when timestamps are on.
    /// Needs a voice-activity model.
    Conservative,

    /// Conservative, plus a draft every 600 ms of speech.
    /// Needs a voice-activity model.
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
///
/// A `window_full` or `endpoint` decode commits, per the [`CommitRule`];
/// an `interval` decode drafts. At least one of `window_full` and
/// `endpoint` must be on, or nothing would ever be decoded.
#[derive(Config, Debug, PartialEq, Eq)]
pub struct DecodeTriggers {
    /// Decode when a full window of audio has accumulated past the seek
    /// pointer. The only trigger that needs no voice activity. Alongside
    /// `endpoint`, a full window is decoded only while speech is in
    /// progress inside it.
    #[config(default = "true")]
    pub window_full: bool,

    /// Decode when the voice-activity gate closes a region. Needs a
    /// voice-activity model on the driver
    /// ([`with_vad`](super::WhisperStreamDriver::with_vad)).
    #[config(default = "false")]
    pub endpoint: bool,

    /// Decode every so often while speech is in progress, as a draft: once
    /// this much media time has passed since the last draft or commit.
    ///
    /// Whether speech is in progress is the voice-activity gate's to say,
    /// and the gate runs only under `endpoint`; without it no draft is
    /// ever made.
    #[config(default = "None")]
    pub interval: Option<Duration>,
}

/// When a decode's output becomes final.
///
/// Without timestamps every rule commits a decode whole. With them
/// ([`timestamps`](super::WhisperStreamDriverConfig::timestamps)), a
/// decode is split on its timestamps, the closed segments commit, and the
/// seek pointer advances to the last closed timestamp; the rules differ in
/// what becomes of the unfinished tail after it.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum CommitRule {
    /// The closed segments commit, and the unfinished tail is dropped, to
    /// be decoded again with more audio behind it: what upstream's seek
    /// loop does. Right when a window is the last thing that will ever be
    /// said about its audio.
    Complete,

    /// As [`Complete`](Self::Complete), and the unfinished tail is also
    /// emitted, as a [`Draft`](super::TranscriptEvent::Draft) opening on its
    /// timestamp, before it is decoded again.
    LastTimestamp,

    /// Commit a prefix once `runs` consecutive decodes agree on it. The one
    /// rule under which a provisional decode becomes load-bearing.
    ///
    /// Not implemented yet: the config accepts it, and a driver under it
    /// commits as under [`Complete`](Self::Complete).
    Agreement {
        /// Consecutive decodes that must agree.
        runs: usize,
    },
}

/// The emission policy: [`DecodeTriggers`] and a [`CommitRule`].
///
/// Two axes, and two output variants,
/// [`Committed`](super::TranscriptEvent::Committed) and
/// [`Draft`](super::TranscriptEvent::Draft). The policy is part of the
/// [`WhisperStreamDriverConfig`](super::WhisperStreamDriverConfig), so the
/// driver is configured, never forked. The presets are its three
/// deployment targets; anything else is a custom pairing. The
/// [`driver`](super) module docs explain the semantics.
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
    /// Every commit is final. With timestamps on, the unfinished tail of a
    /// decode is also emitted as a draft; a consumer that wants final text
    /// only keeps the [`Committed`](super::TranscriptEvent::Committed)
    /// events.
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
