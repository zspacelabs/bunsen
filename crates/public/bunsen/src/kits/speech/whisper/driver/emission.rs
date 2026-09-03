//! # Emission: what a stream says, and when.
//!
//! Two axes and two output variants. **A `window_full` or `endpoint` decode
//! commits, per the commit rule; an `interval` decode drafts.** Everything the
//! three deployment targets differ by falls out of those two sentences, and
//! the three presets on [`EmissionPolicy`] are those targets.
//!
//! A [`Draft`](WhisperEmission::Draft) always covers *all* audio after the last
//! commit and supersedes the previous draft entirely, so there is no
//! retraction protocol, no sequence numbers, and no way to hold two drafts at
//! once. Under [`offline`](EmissionPolicy::offline) and
//! [`conservative`](EmissionPolicy::conservative) the variant is never
//! constructed.

use std::time::Duration;

use burn::config::Config;

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

/// A span of transcript with its place in media time.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    /// Media time of the segment's start, in seconds, through the stream's
    /// clock.
    pub start: f64,

    /// Media time of the segment's end, in seconds.
    pub end: f64,

    /// The ids the decode produced for this span, prompt and stop token
    /// excluded.
    pub tokens: Vec<i64>,

    /// The text of the text tokens, when the driver has a detokenizer.
    pub text: Option<String>,
}

/// What a push hands back.
#[derive(Debug, Clone, PartialEq)]
pub enum WhisperEmission {
    /// Final. Will never be revised.
    Committed(TranscriptSegment),

    /// Provisional. Covers all audio since the last commit, and replaces the
    /// previous draft whole.
    Draft(TranscriptSegment),
}

impl WhisperEmission {
    /// The segment, whichever variant carries it.
    pub fn segment(&self) -> &TranscriptSegment {
        match self {
            Self::Committed(s) | Self::Draft(s) => s,
        }
    }

    /// Whether this is final.
    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Committed(_))
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

    #[test]
    fn test_emission_accessors() {
        let segment = TranscriptSegment {
            start: 1.0,
            end: 2.0,
            tokens: vec![1, 2],
            text: None,
        };
        let committed = WhisperEmission::Committed(segment.clone());
        let draft = WhisperEmission::Draft(segment.clone());

        assert!(committed.is_committed());
        assert!(!draft.is_committed());
        assert_eq!(committed.segment(), &segment);
        assert_eq!(draft.segment(), &segment);
    }
}
