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

/// A tagged transcript emission.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptEvent {
    /// Final. Will never be revised.
    Committed(TranscriptSegment),

    /// Provisional. Covers all audio since the last commit, and replaces the
    /// previous draft whole.
    Draft(TranscriptSegment),
}

impl TranscriptEvent {
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
