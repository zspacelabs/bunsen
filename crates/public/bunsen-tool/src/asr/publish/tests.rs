use bunsen::kits::speech::asr::driver::{
    SpeakerLabel,
    TranscriptSegment,
};

use super::*;

fn segment(
    text: &str,
    start: f64,
    speaker: Option<(usize, f32)>,
) -> TranscriptSegment {
    TranscriptSegment {
        start,
        end: start + 0.5,
        tokens: Vec::new(),
        text: Some(text.to_string()),
        speaker: speaker.map(|(index, confidence)| SpeakerLabel { index, confidence }),
    }
}

/// Consecutive segments of one speaker are one statement, its
/// confidence their mean; empty ones are left out, and a change of
/// speaker, nobody included, starts a statement.
#[test]
fn test_statements() {
    let st = statements(&[
        segment(" hello", 0.0, Some((0, 0.8))),
        segment("  ", 0.5, Some((1, 0.9))),
        segment("there", 1.0, Some((0, 0.6))),
        segment("hi", 2.0, Some((1, 0.9))),
        segment("um", 3.0, None),
    ]);
    let texts: Vec<&str> = st.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["hello there", "hi", "um"]);
    assert_eq!((st[0].start, st[0].end), (0.0, 1.5));
    let label = st[0].speaker.unwrap();
    assert_eq!(label.index, 0);
    assert!((label.confidence - 0.7).abs() < 1e-6);
    assert_eq!(st[2].speaker, None);

    // Undiarized, the segments are one statement.
    let st = statements(&[segment("the", 0.0, None), segment("moon.", 1.0, None)]);
    assert_eq!(st.len(), 1);
    assert_eq!(st[0].text, "the moon.");
    assert!(statements(&[]).is_empty());
}
