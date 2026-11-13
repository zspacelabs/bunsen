use std::path::Path;

use bunsen::kits::speech::diarization::{
    SpeakerTurn,
    sortformer::pretrained::default_sortformer_factory,
};

use super::*;
use crate::{
    asr::recording_id,
    support::with_presets,
};

/// Text lines and JSON name speakers as `NeMo` and the RTTM do, and
/// round times to the millisecond.
#[test]
fn test_formats() {
    let turn = SpeakerTurn {
        speaker: 2,
        start: 1.23456,
        end: 4.5,
    };
    assert_eq!(text_line(&turn), "1.235 4.500 speaker_2");
    let json = serde_json::to_string(&JsonTurn::from(&turn)).unwrap();
    assert_eq!(json, r#"{"speaker":"speaker_2","start":1.235,"end":4.5}"#);
    assert_eq!(recording_id(Path::new("/a/b/meeting.wav")), "meeting");
}

/// Totals are by speaker, in index order.
#[test]
fn test_speaker_totals() {
    let turn = |speaker, start, end| SpeakerTurn {
        speaker,
        start,
        end,
    };
    let totals = speaker_totals(&[turn(1, 0.0, 1.5), turn(0, 1.0, 2.0), turn(1, 3.0, 4.0)]);
    let got: Vec<(&str, f64, usize)> = totals
        .iter()
        .map(|t| (t.speaker.as_str(), t.seconds, t.turns))
        .collect();
    assert_eq!(got, [("speaker_0", 1.0, 1), ("speaker_1", 2.5, 2)]);
}

/// A diarizer's latency and post-processing are every row's axes: a
/// model's box names them, and they default to `low` and `none`.
#[test]
fn test_presets() {
    assert_eq!(
        with_presets("sortformer-v2[dihard3]", &["high"]).unwrap(),
        "sortformer-v2[dihard3,high]"
    );

    let rows = default_sortformer_factory().unwrap().list();
    assert!(!rows.is_empty());
    for (_, row) in rows {
        let axes: Vec<&str> = row
            .presets
            .axes
            .iter()
            .map(|a| a.axis.name.leaf())
            .collect();
        assert_eq!(axes, ["latency", "postprocess"], "{}", row.name);
        assert_eq!(row.presets.default_names(), ["low", "none"], "{}", row.name);
    }
}
