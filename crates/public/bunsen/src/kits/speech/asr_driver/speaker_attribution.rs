//! # Speaker attribution: which speaker a transcript segment belongs to.

use std::fmt::Debug;

use crate::kits::speech::diarization::{
    SpeakerActivity,
    TurnFilterConfig,
};

/// The speaker a segment is attributed to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeakerLabel {
    /// The diarizer's speaker index.
    pub index: usize,

    /// How strongly the segment is that speaker's, in `[0, 1]`, as the
    /// attribution rule measures it.
    pub confidence: f32,
}

/// A rule that attributes a stretch of a stream to a speaker, from a
/// diarizer's activity over it.
///
/// An [`AsrStreamDriver`](crate::kits::speech::asr_driver::AsrStreamDriver)
/// with a diarizer attached labels every segment it emits through one; the
/// rule sees only the activity, never the model, so any diarizer and any
/// rule combine.
pub trait SpeakerAttribution: Debug + Send + Sync {
    /// The speaker of stream samples `[start, end)`, or `None` when the
    /// activity says nobody.
    ///
    /// # Arguments
    /// * `activity` - the diarizer's frames, covering the stretch as far as the
    ///   diarizer has finalized it.
    /// * `start`, `end` - the stretch, in stream samples.
    fn attribute(
        &self,
        activity: &SpeakerActivity,
        start: usize,
        end: usize,
    ) -> Option<SpeakerLabel>;
}

/// The frames of `activity` that overlap samples `[start, end)`, from `lead`
/// frames before the first.
fn overlapping(
    activity: &SpeakerActivity,
    start: usize,
    end: usize,
    lead: usize,
) -> std::ops::Range<usize> {
    if activity.is_empty() || end <= start {
        return 0..0;
    }
    let f = activity.frame_samples;
    let first = (start.saturating_sub(activity.start) / f).saturating_sub(lead);
    let last = end
        .saturating_sub(activity.start)
        .div_ceil(f)
        .min(activity.frames());
    first.min(last)..last
}

/// `NeMo`'s word rule (`get_word_dict_content_offline`): each speaker's mean
/// probability over the frames the stretch overlaps, each clamped to at least
/// 0.01 and normalized to sum to 1; the speaker with the most. The
/// confidence is that normalized share.
///
/// `lead_frames` widens the stretch backwards by that many frames: `NeMo`
/// reads its words from one frame before their first.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeanActivity {
    /// Frames before the stretch that count towards it.
    pub lead_frames: usize,
}

impl SpeakerAttribution for MeanActivity {
    fn attribute(
        &self,
        activity: &SpeakerActivity,
        start: usize,
        end: usize,
    ) -> Option<SpeakerLabel> {
        let frames = overlapping(activity, start, end, self.lead_frames);
        if frames.is_empty() {
            return None;
        }
        let n = frames.len() as f32;
        let mut means = vec![0.0f32; activity.n_speakers];
        for i in frames {
            for (m, p) in means.iter_mut().zip(activity.frame(i)) {
                *m += p / n;
            }
        }
        for m in &mut means {
            *m = m.max(0.01);
        }
        let total: f32 = means.iter().sum();
        let (index, &best) = means
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("at least one speaker");
        Some(SpeakerLabel {
            index,
            confidence: best / total,
        })
    }
}

/// The speaker whose turns cover most of the stretch: the frames it overlaps
/// run through a [`TurnFilterConfig`], and the speaker with the longest turn
/// time inside the stretch wins. The confidence is the share of the stretch
/// that speaker's turns cover; a stretch no turn touches is nobody's.
///
/// The turns are found over the stretch's own frames, so padding and the
/// filters see no context past its ends.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TurnOverlap {
    /// How the stretch's activity becomes turns.
    pub filter: TurnFilterConfig,
}

impl SpeakerAttribution for TurnOverlap {
    fn attribute(
        &self,
        activity: &SpeakerActivity,
        start: usize,
        end: usize,
    ) -> Option<SpeakerLabel> {
        let frames = overlapping(activity, start, end, 0);
        if frames.is_empty() {
            return None;
        }
        let s = activity.n_speakers;
        let f = activity.frame_samples as f64;
        let origin = (activity.start + frames.start * activity.frame_samples) as f64;
        let probs = &activity.probs[frames.start * s..frames.end * s];
        // Turn times are in units of frames here: seconds of one-sample
        // "frames" would do as well, and this keeps them exact.
        let turns = self.filter.turns(probs, s, 1.0);

        let (lo, hi) = (start as f64, end as f64);
        let mut covered = vec![0.0f64; s];
        for t in turns {
            let (a, b) = (origin + t.start * f, origin + t.end * f);
            covered[t.speaker] += (b.min(hi) - a.max(lo)).max(0.0);
        }
        let (index, &best) = covered
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("at least one speaker");
        (best > 0.0).then(|| SpeakerLabel {
            index,
            confidence: (best / (hi - lo)).min(1.0) as f32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two speakers over eight frames of 10 samples: speaker 0 talks in the
    /// first half, speaker 1 in the second.
    fn activity() -> SpeakerActivity {
        let mut probs = Vec::new();
        for i in 0..8 {
            probs.extend(if i < 4 { [0.9, 0.1] } else { [0.05, 0.8] });
        }
        SpeakerActivity::try_new(100, 10, 2, probs).unwrap()
    }

    #[test]
    fn test_mean_activity_follows_the_frames() {
        let rule = MeanActivity::default();
        let a = activity();
        assert_eq!(rule.attribute(&a, 100, 140).unwrap().index, 0);
        assert_eq!(rule.attribute(&a, 145, 180).unwrap().index, 1);

        // Mostly the second half: speaker 1, with less confidence.
        let mixed = rule.attribute(&a, 125, 180).unwrap();
        assert_eq!(mixed.index, 1);
        let pure = rule.attribute(&a, 140, 180).unwrap();
        assert!(mixed.confidence < pure.confidence);

        // Means 0.9 and 0.1, normalized.
        assert!((rule.attribute(&a, 100, 140).unwrap().confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_lead_frames_reach_back() {
        let a = activity();
        // The stretch's own frame is speaker 1's; the two before are 0's.
        assert_eq!(
            MeanActivity::default()
                .attribute(&a, 140, 150)
                .unwrap()
                .index,
            1
        );
        let rule = MeanActivity { lead_frames: 2 };
        assert_eq!(rule.attribute(&a, 140, 150).unwrap().index, 0);
    }

    #[test]
    fn test_nothing_covered_is_nobody() {
        let a = activity();
        assert!(MeanActivity::default().attribute(&a, 300, 400).is_none());
        assert!(MeanActivity::default().attribute(&a, 120, 120).is_none());
        let empty = SpeakerActivity::empty(0, 10, 2);
        assert!(MeanActivity::default().attribute(&empty, 0, 50).is_none());
    }

    #[test]
    fn test_turn_overlap() {
        let rule = TurnOverlap::default();
        let a = activity();
        let label = rule.attribute(&a, 120, 180).unwrap();
        assert_eq!(label.index, 1);
        assert!((label.confidence - 40.0 / 60.0).abs() < 1e-6);

        let label = rule.attribute(&a, 100, 130).unwrap();
        assert_eq!(label.index, 0);
        assert!((label.confidence - 1.0).abs() < 1e-6);

        // Nobody above the thresholds: nobody.
        let quiet = SpeakerActivity::try_new(0, 10, 2, vec![0.1; 8]).unwrap();
        assert!(rule.attribute(&quiet, 0, 40).is_none());
    }
}
