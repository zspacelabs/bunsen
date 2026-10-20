//! Where to cut a window at a change of speaker.

use burn::config::Config;

use crate::kits::speech::diarization::SpeakerActivity;

/// Config for [`speaker_cut`]: when a change of speaker ends a window.
#[derive(Config, Debug, PartialEq)]
pub struct SpeakerCutConfig {
    /// A window runs at least this long, in seconds, before a change of
    /// speaker may end it.
    #[config(default = "2.0")]
    pub min_window_seconds: f64,

    /// A speaker must hold this long, in seconds, to count as a turn: a
    /// shorter stretch (a backchannel, a flicker) does not end the window.
    #[config(default = "0.5")]
    pub min_turn_seconds: f64,

    /// A frame's dominant speaker must reach this probability; below it the
    /// frame is nobody's.
    #[config(default = "0.5")]
    pub threshold: f64,
}

impl Default for SpeakerCutConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Where a window is cut at a change of speaker, for an engine that
/// decodes a window whole: the first frame, past
/// [`min_window_seconds`](SpeakerCutConfig::min_window_seconds), that starts
/// a turn of a speaker other than the window's current one, a turn being a
/// speaker dominant for
/// [`min_turn_seconds`](SpeakerCutConfig::min_turn_seconds) running. Decoded up
/// to there, the window is one speaker's, as far as the turns go; the rest is
/// decoded with what follows.
///
/// A frame's dominant speaker is the one with the highest mean probability
/// over the activity frames the frame overlaps, when it reaches the
/// threshold; frames the activity does not cover are nobody's.
///
/// # Arguments
/// * `activity` - the diarizer's frames over the window.
/// * `start` - the stream sample the window's first frame starts at.
/// * `hop` - the samples in one of the window's frames.
/// * `frames` - the window's frames.
/// * `sample_rate` - in Hz.
/// * `config` - the thresholds.
///
/// # Returns
/// The frame to cut at, in `1..frames`; `None` when no turn ends the window.
pub fn speaker_cut(
    activity: &SpeakerActivity,
    start: usize,
    hop: usize,
    frames: usize,
    sample_rate: usize,
    config: &SpeakerCutConfig,
) -> Option<usize> {
    let to_frames = |seconds: f64| ((seconds * sample_rate as f64) / hop as f64).round() as usize;
    let min_window = to_frames(config.min_window_seconds).max(1);
    let min_turn = to_frames(config.min_turn_seconds).max(1);

    // Runs of each frame's dominant speaker.
    let mut runs: Vec<(Option<usize>, usize, usize)> = Vec::new();
    for k in 0..frames {
        let who = dominant(activity, start + k * hop, hop, config.threshold);
        match runs.last_mut() {
            Some((speaker, _, len)) if *speaker == who => *len += 1,
            _ => runs.push((who, k, 1)),
        }
    }

    let mut current = None;
    for (speaker, at, len) in runs {
        let Some(speaker) = speaker else {
            continue;
        };
        if len < min_turn {
            continue;
        }
        match current {
            None => current = Some(speaker),
            Some(c) if c == speaker => {}
            Some(_) if at >= min_window => return Some(at),
            Some(_) => current = Some(speaker),
        }
    }
    None
}

/// The speaker dominant over samples `[start, start + len)`, if one reaches
/// `threshold`.
fn dominant(
    activity: &SpeakerActivity,
    start: usize,
    len: usize,
    threshold: f64,
) -> Option<usize> {
    let window = activity.window(start, start + len);
    if window.is_empty() {
        return None;
    }
    let n = window.frames() as f32;
    let mut means = vec![0.0f32; window.n_speakers];
    for i in 0..window.frames() {
        for (m, p) in means.iter_mut().zip(window.frame(i)) {
            *m += p / n;
        }
    }
    let (speaker, &best) = means.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    (best as f64 >= threshold).then_some(speaker)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Activity at 100 Hz frames of 10 samples (a 1 kHz toy rate), one
    /// speaker per stretch: `(speaker, frames)`; `None` is silence.
    fn activity(stretches: &[(Option<usize>, usize)]) -> SpeakerActivity {
        let mut probs = Vec::new();
        for &(who, n) in stretches {
            for _ in 0..n {
                probs.extend((0..2).map(|k| if who == Some(k) { 0.9 } else { 0.05 }));
            }
        }
        SpeakerActivity::try_new(0, 10, 2, probs).unwrap()
    }

    fn config() -> SpeakerCutConfig {
        // 2 s and 0.5 s at 1 kHz with 10-sample frames: 200 and 50 frames.
        SpeakerCutConfig::new()
    }

    #[test]
    fn test_cuts_at_the_first_turn_past_the_minimum() {
        let a = activity(&[(Some(0), 300), (Some(1), 100), (Some(0), 100)]);
        assert_eq!(speaker_cut(&a, 0, 10, 500, 1000, &config()), Some(300));
    }

    #[test]
    fn test_short_stretches_and_silence_do_not_cut() {
        // A backchannel of 30 frames, then silence, then the same speaker.
        let a = activity(&[(Some(0), 250), (Some(1), 30), (None, 50), (Some(0), 170)]);
        assert_eq!(speaker_cut(&a, 0, 10, 500, 1000, &config()), None);
    }

    #[test]
    fn test_an_early_turn_moves_the_current_speaker() {
        // Speaker 1 takes over before the minimum, and holds: no cut for that
        // turn; speaker 0's return past it cuts.
        let a = activity(&[(Some(0), 100), (Some(1), 200), (Some(0), 200)]);
        assert_eq!(speaker_cut(&a, 0, 10, 500, 1000, &config()), Some(300));
    }

    #[test]
    fn test_frames_past_the_activity_are_nobodys() {
        let a = activity(&[(Some(0), 250)]);
        assert_eq!(speaker_cut(&a, 0, 10, 500, 1000, &config()), None);
        // Offset: the window starts 100 frames into the activity.
        let a = activity(&[(Some(0), 300), (Some(1), 100)]);
        assert_eq!(speaker_cut(&a, 1000, 10, 300, 1000, &config()), Some(200));
    }
}
