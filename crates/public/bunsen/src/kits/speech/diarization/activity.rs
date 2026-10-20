//! # Speaker activity, and the diarizer seam.

use std::fmt::Debug;

use dyn_clone::DynClone;

use crate::errors::{
    BunsenError,
    BunsenResult,
    ConstraintError,
    Rule,
};

/// Per-frame speaker probabilities: `[frames, n_speakers]`, row-major.
///
/// Frame `i` covers samples `[start + i * frame_samples, start + (i + 1) *
/// frame_samples)` of its stream. Speaker `k` is the diarizer's `k`-th
/// speaker, numbered in the order it first heard them.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerActivity {
    /// The stream sample the first frame starts at.
    pub start: usize,

    /// The samples in one frame.
    pub frame_samples: usize,

    /// The speakers each frame scores.
    pub n_speakers: usize,

    /// `frames * n_speakers` probabilities, frame-major.
    pub probs: Vec<f32>,
}

impl SpeakerActivity {
    /// Activity with no frames, from `start`.
    pub fn empty(
        start: usize,
        frame_samples: usize,
        n_speakers: usize,
    ) -> Self {
        Self {
            start,
            frame_samples,
            n_speakers,
            probs: Vec::new(),
        }
    }

    /// Activity over `probs`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, if `frame_samples` or `n_speakers` is zero
    /// or `probs` is not a whole number of frames.
    pub fn try_new(
        start: usize,
        frame_samples: usize,
        n_speakers: usize,
        probs: Vec<f32>,
    ) -> BunsenResult<Self> {
        for (field, value) in [("frame_samples", frame_samples), ("n_speakers", n_speakers)] {
            if value == 0 {
                return Err(ConstraintError::zero_or_empty("SpeakerActivity", field).into());
            }
        }
        if !probs.len().is_multiple_of(n_speakers) {
            return Err(ConstraintError::new(
                "SpeakerActivity",
                "probs",
                Rule::NotMultiple {
                    value: probs.len(),
                    of: format!("n_speakers ({n_speakers})"),
                },
            )
            .into());
        }
        Ok(Self {
            start,
            frame_samples,
            n_speakers,
            probs,
        })
    }

    /// The frame count.
    pub fn frames(&self) -> usize {
        self.probs.len() / self.n_speakers.max(1)
    }

    /// Whether there are no frames.
    pub fn is_empty(&self) -> bool {
        self.probs.is_empty()
    }

    /// The stream sample one past the last frame.
    pub fn end(&self) -> usize {
        self.start + self.frames() * self.frame_samples
    }

    /// Frame `i`'s probabilities, one per speaker.
    ///
    /// # Panics
    /// If `i` is not a frame.
    pub fn frame(
        &self,
        i: usize,
    ) -> &[f32] {
        &self.probs[i * self.n_speakers..(i + 1) * self.n_speakers]
    }

    /// Appends `next`, which must follow these frames.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `next` does
    /// not start at [`end`](Self::end), or has another frame size or speaker
    /// count.
    pub fn append(
        &mut self,
        next: SpeakerActivity,
    ) -> BunsenResult<()> {
        if next.frame_samples != self.frame_samples || next.n_speakers != self.n_speakers {
            return Err(BunsenError::illegal(format!(
                "SpeakerActivity::append: frames of {} samples over {} speakers cannot take \
                 frames of {} over {}",
                self.frame_samples, self.n_speakers, next.frame_samples, next.n_speakers
            )));
        }
        if !next.is_empty() && next.start != self.end() {
            return Err(BunsenError::illegal(format!(
                "SpeakerActivity::append: the next frames start at sample {}, not at {}",
                next.start,
                self.end()
            )));
        }
        self.probs.extend(next.probs);
        Ok(())
    }

    /// The frames that overlap stream samples `[start, end)`.
    pub fn window(
        &self,
        start: usize,
        end: usize,
    ) -> SpeakerActivity {
        let f = self.frame_samples;
        let last = end
            .saturating_sub(self.start)
            .div_ceil(f)
            .min(self.frames());
        let first = (start.saturating_sub(self.start) / f).min(last);
        Self {
            start: self.start + first * f,
            frame_samples: f,
            n_speakers: self.n_speakers,
            probs: self.probs[first * self.n_speakers..last * self.n_speakers].to_vec(),
        }
    }

    /// Drops every frame that ends at or before stream sample `sample`.
    pub fn drop_before(
        &mut self,
        sample: usize,
    ) {
        let frames = (sample.saturating_sub(self.start) / self.frame_samples).min(self.frames());
        self.probs.drain(..frames * self.n_speakers);
        self.start += frames * self.frame_samples;
    }
}

/// A diarizer: per-frame speaker activity, which a
/// [`TurnFilter`](crate::kits::speech::diarization::TurnFilter) turns into
/// speaker turns, and an ASR driver attributes transcript segments with.
///
/// Shared and immutable; each stream opens its own
/// [`SpeakerActivityStream`] for its state.
pub trait SpeakerActivityModel: Debug + Send + Sync {
    /// The sample rate it runs at, in Hz.
    fn sample_rate(&self) -> usize;

    /// The samples in one output frame.
    fn frame_samples(&self) -> usize;

    /// The most speakers it tells apart.
    fn max_speakers(&self) -> usize;

    /// Opens a stream.
    ///
    /// # Errors
    /// As the model's stream state.
    fn new_stream(&self) -> BunsenResult<Box<dyn SpeakerActivityStream>>;
}

/// One stream's diarization state.
///
/// `Clone`, through [`DynClone`], because the contexts that hold one boxed
/// are.
pub trait SpeakerActivityStream: Debug + Send + Sync + DynClone {
    /// Takes samples in, following the samples of the previous call, and
    /// returns the frames they finalize, scored over the model's
    /// [`max_speakers`](SpeakerActivityModel::max_speakers).
    ///
    /// A frame is final once the model has seen all the audio it looks at:
    /// a diarizer with right context finalizes a frame some samples after
    /// its end. The frames returned over a stream are contiguous from sample
    /// 0, however the samples are split across calls.
    ///
    /// # Errors
    /// As the model.
    fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<SpeakerActivity>;

    /// Ends the input, and returns the frames left, the last one covering the
    /// last sample.
    ///
    /// # Errors
    /// As the model.
    fn finish(&mut self) -> BunsenResult<SpeakerActivity>;
}

dyn_clone::clone_trait_object!(<> SpeakerActivityStream);

#[cfg(test)]
mod tests {
    use super::*;

    fn activity(
        start: usize,
        frames: usize,
    ) -> SpeakerActivity {
        let probs = (0..frames * 2).map(|i| i as f32 / 100.0).collect();
        SpeakerActivity::try_new(start, 10, 2, probs).unwrap()
    }

    #[test]
    fn test_frames_and_ends() {
        let a = activity(30, 4);
        assert_eq!(a.frames(), 4);
        assert_eq!(a.end(), 70);
        assert_eq!(a.frame(1), [0.02, 0.03]);
        assert!(!a.is_empty());
        assert!(SpeakerActivity::empty(5, 10, 2).is_empty());
    }

    #[test]
    fn test_try_new_checks_its_shape() {
        assert!(SpeakerActivity::try_new(0, 10, 2, vec![0.0; 3]).is_err());
        assert!(SpeakerActivity::try_new(0, 0, 2, vec![]).is_err());
        assert!(SpeakerActivity::try_new(0, 10, 0, vec![]).is_err());
    }

    #[test]
    fn test_append_must_follow() {
        let mut a = activity(0, 2);
        a.append(activity(20, 1)).unwrap();
        assert_eq!(a.frames(), 3);
        assert_eq!(a.end(), 30);

        assert!(a.append(activity(40, 1)).is_err());
        let other = SpeakerActivity::try_new(30, 5, 2, vec![0.0; 2]).unwrap();
        assert!(a.append(other).is_err());
        a.append(SpeakerActivity::empty(99, 10, 2)).unwrap();
        assert_eq!(a.frames(), 3);
    }

    #[test]
    fn test_window_takes_the_overlapping_frames() {
        let a = activity(30, 4);
        let w = a.window(45, 52);
        assert_eq!(w.start, 40);
        assert_eq!(w.frames(), 2);
        assert_eq!(w.frame(0), a.frame(1));
        assert!(a.window(70, 90).is_empty());
        assert_eq!(a.window(0, 1000), a);
    }

    #[test]
    fn test_drop_before_keeps_frames_that_end_after() {
        let mut a = activity(0, 4);
        a.drop_before(25);
        assert_eq!(a.start, 20);
        assert_eq!(a.frames(), 2);
        assert_eq!(a.frame(0), [0.04, 0.05]);

        a.drop_before(1000);
        assert!(a.is_empty());
        assert_eq!(a.start, 40);
    }
}
