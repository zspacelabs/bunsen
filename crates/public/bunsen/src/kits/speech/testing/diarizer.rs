//! A diarizer that knows who speaks when.
use crate::{
    errors::BunsenResult,
    kits::speech::diarization::{
        SpeakerActivity,
        SpeakerActivityModel,
        SpeakerActivityStream,
    },
};

/// A diarizer that reads a script: from each `(sample, speaker)` on, that
/// speaker talks (`None`: nobody), at probability 0.9 against 0.05 for the
/// others, in frames of `frame_samples`, each finalized `latency` samples
/// after its end.
///
/// For tests of what a driver or an engine does with speaker activity,
/// without a model.
#[derive(Clone, Debug)]
pub struct ScriptedDiarizer {
    /// The sample rate, in Hz.
    pub sample_rate: usize,

    /// The samples in one frame.
    pub frame_samples: usize,

    /// The speakers each frame scores.
    pub n_speakers: usize,

    /// The samples past a frame's end the diarizer waits for before
    /// finalizing it.
    pub latency: usize,

    /// From each sample on, who talks, in order of sample.
    pub script: Vec<(usize, Option<usize>)>,
}

impl ScriptedDiarizer {
    /// Who talks at `sample`.
    pub fn speaker_at(
        &self,
        sample: usize,
    ) -> Option<usize> {
        self.script
            .iter()
            .rev()
            .find(|(from, _)| *from <= sample)
            .and_then(|&(_, who)| who)
    }
}

impl SpeakerActivityModel for ScriptedDiarizer {
    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    fn max_speakers(&self) -> usize {
        self.n_speakers
    }

    fn new_stream(&self) -> BunsenResult<Box<dyn SpeakerActivityStream>> {
        Ok(Box::new(ScriptedStream {
            diarizer: self.clone(),
            seen: 0,
            emitted: 0,
        }))
    }
}

#[derive(Clone, Debug)]
struct ScriptedStream {
    diarizer: ScriptedDiarizer,
    seen: usize,
    emitted: usize,
}

impl ScriptedStream {
    /// Frames `[emitted, upto)`.
    fn frames(
        &mut self,
        upto: usize,
    ) -> BunsenResult<SpeakerActivity> {
        let d = &self.diarizer;
        let start = self.emitted * d.frame_samples;
        let mut probs = Vec::new();
        for f in self.emitted..upto {
            let who = d.speaker_at(f * d.frame_samples);
            probs.extend((0..d.n_speakers).map(|k| if who == Some(k) { 0.9 } else { 0.05 }));
        }
        self.emitted = upto.max(self.emitted);
        SpeakerActivity::try_new(start, d.frame_samples, d.n_speakers, probs)
    }
}

impl SpeakerActivityStream for ScriptedStream {
    fn push(
        &mut self,
        samples: &[f32],
    ) -> BunsenResult<SpeakerActivity> {
        self.seen += samples.len();
        let upto = self.seen.saturating_sub(self.diarizer.latency) / self.diarizer.frame_samples;
        self.frames(upto)
    }

    fn finish(&mut self) -> BunsenResult<SpeakerActivity> {
        let upto = self.seen.div_ceil(self.diarizer.frame_samples);
        self.frames(upto)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_follows_its_script() {
        let d = ScriptedDiarizer {
            sample_rate: 100,
            frame_samples: 10,
            n_speakers: 2,
            latency: 15,
            script: vec![(0, Some(0)), (30, None), (50, Some(1))],
        };
        assert_eq!(d.speaker_at(29), Some(0));
        assert_eq!(d.speaker_at(30), None);
        assert_eq!(d.speaker_at(77), Some(1));

        let mut stream = d.new_stream().unwrap();
        let mut a = stream.push(&[0.0; 45]).unwrap();
        assert_eq!(a.frames(), 3);
        a.append(stream.push(&[0.0; 30]).unwrap()).unwrap();
        a.append(stream.finish().unwrap()).unwrap();
        assert_eq!(a.frames(), 8);
        assert_eq!(a.frame(0), [0.9, 0.05]);
        assert_eq!(a.frame(3), [0.05, 0.05]);
        assert_eq!(a.frame(7), [0.05, 0.9]);
    }
}
