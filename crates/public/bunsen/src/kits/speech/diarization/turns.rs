//! # Speaker turns: activity to who spoke when.

use burn::config::Config;

/// One speaker's stretch of speech.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerTurn {
    /// The speaker: the diarizer's index.
    pub speaker: usize,

    /// The start, in seconds from the first frame the filter saw.
    pub start: f64,

    /// The end, in seconds.
    pub end: f64,
}

impl SpeakerTurn {
    /// The turn's length, in seconds.
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Config for [`TurnFilter`]: `NeMo`'s diarization post-processing
/// (`PostProcessingParams`, applied per speaker by `binarization_vectorized`
/// then `filtering`).
///
/// Each speaker's track is binarized with hysteresis. When `onset >= offset`,
/// a frame above `onset` turns speech on, one below `offset` turns it off, and
/// one between keeps the state. When `onset < offset`, a frame at or above
/// `offset` turns it on, one at or below `onset` turns it off, and one strictly
/// between *toggles* it. A run of speech frames from frame `s` up to frame `e`
/// (exclusive) is the stretch `[s * frame - pad_onset, e * frame +
/// pad_offset]`, its start clamped at 0, and stretches that then overlap or
/// touch merge.
///
/// Then the filters, when their thresholds are positive: turns shorter than
/// `min_duration_on` are dropped, and gaps between turns shorter than
/// `min_duration_off` are filled. `filter_speech_first` (the default) drops
/// before it fills, as `NeMo` does by default; otherwise it fills first.
///
/// [`none`](Self::none) is `NeMo`'s bypass: thresholds of 0.5, no padding, no
/// filters.
#[derive(Config, Debug, PartialEq)]
pub struct TurnFilterConfig {
    /// The onset threshold.
    #[config(default = "0.5")]
    pub onset: f64,

    /// The offset threshold.
    #[config(default = "0.5")]
    pub offset: f64,

    /// Seconds added before each turn.
    #[config(default = "0.0")]
    pub pad_onset: f64,

    /// Seconds added after each turn.
    #[config(default = "0.0")]
    pub pad_offset: f64,

    /// Turns shorter than this, in seconds, are dropped; 0 keeps them all.
    #[config(default = "0.0")]
    pub min_duration_on: f64,

    /// Gaps shorter than this, in seconds, are filled; 0 fills none.
    #[config(default = "0.0")]
    pub min_duration_off: f64,

    /// Drop short turns before filling short gaps.
    #[config(default = "true")]
    pub filter_speech_first: bool,
}

impl Default for TurnFilterConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnFilterConfig {
    /// No post-processing: `NeMo`'s bypass, thresholds of 0.5, no padding, no
    /// filters.
    pub fn none() -> Self {
        Self::new()
    }

    /// `NeMo`'s tuned parameters for streaming Sortformer v2 on DIHARD III dev
    /// (`diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml`).
    pub fn dihard3_v2() -> Self {
        Self::new()
            .with_onset(0.56)
            .with_offset(1.0)
            .with_pad_onset(0.063)
            .with_pad_offset(0.002)
            .with_min_duration_on(0.007)
            .with_min_duration_off(0.151)
    }

    /// `NeMo`'s tuned parameters for streaming Sortformer v2 on CALLHOME part 1
    /// (`diar_streaming_sortformer_4spk-v2_callhome-part1.yaml`).
    pub fn callhome_v2() -> Self {
        Self::new()
            .with_onset(0.641)
            .with_offset(0.561)
            .with_pad_onset(0.229)
            .with_pad_offset(0.079)
            .with_min_duration_on(0.511)
            .with_min_duration_off(0.296)
    }

    /// Opens a streaming filter.
    ///
    /// # Arguments
    /// * `n_speakers` - the speakers each frame scores.
    /// * `frame_seconds` - one frame's length, in seconds.
    pub fn init(
        &self,
        n_speakers: usize,
        frame_seconds: f64,
    ) -> TurnFilter {
        TurnFilter {
            config: self.clone(),
            frame_seconds,
            frames: 0,
            tracks: vec![Track::default(); n_speakers],
        }
    }

    /// The whole-recording form: every turn in `probs`, ordered by start,
    /// then speaker.
    ///
    /// # Arguments
    /// * `probs` - `[frames, n_speakers]`, frame-major.
    /// * `n_speakers` - the speakers each frame scores.
    /// * `frame_seconds` - one frame's length, in seconds.
    pub fn turns(
        &self,
        probs: &[f32],
        n_speakers: usize,
        frame_seconds: f64,
    ) -> Vec<SpeakerTurn> {
        let mut filter = self.init(n_speakers, frame_seconds);
        let mut turns = filter.push(probs);
        turns.extend(filter.finish());
        sort_turns(&mut turns);
        turns
    }
}

/// Orders turns by start, then speaker.
fn sort_turns(turns: &mut [SpeakerTurn]) {
    turns.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.speaker.cmp(&b.speaker)));
}

/// Speaker activity to [`SpeakerTurn`]s, as frames arrive: `NeMo`'s
/// post-processing ([`TurnFilterConfig`]), streamed.
///
/// A turn is emitted once nothing later can change it: once no later speech
/// could merge into it through padding or fill the gap after it. Over a whole
/// recording the turns are exactly [`TurnFilterConfig::turns`]'s, however the
/// frames are split across calls.
#[derive(Clone, Debug)]
pub struct TurnFilter {
    config: TurnFilterConfig,
    frame_seconds: f64,

    /// Frames seen.
    frames: usize,

    tracks: Vec<Track>,
}

/// One speaker's state, through the three stages: binarized runs, padded and
/// merged stretches, then the filters.
#[derive(Clone, Debug, Default)]
struct Track {
    /// The binarized state of the last frame.
    above: bool,

    /// The frame the open run of speech frames started at.
    run_start: Option<usize>,

    /// A padded stretch that a later run could still merge into.
    stretch: Option<(f64, f64)>,

    /// A turn whose following gap could still be filled.
    held: Option<(f64, f64)>,
}

impl TurnFilter {
    /// The config.
    pub fn config(&self) -> &TurnFilterConfig {
        &self.config
    }

    /// Frames seen.
    pub fn frames_seen(&self) -> usize {
        self.frames
    }

    /// Whether `speaker`'s last frame was speech.
    ///
    /// # Panics
    /// If `speaker` is not one of the filter's speakers.
    pub fn is_speaking(
        &self,
        speaker: usize,
    ) -> bool {
        self.tracks[speaker].above
    }

    /// Takes frames in, and returns the turns they finalize, ordered by start.
    ///
    /// # Arguments
    /// * `probs` - `[frames, n_speakers]`, frame-major, following the frames of
    ///   the previous call.
    ///
    /// # Panics
    /// If `probs` is not a whole number of frames.
    pub fn push(
        &mut self,
        probs: &[f32],
    ) -> Vec<SpeakerTurn> {
        let n = self.tracks.len();
        assert!(
            n > 0 && probs.len().is_multiple_of(n),
            "TurnFilter::push: {} probabilities are not frames of {n} speakers",
            probs.len()
        );
        let mut out = Vec::new();
        for frame in probs.chunks_exact(n) {
            for (speaker, &p) in frame.iter().enumerate() {
                self.step(speaker, p, &mut out);
            }
            self.frames += 1;
        }
        for speaker in 0..n {
            self.release(speaker, false, &mut out);
        }
        sort_turns(&mut out);
        out
    }

    /// Ends the input, and returns every turn left, ordered by start.
    pub fn finish(&mut self) -> Vec<SpeakerTurn> {
        let mut out = Vec::new();
        for speaker in 0..self.tracks.len() {
            if let Some(start) = self.tracks[speaker].run_start.take() {
                self.close_run(speaker, start, self.frames, &mut out);
            }
            self.tracks[speaker].above = false;
            self.release(speaker, true, &mut out);
        }
        sort_turns(&mut out);
        out
    }

    /// Binarizes one frame of one speaker, opening or closing a run.
    ///
    /// The comparisons are in `f32`, the probabilities' precision, as `NeMo`
    /// makes them.
    fn step(
        &mut self,
        speaker: usize,
        p: f32,
        out: &mut Vec<SpeakerTurn>,
    ) {
        let (onset, offset) = (self.config.onset as f32, self.config.offset as f32);
        let track = &mut self.tracks[speaker];
        let was = track.above;
        let now = if onset >= offset {
            if p > onset {
                true
            } else if p < offset {
                false
            } else {
                was
            }
        } else if p >= offset {
            true
        } else if p <= onset {
            false
        } else {
            !was
        };
        track.above = now;

        let i = self.frames;
        if now && !was {
            track.run_start = Some(i);
        } else if was && !now {
            let start = track.run_start.take().expect("an open run");
            self.close_run(speaker, start, i, out);
        }
    }

    /// The padded stretch of the run of frames `[start, end)`, merged into the
    /// open stretch or replacing it.
    fn close_run(
        &mut self,
        speaker: usize,
        start: usize,
        end: usize,
        out: &mut Vec<SpeakerTurn>,
    ) {
        let s = self.padded_start(start);
        let e = end as f64 * self.frame_seconds + self.config.pad_offset;
        if e <= s {
            return;
        }
        let track = &mut self.tracks[speaker];
        match &mut track.stretch {
            Some((_, open_end)) if *open_end >= s => *open_end = open_end.max(e),
            _ => {
                if let Some(done) = track.stretch.replace((s, e)) {
                    self.filter(speaker, done, out);
                }
            }
        }
    }

    /// Where a run starting at frame `frame` starts, padded.
    fn padded_start(
        &self,
        frame: usize,
    ) -> f64 {
        (frame as f64 * self.frame_seconds - self.config.pad_onset).max(0.0)
    }

    /// The earliest a stretch not yet formed could start: at the open run's
    /// padded start, or at the next frame's.
    fn next_run_start(
        &self,
        speaker: usize,
    ) -> f64 {
        self.padded_start(self.tracks[speaker].run_start.unwrap_or(self.frames))
    }

    /// Emits what nothing later can change; with `all`, everything.
    fn release(
        &mut self,
        speaker: usize,
        all: bool,
        out: &mut Vec<SpeakerTurn>,
    ) {
        // A stretch is final once no later run can reach it.
        let next = self.next_run_start(speaker);
        if let Some((_, e)) = self.tracks[speaker].stretch
            && (all || next > e)
        {
            let done = self.tracks[speaker].stretch.take().expect("present");
            self.filter(speaker, done, out);
        }

        // A held turn is final once no later turn could start within the
        // gap-fill threshold of its end.
        let next = self.tracks[speaker]
            .stretch
            .map_or(self.next_run_start(speaker), |(s, _)| s);
        if let Some((_, e)) = self.tracks[speaker].held
            && (all || next - e >= self.config.min_duration_off)
        {
            let (s, e) = self.tracks[speaker].held.take().expect("present");
            self.emit_filled(speaker, (s, e), out);
        }
    }

    /// A final stretch through the filters, in the configured order.
    fn filter(
        &mut self,
        speaker: usize,
        stretch: (f64, f64),
        out: &mut Vec<SpeakerTurn>,
    ) {
        if self.config.filter_speech_first && self.too_short(stretch) {
            return;
        }
        if self.config.min_duration_off <= 0.0 {
            self.emit_filled(speaker, stretch, out);
            return;
        }
        let track = &mut self.tracks[speaker];
        match &mut track.held {
            Some((_, held_end)) if stretch.0 - *held_end < self.config.min_duration_off => {
                *held_end = held_end.max(stretch.1)
            }
            _ => {
                if let Some(done) = track.held.replace(stretch) {
                    self.emit_filled(speaker, done, out);
                }
            }
        }
    }

    /// A turn whose gaps are settled: dropped if short and the filters fill
    /// first, emitted otherwise.
    fn emit_filled(
        &self,
        speaker: usize,
        (start, end): (f64, f64),
        out: &mut Vec<SpeakerTurn>,
    ) {
        if !self.config.filter_speech_first && self.too_short((start, end)) {
            return;
        }
        out.push(SpeakerTurn {
            speaker,
            start,
            end,
        });
    }

    fn too_short(
        &self,
        (start, end): (f64, f64),
    ) -> bool {
        self.config.min_duration_on > 0.0 && end - start < self.config.min_duration_on
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        RngExt,
        SeedableRng,
        rngs::StdRng,
    };

    use super::*;

    /// `NeMo`'s `binarization_vectorized`, transliterated over one track: the
    /// reference the streaming filter is held to.
    fn binarize_reference(
        track: &[f32],
        c: &TurnFilterConfig,
        frame: f64,
    ) -> Vec<(f64, f64)> {
        let (onset, offset) = (c.onset as f32, c.offset as f32);
        let mut above = Vec::with_capacity(track.len());
        let mut state = false;
        for &p in track {
            state = if onset >= offset {
                if p > onset {
                    true
                } else if p < offset {
                    false
                } else {
                    state
                }
            } else if p >= offset {
                true
            } else if p <= onset {
                false
            } else {
                !state
            };
            above.push(state);
        }
        let mut segments = Vec::new();
        let mut start = None;
        for (i, &a) in above.iter().chain(std::iter::once(&false)).enumerate() {
            match (start, a) {
                (None, true) => start = Some(i),
                (Some(s), false) => {
                    let s = (s as f64 * frame - c.pad_onset).max(0.0);
                    let e = i as f64 * frame + c.pad_offset;
                    if e > s {
                        segments.push((s, e));
                    }
                    start = None;
                }
                _ => {}
            }
        }
        merge_reference(segments)
    }

    /// `NeMo`'s `merge_overlap_segment`: sorted by start, a segment that
    /// starts at or before the previous one's end joins it.
    fn merge_reference(mut segments: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        segments.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (s, e) in segments {
            match merged.last_mut() {
                Some(last) if last.1 >= s => last.1 = last.1.max(e),
                _ => merged.push((s, e)),
            }
        }
        merged
    }

    /// `NeMo`'s `filtering`.
    fn filter_reference(
        segments: Vec<(f64, f64)>,
        c: &TurnFilterConfig,
    ) -> Vec<(f64, f64)> {
        let drop_short = |s: Vec<(f64, f64)>| {
            if c.min_duration_on > 0.0 {
                s.into_iter()
                    .filter(|(a, b)| b - a >= c.min_duration_on)
                    .collect()
            } else {
                s
            }
        };
        let fill_gaps = |s: Vec<(f64, f64)>| {
            if c.min_duration_off > 0.0 && !s.is_empty() {
                let mut all = s.clone();
                for w in s.windows(2) {
                    if w[1].0 - w[0].1 < c.min_duration_off {
                        all.push((w[0].1, w[1].0));
                    }
                }
                merge_reference(all)
            } else {
                s
            }
        };
        if c.filter_speech_first {
            fill_gaps(drop_short(segments))
        } else {
            drop_short(fill_gaps(segments))
        }
    }

    fn reference_turns(
        probs: &[f32],
        n_speakers: usize,
        c: &TurnFilterConfig,
        frame: f64,
    ) -> Vec<SpeakerTurn> {
        let mut turns = Vec::new();
        for speaker in 0..n_speakers {
            let track: Vec<f32> = probs
                .iter()
                .skip(speaker)
                .step_by(n_speakers)
                .copied()
                .collect();
            for (start, end) in filter_reference(binarize_reference(&track, c, frame), c) {
                turns.push(SpeakerTurn {
                    speaker,
                    start,
                    end,
                });
            }
        }
        sort_turns(&mut turns);
        turns
    }

    /// A track of runs: each speaker drifts between speech and silence, with
    /// jitter around the thresholds.
    fn tracks(
        rng: &mut StdRng,
        frames: usize,
        n_speakers: usize,
    ) -> Vec<f32> {
        let mut levels = vec![0.0f32; n_speakers];
        let mut probs = Vec::with_capacity(frames * n_speakers);
        for _ in 0..frames {
            for level in levels.iter_mut() {
                if rng.random::<f32>() < 0.08 {
                    *level = if *level > 0.5 { 0.1 } else { 0.9 };
                }
                let p = *level + (rng.random::<f32>() - 0.5) * 0.7;
                probs.push(p.clamp(0.0, 1.0));
            }
        }
        probs
    }

    fn configs() -> Vec<TurnFilterConfig> {
        vec![
            TurnFilterConfig::none(),
            TurnFilterConfig::dihard3_v2(),
            TurnFilterConfig::callhome_v2(),
            TurnFilterConfig::callhome_v2().with_filter_speech_first(false),
            TurnFilterConfig::new()
                .with_onset(0.7)
                .with_offset(0.3)
                .with_pad_onset(0.15)
                .with_pad_offset(0.25)
                .with_min_duration_on(0.3)
                .with_min_duration_off(0.4),
        ]
    }

    fn assert_turns_eq(
        got: &[SpeakerTurn],
        want: &[SpeakerTurn],
        what: &str,
    ) {
        assert_eq!(got.len(), want.len(), "{what}: {got:?} != {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.speaker, w.speaker, "{what}");
            assert!(
                (g.start - w.start).abs() < 1e-9 && (g.end - w.end).abs() < 1e-9,
                "{what}: {g:?} != {w:?}"
            );
        }
    }

    /// The whole-recording form is `NeMo`'s post-processing, under every
    /// preset and both filter orders.
    #[test]
    fn test_turns_are_nemo_post_processing() {
        let mut rng = StdRng::seed_from_u64(7);
        for (k, config) in configs().iter().enumerate() {
            for trial in 0..20 {
                let probs = tracks(&mut rng, 300, 3);
                let got = config.turns(&probs, 3, 0.08);
                let want = reference_turns(&probs, 3, config, 0.08);
                assert_turns_eq(&got, &want, &format!("config {k}, trial {trial}"));
            }
        }
    }

    /// Streamed in pieces of any size, the filter emits the whole-recording
    /// turns, and each turn only once it is final.
    #[test]
    fn test_streaming_is_the_whole_recording() {
        let mut rng = StdRng::seed_from_u64(11);
        for (k, config) in configs().iter().enumerate() {
            for trial in 0..20 {
                let probs = tracks(&mut rng, 300, 2);
                let want = config.turns(&probs, 2, 0.01);

                let mut filter = config.init(2, 0.01);
                let mut got = Vec::new();
                let mut at = 0;
                while at < probs.len() / 2 {
                    let n = rng.random_range(1..=17).min(probs.len() / 2 - at);
                    for turn in filter.push(&probs[2 * at..2 * (at + n)]) {
                        // Nothing later may reach back into an emitted turn.
                        let now = (at + n) as f64 * 0.01;
                        assert!(
                            turn.end <= now + config.pad_offset + 1e-9,
                            "{turn:?} at {now}"
                        );
                        got.push(turn);
                    }
                    at += n;
                }
                got.extend(filter.finish());
                sort_turns(&mut got);
                assert_turns_eq(&got, &want, &format!("config {k}, trial {trial}"));
            }
        }
    }

    /// `testdata/diarization/nemo.json`: `NeMo`'s own post-processing of
    /// seeded tracks (`tools/gen_turn_fixtures.py`).
    #[derive(serde::Deserialize)]
    struct NemoFixture {
        turns: Vec<NemoTurnCase>,
    }

    #[derive(serde::Deserialize)]
    struct NemoTurnCase {
        config: String,
        frame_seconds: f64,
        n_speakers: usize,
        probs: Vec<f32>,
        turns: Vec<NemoTurn>,
    }

    #[derive(serde::Deserialize)]
    struct NemoTurn {
        speaker: usize,
        start: f64,
        end: f64,
    }

    fn fixture_config(name: &str) -> TurnFilterConfig {
        match name {
            "none" => TurnFilterConfig::none(),
            "dihard3_v2" => TurnFilterConfig::dihard3_v2(),
            "callhome_v2" => TurnFilterConfig::callhome_v2(),
            "callhome_v2_fill_first" => {
                TurnFilterConfig::callhome_v2().with_filter_speech_first(false)
            }
            "wide" => TurnFilterConfig::new()
                .with_onset(0.7)
                .with_offset(0.3)
                .with_pad_onset(0.15)
                .with_pad_offset(0.25)
                .with_min_duration_on(0.3)
                .with_min_duration_off(0.4),
            other => panic!("no config named {other}"),
        }
    }

    /// The turns are the ones `NeMo` itself computes, to its `f32` times.
    #[test]
    fn test_turns_match_nemo() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/diarization/nemo.json"
        );
        let fixture: NemoFixture =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(!fixture.turns.is_empty());
        for (k, case) in fixture.turns.iter().enumerate() {
            let got = fixture_config(&case.config).turns(
                &case.probs,
                case.n_speakers,
                case.frame_seconds,
            );
            let what = format!("case {k} ({})", case.config);
            assert_eq!(got.len(), case.turns.len(), "{what}: {got:?}");
            for (g, w) in got.iter().zip(&case.turns) {
                assert_eq!(g.speaker, w.speaker, "{what}");
                assert!(
                    (g.start - w.start).abs() < 1e-5 && (g.end - w.end).abs() < 1e-5,
                    "{what}: {g:?} != ({}, {}, {})",
                    w.speaker,
                    w.start,
                    w.end
                );
            }
        }
    }

    /// Hysteresis by hand, including the toggle between the thresholds.
    #[test]
    fn test_hysteresis() {
        let keep = TurnFilterConfig::new().with_onset(0.6).with_offset(0.4);
        // On above 0.6, held through 0.5, off below 0.4.
        let probs = [0.1, 0.7, 0.5, 0.5, 0.3, 0.5, 0.5];
        let turns = keep.turns(&probs, 1, 1.0);
        assert_eq!(
            turns,
            [SpeakerTurn {
                speaker: 0,
                start: 1.0,
                end: 4.0
            }]
        );

        let toggle = TurnFilterConfig::new().with_onset(0.4).with_offset(0.6);
        // Between the thresholds each frame flips the state.
        let probs = [0.5, 0.5, 0.5, 0.1, 0.7, 0.7];
        let turns = toggle.turns(&probs, 1, 1.0);
        let spans: Vec<_> = turns.iter().map(|t| (t.start, t.end)).collect();
        assert_eq!(spans, [(0.0, 1.0), (2.0, 3.0), (4.0, 6.0)]);
    }

    /// Padding merges neighbours, the start clamps at zero, and the filters
    /// drop short turns and fill short gaps.
    #[test]
    fn test_padding_and_filters() {
        let probs = [0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9];
        let c = TurnFilterConfig::new()
            .with_pad_onset(0.5)
            .with_pad_offset(0.6);
        let spans: Vec<_> = c
            .turns(&probs, 1, 1.0)
            .iter()
            .map(|t| (t.start, t.end))
            .collect();
        assert_eq!(spans, [(0.0, 1.6), (2.5, 5.6), (9.5, 11.6)]);

        let c = TurnFilterConfig::new().with_min_duration_on(1.5);
        let spans: Vec<_> = c
            .turns(&probs, 1, 1.0)
            .iter()
            .map(|t| (t.start, t.end))
            .collect();
        assert_eq!(spans, [(3.0, 5.0)]);

        let c = TurnFilterConfig::new().with_min_duration_off(2.5);
        let spans: Vec<_> = c
            .turns(&probs, 1, 1.0)
            .iter()
            .map(|t| (t.start, t.end))
            .collect();
        assert_eq!(spans, [(0.0, 5.0), (10.0, 11.0)]);
    }

    #[test]
    fn test_speaking_state() {
        let mut filter = TurnFilterConfig::none().init(2, 0.08);
        assert!(filter.push(&[0.9, 0.1]).is_empty());
        assert!(filter.is_speaking(0));
        assert!(!filter.is_speaking(1));
        assert_eq!(filter.frames_seen(), 1);
        let turns = filter.finish();
        assert_eq!(turns.len(), 1);
        assert!((turns[0].end - 0.08).abs() < 1e-12);
        assert!(!filter.is_speaking(0));
    }
}
