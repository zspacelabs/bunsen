//! # The speech gate: probabilities in, regions out.
//!
//! Silero emits one speech probability per 512-sample chunk. Turning that
//! into regions is a hysteresis machine: a region **opens** when a
//! probability reaches `threshold`, stays open through anything above
//! `neg_threshold`, and **closes** only after `min_silence` of probabilities
//! below it. The gap between the two thresholds is what stops a region
//! flickering on syllable boundaries; the minimum silence is what stops it
//! splitting between words. Both references implement exactly this machine
//! with different constants, and both are presets here.
//!
//! [`SpeechGate`] is the machine as a streaming fold, one chunk at a time,
//! emitting a raw region when one closes &mdash; the form the driver needs.
//! [`SpeechGateConfig::speech_regions`] is the whole-clip form: the fold, then
//! the padding, and it reproduces `faster-whisper`'s `get_speech_timestamps`
//! exactly, which is what its tests are checked against.

use burn::config::Config;

use crate::kits::speech::whisper::regions::{
    SpeechRegion,
    pad_regions,
};

/// The machine's constants. Names follow `faster-whisper`'s `VadOptions`.
#[derive(Config, Debug, PartialEq)]
pub struct SpeechGateConfig {
    /// The audio sample rate.
    #[config(default = "16_000")]
    pub sample_rate: usize,

    /// The samples per chunk.
    #[config(default = "512")]
    pub chunk_samples: usize,

    /// A probability at or above this opens a region.
    #[config(default = "0.5")]
    pub threshold: f64,

    /// A probability below this counts as silence inside an open region.
    /// Defaults to `max(threshold - 0.15, 0.01)`.
    #[config(default = "None")]
    pub neg_threshold: Option<f64>,

    /// Regions shorter than this are dropped.
    #[config(default = "0")]
    pub min_speech_ms: usize,

    /// Regions longer than this are split: at the longest silence longer
    /// than [`min_silence_at_max_speech_ms`](Self::min_silence_at_max_speech_ms),
    /// or at the last one without
    /// [`split_at_longest_silence`](Self::split_at_longest_silence), or
    /// failing either just before the limit. `None` never splits.
    #[config(default = "None")]
    pub max_speech_s: Option<f64>,

    /// Silence that must elapse before an open region closes.
    #[config(default = "2000")]
    pub min_silence_ms: usize,

    /// Padding added on each side of a region by
    /// [`SpeechGateConfig::speech_regions`].
    #[config(default = "400")]
    pub speech_pad_ms: usize,

    /// A silence at least this long is a candidate split point for a
    /// region that reaches its maximum length.
    #[config(default = "98")]
    pub min_silence_at_max_speech_ms: usize,

    /// Split an over-long region at the longest candidate silence rather
    /// than the last one. Upstream's default, and the better cut: the
    /// longest interior pause beats the most recent one.
    #[config(default = "true")]
    pub split_at_longest_silence: bool,
}

impl SpeechGateConfig {
    fn ms_to_samples(
        &self,
        ms: usize,
    ) -> usize {
        (self.sample_rate * ms) / 1000
    }

    /// [`Self::speech_pad_ms`], in sample count.
    pub fn speech_pad_samples(&self) -> usize {
        self.ms_to_samples(self.speech_pad_ms)
    }

    /// [`Self::min_speech_ms`], in sample count.
    pub fn min_speech_samples(&self) -> usize {
        self.ms_to_samples(self.min_speech_ms)
    }

    /// [`Self::min_silence_ms`], in sample count.
    pub fn min_silence_samples(&self) -> usize {
        self.ms_to_samples(self.min_silence_ms)
    }

    /// [`Self::min_silence_at_max_speech_ms`], in sample count.
    pub fn min_silence_at_max_samples(&self) -> usize {
        self.ms_to_samples(self.min_silence_at_max_speech_ms)
    }

    /// Max speech length before splitting, in sample count.
    pub fn max_speech_samples(&self) -> Option<usize> {
        self.max_speech_s.map(|max_speech_s| {
            (self.sample_rate as f64 * max_speech_s) as usize
                - self.chunk_samples
                - 2 * self.speech_pad_samples()
        })
    }
}

impl SpeechGateConfig {
    /// Initializes a [`SpeechGate`] with these constants.
    pub fn init(&self) -> SpeechGate {
        SpeechGate {
            config: self.clone(),

            index: 0,
            triggered: false,
            start: 0,
            temp_end: 0,
            prev_end: 0,
            next_start: 0,
            possible_ends: Vec::new(),
        }
    }

    /// The whole-clip form: every region in `probs`, padded.
    ///
    /// This is `faster-whisper`'s `get_speech_timestamps` over an already
    /// computed probability track, and agrees with it exactly.
    ///
    /// # Arguments
    /// * `probs` - one speech probability per [`Self::chunk_samples`], covering
    ///   the clip (the last chunk zero-padded, as Silero is fed).
    /// * `total_samples` - the clip's true length, which the last region is
    ///   clamped to.
    pub fn speech_regions(
        &self,
        probs: &[f32],
        total_samples: usize,
    ) -> Vec<SpeechRegion> {
        let mut gate = self.init();
        let mut regions: Vec<SpeechRegion> = probs.iter().filter_map(|&p| gate.step(p)).collect();
        regions.extend(gate.finish(total_samples));
        pad_regions(&mut regions, self.speech_pad_samples(), total_samples);
        regions
    }

    /// `faster-whisper`'s defaults: 2 s of silence to close, 400 ms of
    /// padding. Sentences glue together; first words survive. What
    /// [`new`](Self::new) gives.
    pub fn faster_whisper() -> Self {
        Self::new()
    }

    /// `fast-whisper-burn`'s tuning: 100 ms of silence to close, 30 ms of
    /// padding, regions under 250 ms dropped. Cuts between sentences, so
    /// time-to-first-token is short; neighbours are glued back afterwards
    /// with [`merge_gaps`](super::regions::merge_gaps).
    pub fn fast_whisper_burn() -> Self {
        Self::new()
            .with_min_silence_ms(100)
            .with_speech_pad_ms(30)
            .with_min_speech_ms(250)
    }

    /// The silence threshold, explicit or derived.
    pub fn neg_threshold_value(&self) -> f64 {
        self.neg_threshold
            .unwrap_or_else(|| (self.threshold - 0.15).max(0.01))
    }
}

/// The hysteresis machine as a streaming fold over one probability per chunk.
///
/// Emits a **raw** region when one closes; padding is a separate step, since
/// a region's end pad depends on where the next region starts. A port of
/// `faster-whisper`'s `get_speech_timestamps` loop, including its handling
/// of regions that reach the maximum length.
#[derive(Debug, Clone)]
#[allow(unused)]
pub struct SpeechGate {
    config: SpeechGateConfig,

    /// Chunks seen so far.
    index: usize,
    triggered: bool,
    start: usize,
    /// Where the current silence began; `0` when there is none, as upstream
    /// has it (sample 0 can never begin a silence inside a region).
    temp_end: usize,
    prev_end: usize,
    next_start: usize,
    possible_ends: Vec<(usize, usize)>,
}

impl SpeechGate {
    /// Whether a region is open.
    pub fn is_open(&self) -> bool {
        self.triggered
    }

    /// The sample the open region began at, if one is open.
    pub fn open_since(&self) -> Option<usize> {
        self.triggered.then_some(self.start)
    }

    /// Chunks consumed so far.
    pub fn chunks_seen(&self) -> usize {
        self.index
    }

    /// Resets the gate.
    pub fn reset(&mut self) {
        self.prev_end = 0;
        self.next_start = 0;
        self.temp_end = 0;
        self.possible_ends.clear();
    }

    /// Feeds the next chunk's probability; returns the region it closed,
    /// if it closed one.
    pub fn step(
        &mut self,
        prob: f32,
    ) -> Option<SpeechRegion> {
        let cur = self.config.chunk_samples * self.index;
        self.index += 1;
        let prob = f64::from(prob);

        if prob >= self.config.threshold && self.temp_end != 0 {
            let silence = cur - self.temp_end;
            if silence > self.config.min_silence_at_max_samples() {
                self.possible_ends.push((self.temp_end, silence));
            }
            self.temp_end = 0;
            if self.next_start < self.prev_end {
                self.next_start = cur;
            }
        }

        if prob >= self.config.threshold && !self.triggered {
            self.triggered = true;
            self.start = cur;
            return None;
        }

        let mut closed = None;
        if self.triggered
            && let Some(max_speech_samples) = self.config.max_speech_samples()
            && (cur - self.start) > max_speech_samples
        {
            if self.config.split_at_longest_silence && !self.possible_ends.is_empty() {
                // The longest silence; the first of equals, as upstream's
                // `max` picks.
                let (end, duration) = *self
                    .possible_ends
                    .iter()
                    .rev()
                    .max_by_key(|(_, duration)| *duration)
                    .expect("not empty");
                closed = Some(SpeechRegion::new(self.start, end));
                let next_start = end + duration;
                if next_start < end + cur {
                    self.start = next_start;
                } else {
                    self.triggered = false;
                }
                self.reset();
            } else if self.prev_end != 0 {
                closed = Some(SpeechRegion::new(self.start, self.prev_end));
                if self.next_start < self.prev_end {
                    self.triggered = false;
                } else {
                    self.start = self.next_start;
                }
                self.reset();
            } else {
                closed = Some(SpeechRegion::new(self.start, cur));
                self.triggered = false;
                self.reset();
                return closed;
            }
        }

        if prob < self.config.neg_threshold_value() && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = cur;
            }
            let silence = cur - self.temp_end;

            if !self.config.split_at_longest_silence
                && silence > self.config.min_silence_at_max_samples()
            {
                self.prev_end = self.temp_end;
            }

            if silence < self.config.min_silence_samples() {
                return closed;
            }

            let end = self.temp_end;
            let region = ((end - self.start) > self.config.min_speech_samples())
                .then(|| SpeechRegion::new(self.start, end));
            self.triggered = false;
            self.reset();
            return closed.or(region);
        }

        closed
    }

    /// Ends the stream at `total_samples`, closing an open region if it is
    /// long enough to keep.
    pub fn finish(
        self,
        total_samples: usize,
    ) -> Option<SpeechRegion> {
        (self.triggered
            && (total_samples.saturating_sub(self.start)) > self.config.min_speech_samples())
        .then(|| SpeechRegion::new(self.start, total_samples.max(self.start)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: usize = 512;

    /// `n` chunks at probability `p`.
    fn run(
        p: f32,
        n: usize,
    ) -> Vec<f32> {
        vec![p; n]
    }

    fn track(parts: &[(f32, usize)]) -> Vec<f32> {
        parts.iter().flat_map(|&(p, n)| run(p, n)).collect()
    }

    fn regions(pairs: &[(usize, usize)]) -> Vec<SpeechRegion> {
        pairs
            .iter()
            .map(|&(s, e)| SpeechRegion::new(s, e))
            .collect()
    }

    /// 100 ms of silence to close (3.125 chunks), 30 ms of padding (480
    /// samples): short enough that every branch fits in a few dozen chunks.
    fn quick() -> SpeechGateConfig {
        SpeechGateConfig::new()
            .with_min_silence_ms(100)
            .with_speech_pad_ms(30)
    }

    #[test]
    fn test_presets_and_derived_threshold() {
        let fw = SpeechGateConfig::faster_whisper();
        assert_eq!(fw, SpeechGateConfig::new());
        assert_eq!(fw.min_silence_ms, 2000);
        assert_eq!(fw.speech_pad_ms, 400);
        assert_eq!(fw.speech_pad_samples(), 6400);
        assert!((fw.neg_threshold_value() - 0.35).abs() < 1e-12);

        let burn = SpeechGateConfig::fast_whisper_burn();
        assert_eq!(burn.min_silence_ms, 100);
        assert_eq!(burn.speech_pad_ms, 30);
        assert_eq!(burn.min_speech_ms, 250);

        let low = SpeechGateConfig::new().with_threshold(0.1);
        assert!(
            (low.neg_threshold_value() - 0.01).abs() < 1e-12,
            "floored at 0.01"
        );
        let explicit = SpeechGateConfig::new().with_neg_threshold(Some(0.2));
        assert!((explicit.neg_threshold_value() - 0.2).abs() < 1e-12);
    }

    // Every expectation below was produced by running `faster-whisper`'s
    // own `get_speech_timestamps` over the same probability track, with its
    // model replaced by the track. They are the reference's answers, not
    // this port's.

    #[test]
    fn test_one_region() {
        let t = track(&[(0.1, 3), (0.9, 10), (0.1, 10)]);
        let total_samples = 23 * C;
        let config = quick();
        assert_eq!(
            config.speech_regions(&t, total_samples),
            regions(&[(1056, 7136)])
        );
    }

    /// A dip shorter than `min_silence` does not split; one long enough
    /// does.
    #[test]
    fn test_min_silence_decides_a_split() {
        let short = track(&[(0.1, 3), (0.9, 10), (0.2, 2), (0.9, 10), (0.1, 10)]);
        let total_samples = 35 * C;
        let config = quick();
        assert_eq!(
            config.speech_regions(&short, total_samples),
            regions(&[(1056, 13280)])
        );

        let long = track(&[(0.1, 3), (0.9, 10), (0.2, 4), (0.9, 10), (0.1, 10)]);
        let total_samples = 37 * C;
        let config = quick();
        assert_eq!(
            config.speech_regions(&long, total_samples),
            regions(&[(1056, 14304)])
        );
    }

    /// The hysteresis band: between `neg_threshold` and `threshold` keeps an
    /// open region open, and never opens a closed one.
    #[test]
    fn test_hysteresis_band() {
        let keeps = track(&[(0.1, 3), (0.9, 2), (0.4, 8), (0.1, 10)]);
        let total_samples = 23 * C;
        let config = quick();
        assert_eq!(
            config.speech_regions(&keeps, total_samples),
            regions(&[(1056, 7136)])
        );

        let never = run(0.4, 20);
        let total_samples = 20 * C;
        let config = quick();
        assert!(config.speech_regions(&never, total_samples).is_empty());
    }

    #[test]
    fn test_min_speech_drops_short_regions() {
        let t = track(&[(0.1, 3), (0.9, 3), (0.1, 10), (0.9, 10), (0.1, 10)]);
        let config = quick().with_min_speech_ms(250);
        let total_samples = 36 * C;
        assert_eq!(
            config.speech_regions(&t, total_samples),
            regions(&[(7712, 13792)])
        );
    }

    /// A region open at the end of the stream closes there, at the true
    /// length rather than the padded one.
    #[test]
    fn test_end_of_stream_closes() {
        let t = track(&[(0.1, 3), (0.9, 10)]);
        let total_samples = 13 * C;
        let config = quick();
        assert_eq!(
            config.speech_regions(&t, total_samples),
            regions(&[(1056, 6656)])
        );
        let total_samples = 13 * C - 100;
        let config = quick();
        assert_eq!(
            config.speech_regions(&t, total_samples),
            regions(&[(1056, 6556)])
        );
    }

    /// An over-long region with no silence to split at is cut just before
    /// the limit, and speech resumes as a new region at once.
    #[test]
    fn test_max_speech_cuts_without_a_silence() {
        let t = track(&[(0.1, 2), (0.9, 30), (0.1, 5)]);
        let config = quick().with_max_speech_s(Some(0.5));
        let total_samples = 37 * C;
        assert_eq!(
            config.speech_regions(&t, total_samples),
            regions(&[(544, 7936), (7936, 15104), (15104, 16864)])
        );
    }

    /// With a qualifying silence behind it, the cut lands there instead:
    /// at the longest one, by default.
    #[test]
    fn test_max_speech_cuts_at_the_longest_silence() {
        let t = track(&[(0.1, 2), (0.9, 8), (0.2, 4), (0.9, 30), (0.1, 5)]);
        let config = SpeechGateConfig::new()
            .with_min_silence_ms(300)
            .with_speech_pad_ms(30)
            .with_max_speech_s(Some(1.0));
        let total_samples = 49 * C;
        assert_eq!(
            config.speech_regions(&t, total_samples),
            regions(&[(544, 5600), (6688, 22496)])
        );

        // Two candidates: the longer one wins, wherever it is.
        let t = track(&[
            (0.1, 2),
            (0.9, 6),
            (0.2, 5),
            (0.9, 4),
            (0.2, 4),
            (0.9, 30),
            (0.1, 5),
        ]);
        let total_samples = 56 * C;
        assert_eq!(
            config.speech_regions(&t, total_samples),
            regions(&[(544, 4576), (6176, 21760), (21760, 28672)])
        );
    }

    /// Asked for the last silence instead: a dip too short to qualify is
    /// ignored and the cut lands just before the limit; a qualifying one is
    /// used; of two qualifying ones, the later wins.
    #[test]
    fn test_max_speech_cuts_at_the_last_silence_when_asked() {
        let config = SpeechGateConfig::new()
            .with_min_silence_ms(300)
            .with_speech_pad_ms(30)
            .with_max_speech_s(Some(1.0))
            .with_split_at_longest_silence(false);

        let short_dip = track(&[(0.1, 2), (0.9, 8), (0.2, 4), (0.9, 30), (0.1, 5)]);
        let total_samples = 49 * C;
        assert_eq!(
            config.speech_regions(&short_dip, total_samples),
            regions(&[(544, 16128), (16128, 25088)])
        );

        let one_dip = track(&[(0.1, 2), (0.9, 8), (0.2, 5), (0.9, 30), (0.1, 5)]);
        let total_samples = 50 * C;
        assert_eq!(
            config.speech_regions(&one_dip, total_samples),
            regions(&[(544, 5600), (7200, 23008)])
        );

        let two_dips = track(&[
            (0.1, 2),
            (0.9, 6),
            (0.2, 5),
            (0.9, 4),
            (0.2, 5),
            (0.9, 30),
            (0.1, 5),
        ]);
        let total_samples = 57 * C;
        assert_eq!(
            config.speech_regions(&two_dips, total_samples),
            regions(&[(544, 9184), (10784, 26592)])
        );
    }

    /// Padding: clamped at the start, split across a narrow gap, full
    /// across a wide one, clamped at the end.
    #[test]
    fn test_padding() {
        let t = track(&[
            (0.9, 6),
            (0.1, 2),
            (0.9, 6),
            (0.1, 6),
            (0.9, 6),
            (0.1, 4),
            (0.9, 4),
        ]);
        let config = SpeechGateConfig::new()
            .with_min_silence_ms(40)
            .with_speech_pad_ms(60);
        let total_samples = 34 * C;
        assert_eq!(
            config.speech_regions(&t, total_samples),
            regions(&[(0, 8128), (9280, 14272), (14400, 17408)])
        );
    }

    /// `faster-whisper`'s own defaults: two seconds of silence to close.
    #[test]
    fn test_defaults() {
        let split = track(&[(0.1, 5), (0.9, 40), (0.1, 70), (0.9, 40), (0.1, 30)]);
        let total_samples = 185 * C;
        let config = &SpeechGateConfig::new();
        assert_eq!(
            config.speech_regions(&split, total_samples),
            regions(&[(0, 29440), (52480, 94720)])
        );

        let glued = track(&[(0.1, 5), (0.9, 40), (0.1, 50), (0.9, 40), (0.1, 70)]);
        let total_samples = 205 * C;
        let config = &SpeechGateConfig::new();
        assert_eq!(
            config.speech_regions(&glued, total_samples),
            regions(&[(0, 75520)])
        );
    }

    /// The streaming fold says the same as the whole-clip form, one chunk
    /// at a time, and reports its state on the way.
    #[test]
    fn test_streaming_fold_matches_whole_clip() {
        let t = track(&[(0.1, 3), (0.9, 10), (0.2, 4), (0.9, 10), (0.1, 10)]);
        let config = quick();

        let mut gate = config.init();
        let mut raw = Vec::new();
        let mut opened_at = None;
        for (i, &p) in t.iter().enumerate() {
            assert_eq!(gate.chunks_seen(), i as usize);
            let was_open = gate.is_open();
            if let Some(region) = gate.step(p) {
                raw.push(region);
            }
            if !was_open && gate.is_open() {
                opened_at.get_or_insert(gate.open_since());
            }
        }
        assert_eq!(opened_at, Some(Some(3 * C)));
        raw.extend(gate.finish(37 * C));

        let mut padded = raw.clone();
        pad_regions(&mut padded, config.speech_pad_samples(), 37 * C);
        let total_samples = 37 * C;
        assert_eq!(padded, config.speech_regions(&t, total_samples));
        // Four silent chunks are 1536 samples of elapsed silence, under the
        // 1600 that closes a region, so the dip does not split it.
        assert_eq!(raw, regions(&[(1536, 13824)]));
    }

    /// Against the real front end: Silero over four seconds of the moon
    /// speech, cut to hold a one-second pause. The quick tuning splits at the
    /// pause; the default tuning, with two seconds of patience, does not.
    #[cfg(feature = "silero-weights")]
    mod silero {
        use burn::{
            Tensor,
            prelude::TensorData,
        };

        use super::*;
        use crate::{
            kits::speech::silero_vad::{
                SileroVad,
                SileroVadContextConfig,
                SileroVadMeta,
            },
            support::{
                audio::load_audio_mono_sr,
                testing::CpuBackend,
            },
        };

        type B = CpuBackend;

        /// Silero's probabilities for the clip, one per chunk.
        fn probabilities() -> (Vec<f32>, usize) {
            let device = Default::default();
            let vad = SileroVad::<B>::load_16khz_pretrained(&device).unwrap();
            let path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/testdata/audio/jfk_moon_4s.mp3"
            );
            let sample_rate: usize = 16_000;

            let mut wav = load_audio_mono_sr(path, sample_rate).unwrap();
            let total = wav.len() as usize;

            let chunk = vad.chunk_size();
            assert_eq!(chunk as usize, 512);
            let tail = wav.len() % chunk;
            if tail != 0 {
                wav.resize(wav.len() + chunk - tail, 0.0);
            }
            let steps = wav.len() / chunk;
            let chunks: Tensor<B, 3> =
                Tensor::from_data(TensorData::new(wav, [steps, 1, chunk]), &device);

            let context = SileroVadContextConfig::new(sample_rate).init(&vad, &device);
            let (probs, _) = vad.context_forward_sequence(chunks, context);
            let probs: Vec<f32> = probs.to_data().convert::<f32>().to_vec().unwrap();
            assert_eq!(probs.len(), steps);
            (probs, total)
        }

        #[test]
        fn test_golden_track() {
            let (probs, total) = probabilities();

            let config = &SpeechGateConfig::fast_whisper_burn();
            let quick = config.speech_regions(&probs, total);
            assert_eq!(total, 64_000, "4.0 s at 16 kHz");
            // "...the highest mountain?" to 1.02 s, the pause, then
            // "Why, thirty-five years ago..." from 1.86 s.
            assert_eq!(quick, regions(&[(0, 16_352), (29_728, 56_800)]));

            let config = &SpeechGateConfig::new();
            let patient = config.speech_regions(&probs, total);
            assert_eq!(
                patient.len(),
                1,
                "two seconds of patience glue the pause: {patient:?}"
            );
            assert_eq!(patient[0].start, 0);
            assert_eq!(patient[0].end, 64_000);
        }
    }
}
