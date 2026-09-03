use burn::config::Config;

use crate::{
    errors::{
        BunsenError,
        BunsenResult,
    },
    kits::speech::whisper::driver::StreamClock,
};

/// A half-open span of samples, `start..end`, in the stream's own sample
/// index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeechRegion {
    /// First sample of the region.
    pub start: usize,
    /// One past the last sample of the region.
    pub end: usize,
}

impl SpeechRegion {
    /// A region from `start` up to `end`.
    ///
    /// # Panics
    /// If `end < start`.
    pub fn new(
        start: usize,
        end: usize,
    ) -> Self {
        assert!(end >= start, "region end {end} is before its start {start}");
        Self { start, end }
    }

    /// Samples in the region.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the region has no samples.
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }

    /// Shift the region forward by `offset`.
    pub fn offset(
        self,
        offset: usize,
    ) -> Self {
        Self {
            start: self.start + offset,
            end: self.end + offset,
        }
    }

    /// Scale the region by `factor`.
    pub fn scale(
        self,
        factor: usize,
    ) -> Self {
        Self {
            start: self.start * factor,
            end: self.end * factor,
        }
    }

    /// The region widened onto a grid: start rounded down, end rounded up.
    ///
    /// The end is not clamped to the stream; a region that runs past the
    /// audio is decoded against silence, which is what the padding was
    /// for anyway.
    pub fn snap_outward(
        &self,
        grid: usize,
    ) -> Self {
        assert_ne!(grid, 0, "a grid needs a non-zero step");
        Self {
            start: self.start / grid * grid,
            end: self.end.div_ceil(grid) * grid,
        }
    }

    /// The region's own clock: the parent stream's, sliced so that the
    /// region's sample 0 is its `start`.
    pub fn clock(
        &self,
        parent: &StreamClock,
    ) -> StreamClock {
        parent.slice(self.start, self.end)
    }
}

#[cfg(any(test, debug_assertions))]
fn assert_region_sequence(regions: &[SpeechRegion]) -> BunsenResult<()> {
    for w in regions.windows(2) {
        let prev = &w[0];
        let next = &w[1];
        if prev.end > next.start {
            return Err(BunsenError::Invalid(format!(
                "region {:?} ends after region {:?}: {:?}",
                prev, next, regions
            )));
        }
    }
    Ok(())
}

/// Pads regions outward by `pad` samples.
///
/// The first start and the last end are clamped to the stream. Between two
/// regions, a gap narrower than `2 * pad` is split down the middle rather
/// than letting the pads overlap; a wider one gives each side its full pad.
///
/// # Arguments
/// * `regions` - raw regions, in order, non-overlapping.
/// * `pad` - samples to add on each side.
/// * `total` - samples in the stream.
pub fn pad_regions(
    regions: &mut [SpeechRegion],
    pad: usize,
    total: usize,
) {
    #[cfg(any(test, debug_assertions))]
    assert_region_sequence(regions).unwrap();

    let n = regions.len();
    for i in 0..n {
        if i == 0 {
            regions[0].start = regions[0].start.saturating_sub(pad);
        }
        if i + 1 < n {
            let silence = regions[i + 1].start - regions[i].end;
            if silence < 2 * pad {
                regions[i].end += silence / 2;
                regions[i + 1].start = regions[i + 1].start.saturating_sub(silence / 2);
            } else {
                regions[i].end = total.min(regions[i].end + pad);
                regions[i + 1].start = regions[i + 1].start.saturating_sub(pad);
            }
        } else {
            regions[i].end = total.min(regions[i].end + pad);
        }
    }
}

/// Merges regions where `next.start - prev.end <= gap`.
pub fn merge_gaps(
    regions: &[SpeechRegion],
    gap: usize,
) -> Vec<SpeechRegion> {
    #[cfg(any(test, debug_assertions))]
    assert_region_sequence(regions).unwrap();

    let mut out: Vec<SpeechRegion> = Vec::with_capacity(regions.len());
    for &region in regions {
        match out.last_mut() {
            Some(last) if region.start.saturating_sub(last.end) <= gap => {
                last.end = last.end.max(region.end);
            }
            _ => out.push(region),
        }
    }
    out
}

/// Config for [`VoiceActivityFilter`].
#[derive(Config, Debug, PartialEq)]
pub struct VoiceActivityFilterConfig {
    /// The audio sample rate.
    #[config(default = "16_000")]
    pub sample_rate: usize,

    /// The samples per chunk.
    #[config(default = "512")]
    pub samples_per_chunk: usize,

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
    /// [`VoiceActivityFilterConfig::speech_regions`].
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

impl Default for VoiceActivityFilterConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl VoiceActivityFilterConfig {
    /// The silence threshold, explicit or derived.
    pub fn neg_threshold_value(&self) -> f64 {
        self.neg_threshold
            .unwrap_or_else(|| (self.threshold - 0.15).max(0.01))
    }

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
                - self.samples_per_chunk
                - 2 * self.speech_pad_samples()
        })
    }

    /// Initializes a [`VoiceActivityFilter`] with these constants.
    pub fn init(&self) -> VoiceActivityFilter {
        VoiceActivityFilter {
            config: self.clone(),

            chunk_count: 0,
            open_speech: false,
            start: 0,
            temp_end: None,
            possible_ends: Vec::new(),
        }
    }

    /// The whole-clip form: every region in `probs`, padded.
    ///
    /// This is `faster-whisper`'s `get_speech_timestamps` over an already
    /// computed probability track, and agrees with it exactly.
    ///
    /// # Arguments
    /// * `probs` - one speech probability per [`Self::samples_per_chunk`],
    ///   covering the clip (the last chunk zero-padded, as Silero is fed).
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
    /// with [`merge_gaps`](merge_gaps).
    pub fn fast_whisper_burn() -> Self {
        Self::new()
            .with_min_silence_ms(100)
            .with_speech_pad_ms(30)
            .with_min_speech_ms(250)
    }
}

/// Hysteresis filter for voice activity.
#[derive(Debug, Clone)]
#[allow(unused)]
pub struct VoiceActivityFilter {
    config: VoiceActivityFilterConfig,

    /// Chunks seen so far.
    chunk_count: usize,

    open_speech: bool,
    start: usize,

    /// Where the current silence began, if one is running.
    temp_end: Option<usize>,

    /// Where a region that reaches its maximum length may be cut: the
    /// silences inside it longer than `min_silence_at_max`, as
    /// `(start, end)`, `end` being where speech resumed. All of them under
    /// the longest-silence policy, only the latest under the last-silence
    /// policy; the cut is the longest of the list either way.
    possible_ends: Vec<(usize, usize)>,
}

impl VoiceActivityFilter {
    /// Get the filter's config.
    pub fn config(&self) -> &VoiceActivityFilterConfig {
        &self.config
    }

    /// Chunks consumed so far.
    pub fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    /// The sample count so far.
    pub fn sample_count(&self) -> usize {
        self.config.samples_per_chunk * self.chunk_count
    }

    /// Whether a region is open.
    pub fn is_open(&self) -> bool {
        self.open_speech
    }

    /// The sample the open region began at, if one is open.
    pub fn open_since(&self) -> Option<usize> {
        self.open_speech.then_some(self.start)
    }

    fn reset_possible_endings(&mut self) {
        self.temp_end = None;
        self.possible_ends.clear();
    }

    /// Feeds the next chunk's probability; returns the region it closed,
    /// if it closed one.
    pub fn step(
        &mut self,
        prob: f32,
    ) -> Option<SpeechRegion> {
        let prob = prob as f64;
        let cur = self.sample_count();

        self.chunk_count += 1;

        if prob >= self.config.threshold {
            // This is active speech region.

            if let Some(temp_end) = self.temp_end.take() {
                // A silence was running; it ends here.
                // The silence length from the temp_end to now:
                let silence = cur - temp_end;
                if silence > self.config.min_silence_at_max_samples() {
                    // This is a potential cut-point for a long speech region.
                    if !self.config.split_at_longest_silence {
                        // The last-silence policy keeps only the latest.
                        self.possible_ends.clear();
                    }
                    self.possible_ends.push((temp_end, cur));
                }
            }

            if !self.open_speech {
                // Open a region here.
                self.open_speech = true;
                self.start = cur;
                return None;
            }
        }

        let mut result: Option<SpeechRegion> = None;

        if self.open_speech
            && let Some(max_speech_samples) = self.config.max_speech_samples()
            && (cur - self.start) > max_speech_samples
        {
            // The open region has reached its maximum length; it is cut
            // at the longest candidate silence: the first of equals, as
            // upstream's `max` picks. Under the last-silence policy the
            // list holds one, so it is that one.

            // Select the longest silence with the earliest index.
            let longest = self
                .possible_ends
                .iter()
                .copied()
                .enumerate()
                .max_by_key(|&(idx, (start, end))| (end - start, -(idx as isize)));

            if let Some((idx, (end, resume))) = longest {
                // Cut at that silence; the region continues from where
                // speech resumed after it.
                result = Some(SpeechRegion::new(self.start, end));
                self.start = resume;

                // Reset the soft ending history to after the new cut.
                if let Some(temp_end) = self.temp_end
                    && temp_end <= self.start
                {
                    self.temp_end = None;
                }
                self.possible_ends.drain(..=idx);
            } else {
                // No silence to cut at: cut here, and close the gate.
                result = Some(SpeechRegion::new(self.start, cur));
                self.open_speech = false;
                self.reset_possible_endings();
                return result;
            }
        }

        if prob < self.config.neg_threshold_value() && self.open_speech {
            // This is silence inside an open region.

            // The silence begins here, unless it is already running.
            let temp_end = *self.temp_end.get_or_insert(cur);
            // The silence length from the temp_end to now:
            let silence = cur - temp_end;

            if silence < self.config.min_silence_samples() {
                // Not yet enough silence to close the region.
                return result;
            }

            // Enough silence. The region ends where the silence began,
            // and is kept only if it is long enough.
            let region = ((temp_end - self.start) > self.config.min_speech_samples())
                .then(|| SpeechRegion::new(self.start, temp_end));
            self.open_speech = false;
            self.reset_possible_endings();
            // Never both: a cut above that left the gate open has just
            // reset the silence, so this chunk can at most begin a new one.
            return result.or(region);
        }

        result
    }

    /// Ends the stream at `total_samples`, closing an open region if it is
    /// long enough to keep.
    pub fn finish(
        self,
        total_samples: usize,
    ) -> Option<SpeechRegion> {
        (self.open_speech
            && (total_samples.saturating_sub(self.start)) > self.config.min_speech_samples())
        .then(|| SpeechRegion::new(self.start, total_samples.max(self.start)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: usize = 512;

    fn r(
        start: usize,
        end: usize,
    ) -> SpeechRegion {
        SpeechRegion::new(start, end)
    }

    #[test]
    fn test_region_basics() {
        let region = r(1000, 2600);
        assert_eq!(region.len(), 1600);
        assert!(!region.is_empty());
        assert!(r(5, 5).is_empty());
    }

    #[test]
    fn test_offset() {
        let region = r(1000, 2600);
        assert_eq!(region.offset(100), r(1100, 2700));
    }

    #[test]
    fn test_scale() {
        let region = r(1000, 2600);
        assert_eq!(region.scale(2), r(2000, 5200));
    }

    #[test]
    #[should_panic(expected = "before its start")]
    fn test_region_rejects_inverted() {
        let _ = r(10, 5);
    }

    /// Start down, end up, onto the 320-sample grid: every snapped edge is a
    /// multiple of 320, and the region only ever grows.
    #[test]
    fn test_snap_outward() {
        /// The encoder grid at 16 kHz.
        const GRID: usize = 320;
        for (start, end) in [(0, 1), (512, 1024), (1536, 2048), (319, 321), (640, 640)] {
            let region = r(start, end);
            let snapped = region.snap_outward(GRID);
            assert_eq!(snapped.start % GRID, 0);
            assert_eq!(snapped.end % GRID, 0);
            assert!(snapped.start <= region.start);
            assert!(snapped.end >= region.end);
            assert!(region.start - snapped.start < GRID);
            assert!(snapped.end - region.end < GRID);
        }
        assert_eq!(r(512, 1024).snap_outward(GRID), r(320, 1280));
        assert_eq!(
            r(640, 960).snap_outward(GRID),
            r(640, 960),
            "already on the grid"
        );
    }

    /// A region's clock says the same times the parent's does, from its
    /// own zero.
    #[test]
    fn test_region_clock() {
        let mut parent = StreamClock::uniform(16_000);
        parent.anchor(16_000, 10.0).unwrap();

        let region = r(8_000, 40_000);
        let clock = region.clock(&parent);
        for s in [0, 8_000, 16_000, 31_999] {
            assert!((clock.time_at(s) - parent.time_at(region.start + s)).abs() < 1e-9);
        }
    }

    /// `faster-whisper`'s padding, in its three cases: clamped at the
    /// edges, split down the middle of a narrow gap, full on a wide one.
    #[test]
    fn test_pad_regions() {
        let mut regions = vec![r(100, 1000), r(1400, 2000), r(4000, 5000)];
        pad_regions(&mut regions, 300, 5100);
        assert_eq!(
            regions,
            vec![
                r(0, 1200),    // clamped start; narrow gap (400 < 600) split: +200
                r(1200, 2300), // -200 from the split; wide gap (2000): +300
                r(3700, 5100), // -300; clamped end
            ]
        );

        let mut none: Vec<SpeechRegion> = vec![];
        pad_regions(&mut none, 300, 5100);
        assert!(none.is_empty());
    }

    #[test]
    fn test_merge_gaps() {
        let regions = [r(0, 1000), r(1100, 2000), r(2500, 3000), r(3000, 3500)];
        assert_eq!(merge_gaps(&regions, 200), vec![r(0, 2000), r(2500, 3500)]);
        assert_eq!(
            merge_gaps(&regions, 0),
            vec![r(0, 1000), r(1100, 2000), r(2500, 3500)]
        );
        assert_eq!(merge_gaps(&regions, 10_000), vec![r(0, 3500)]);
        assert!(merge_gaps(&[], 200).is_empty());
    }
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
    fn quick() -> VoiceActivityFilterConfig {
        VoiceActivityFilterConfig::new()
            .with_min_silence_ms(100)
            .with_speech_pad_ms(30)
    }

    #[test]
    fn test_presets_and_derived_threshold() {
        let fw = VoiceActivityFilterConfig::faster_whisper();
        assert_eq!(fw, VoiceActivityFilterConfig::new());
        assert_eq!(fw.min_silence_ms, 2000);
        assert_eq!(fw.speech_pad_ms, 400);
        assert_eq!(fw.speech_pad_samples(), 6400);
        assert!((fw.neg_threshold_value() - 0.35).abs() < 1e-12);

        let burn = VoiceActivityFilterConfig::fast_whisper_burn();
        assert_eq!(burn.min_silence_ms, 100);
        assert_eq!(burn.speech_pad_ms, 30);
        assert_eq!(burn.min_speech_ms, 250);

        assert_float_eq::assert_float_relative_eq!(
            VoiceActivityFilterConfig::new()
                .with_threshold(0.165) // (0.165 - 0.15) = 0.015 > 0.01
                .neg_threshold_value(),
            0.015
        );
        assert_float_eq::assert_float_relative_eq!(
            VoiceActivityFilterConfig::new()
                .with_threshold(0.1) // (0.1 - 0.15) < 0.01
                .neg_threshold_value(),
            0.01
        );
        assert_float_eq::assert_float_relative_eq!(
            VoiceActivityFilterConfig::new()
                .with_neg_threshold(Some(0.2))
                .neg_threshold_value(),
            0.2
        );
    }

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
        let config = VoiceActivityFilterConfig::new()
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
            regions(&[(544, 4576), (6176, 9184), (10272, 26080)])
        );
    }

    /// Asked for the last silence instead: a dip too short to qualify is
    /// ignored and the cut lands just before the limit; a qualifying one is
    /// used; of two qualifying ones, the later wins.
    #[test]
    fn test_max_speech_cuts_at_the_last_silence_when_asked() {
        let config = VoiceActivityFilterConfig::new()
            .with_min_silence_ms(300)
            .with_speech_pad_ms(30)
            .with_max_speech_s(Some(1.0))
            .with_split_at_longest_silence(false);

        let short_dip = track(&[(0.1, 2), (0.9, 8), (0.2, 3), (0.9, 30), (0.1, 5)]);
        let total_samples = 48 * C;
        assert_eq!(
            config.speech_regions(&short_dip, total_samples),
            regions(&[(544, 16128), (16128, 24576)])
        );

        // A four-chunk dip qualifies, as it does under the longest policy.
        // Upstream's legacy path tests the silence a chunk earlier, while
        // it is still running, and misses this one: it cuts just before
        // the limit instead.
        let four_chunk_dip = track(&[(0.1, 2), (0.9, 8), (0.2, 4), (0.9, 30), (0.1, 5)]);
        let total_samples = 49 * C;
        assert_eq!(
            config.speech_regions(&four_chunk_dip, total_samples),
            regions(&[(544, 5600), (6688, 22496)])
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
        let config = VoiceActivityFilterConfig::new()
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
        let config = &VoiceActivityFilterConfig::new();
        assert_eq!(
            config.speech_regions(&split, total_samples),
            regions(&[(0, 29440), (52480, 94720)])
        );

        let glued = track(&[(0.1, 5), (0.9, 40), (0.1, 50), (0.9, 40), (0.1, 70)]);
        let total_samples = 205 * C;
        let config = &VoiceActivityFilterConfig::new();
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
            assert_eq!(gate.chunk_count(), i as usize);
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

            let config = &VoiceActivityFilterConfig::fast_whisper_burn();
            let quick = config.speech_regions(&probs, total);
            assert_eq!(total, 64_000, "4.0 s at 16 kHz");
            // "...the highest mountain?" to 1.02 s, the pause, then
            // "Why, thirty-five years ago..." from 1.86 s.
            assert_eq!(quick, regions(&[(0, 16_352), (29_728, 56_800)]));

            let config = &VoiceActivityFilterConfig::new();
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
