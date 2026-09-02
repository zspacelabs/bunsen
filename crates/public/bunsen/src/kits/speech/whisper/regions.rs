//! # Speech regions: spans of samples, and what is done to them.
//!
//! A [`SpeechRegion`] is a half-open span of the stream's samples. The gate
//! produces them raw; three pure functions then shape them the way the
//! references do &mdash; [`pad_regions`] the way `faster-whisper` pads,
//! [`merge_gaps`] the way `fast-whisper-burn` glues neighbours, and
//! [`SpeechRegion::snap_outward`] onto the encoder grid, which neither does
//! and both need.
//!
//! The snap matters because a voice-activity boundary is a multiple of 512
//! samples and a timestamp token can only name a multiple of 320. Since
//! `lcm(160, 320, 512) = 2560`, an unsnapped edge lands on the encoder grid
//! one time in five. Rounding outward &mdash; start down, end up &mdash;
//! keeps the padding conservative and makes a region's start exactly
//! expressible as both a frame index and a timestamp.

use crate::{
    errors::BunsenError,
    kits::speech::whisper::{
        clock::TimestampHistory,
        tokens::TIMESTAMP_STEP_SAMPLES,
    },
    prelude::BunsenResult,
};

/// The encoder frame grid, in samples: one timestamp step.
pub const ENCODER_GRID: usize = TIMESTAMP_STEP_SAMPLES;

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
        parent: &TimestampHistory,
    ) -> TimestampHistory {
        parent.slice(self.start, self.end)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
        for (start, end) in [(0, 1), (512, 1024), (1536, 2048), (319, 321), (640, 640)] {
            let region = r(start, end);
            let snapped = region.snap_outward(ENCODER_GRID);
            assert_eq!(snapped.start % ENCODER_GRID, 0);
            assert_eq!(snapped.end % ENCODER_GRID, 0);
            assert!(snapped.start <= region.start);
            assert!(snapped.end >= region.end);
            assert!(region.start - snapped.start < ENCODER_GRID);
            assert!(snapped.end - region.end < ENCODER_GRID);
        }
        assert_eq!(r(512, 1024).snap_outward(ENCODER_GRID), r(320, 1280));
        assert_eq!(
            r(640, 960).snap_outward(ENCODER_GRID),
            r(640, 960),
            "already on the grid"
        );
    }

    /// A region's clock says the same times the parent's does, from its
    /// own zero.
    #[test]
    fn test_region_clock() {
        let mut parent = TimestampHistory::uniform(16_000);
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
}
