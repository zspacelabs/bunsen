//! # The stream clock: sample index to media time.
//!
//! Every stream carries a [`TimestampHistory`]: sorted `(sample, time)`
//! anchors plus a sample rate. A bare stream gets one anchor at `(0, 0.0)`,
//! which reproduces exactly the arithmetic upstream does from its seek
//! pointer. Everything richer &mdash; a capture callback's timestamp, a
//! container's presentation time, a dropped buffer becoming a new anchor
//! rather than a permanent shift &mdash; is an addition to that, not a
//! departure. Making the general case the only case costs nothing, because
//! the bare case *is* the general case with one anchor.
//!
//! A timestamp token resolves as `clock.time_at(window_origin + index *
//! 320)`; a region decoded as its own stream keeps correct absolute times
//! through [`slice`](TimestampHistory::slice).

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// A point where the stream's sample index is known in media time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    /// The sample index, counted from the start of the stream.
    pub sample: usize,
    /// The media time of that sample, in seconds.
    pub time: f64,
}

/// A stream's map from sample index to media time.
///
/// Between anchors, time advances at the sample rate; before the first and
/// after the last it extrapolates at the sample rate too. Anchors are kept
/// sorted by sample and are never removed.
#[derive(Debug, Clone, PartialEq)]
pub struct TimestampHistory {
    rate: usize,
    anchors: Vec<Anchor>,
}

impl TimestampHistory {
    /// A clock that starts at time zero and runs at `rate` samples per
    /// second: what a bare stream is given.
    ///
    /// # Panics
    /// If `rate` is zero.
    pub fn uniform(rate: usize) -> Self {
        assert_ne!(rate, 0, "a clock needs a non-zero sample rate");
        Self {
            rate,
            anchors: vec![Anchor {
                sample: 0,
                time: 0.0,
            }],
        }
    }

    /// Samples per second.
    pub fn rate(&self) -> usize {
        self.rate
    }

    /// The anchors, sorted by sample; never empty.
    pub fn anchors(&self) -> &[Anchor] {
        &self.anchors
    }

    /// Records that `sample` was observed at media time `time`.
    ///
    /// Anchors must arrive in sample order. Re-anchoring the same sample
    /// replaces its time, which is how a stream corrects its origin before
    /// any audio has been placed against it.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if `sample` is before the last anchor, or if
    /// `time` is not finite.
    pub fn anchor(
        &mut self,
        sample: usize,
        time: f64,
    ) -> BunsenResult<()> {
        if !time.is_finite() {
            return Err(BunsenError::Invalid(format!(
                "anchor time must be finite, got {time}"
            )));
        }

        let last = self.anchors.last().expect("never empty");
        if sample < last.sample {
            return Err(BunsenError::Invalid(format!(
                "anchors must not go backwards: sample {sample} after {}",
                last.sample,
            )));
        }
        if sample == last.sample {
            self.anchors.pop();
        }

        self.anchors.push(Anchor { sample, time });
        Ok(())
    }

    /// The media time of `sample`.
    ///
    /// From the last anchor at or before `sample`, advancing at the rate;
    /// from the first anchor, backwards, for a sample before it.
    pub fn time_at(
        &self,
        sample: usize,
    ) -> f64 {
        let at = self.anchors.partition_point(|a| a.sample <= sample);
        let base = self.anchors[at.saturating_sub(1)];

        // Signed, so a sample before the first anchor extrapolates backwards.
        let delta = sample as f64 - base.sample as f64;
        base.time + delta / self.rate as f64
    }

    /// The clock of a sub-stream whose sample 0 is this stream's `from`.
    ///
    /// Carries every anchor inside `from..to`, shifted, and opens with an
    /// anchor at 0 for `time_at(from)`, so that for every sample `s` in the
    /// slice, `slice.time_at(s) == self.time_at(from + s)`. This is how a
    /// speech region decoded as its own stream keeps correct absolute times.
    ///
    /// # Panics
    /// If `to < from`.
    pub fn slice(
        &self,
        from: usize,
        to: usize,
    ) -> Self {
        assert!(to >= from, "slice end {to} is before its start {from}");

        let mut anchors = vec![Anchor {
            sample: 0,
            time: self.time_at(from),
        }];
        anchors.extend(
            self.anchors
                .iter()
                .filter(|a| a.sample > from && a.sample < to)
                .map(|a| Anchor {
                    sample: a.sample - from,
                    time: a.time,
                }),
        );

        Self {
            rate: self.rate,
            anchors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(
        a: f64,
        b: f64,
    ) -> bool {
        (a - b).abs() < 1e-9
    }

    /// A bare clock is `sample / rate`, everywhere.
    #[test]
    fn test_uniform_is_sample_over_rate() {
        let clock = TimestampHistory::uniform(16_000);
        assert_eq!(clock.rate(), 16_000);
        assert_eq!(clock.anchors().len(), 1);

        for sample in [0, 1, 160, 16_000, 48_000, 10_000_000] {
            assert!(close(clock.time_at(sample), sample as f64 / 16_000.0));
        }
    }

    /// An anchor shifts everything after it, and only after it.
    #[test]
    fn test_anchor_shifts_from_there_on() {
        let mut clock = TimestampHistory::uniform(16_000);

        // A capture callback says sample 32000 was at 12.5 s: the stream lost
        // half a second somewhere before that.
        clock.anchor(32_000, 12.5).unwrap();

        assert!(
            close(clock.time_at(16_000), 1.0),
            "before the anchor: unchanged"
        );
        assert!(close(clock.time_at(32_000), 12.5));
        assert!(
            close(clock.time_at(48_000), 13.5),
            "after: advances from the anchor"
        );
    }

    #[test]
    fn test_anchor_before_the_first_extrapolates_backwards() {
        let mut clock = TimestampHistory::uniform(16_000);
        clock.anchor(0, 100.0).unwrap();
        assert!(
            close(clock.time_at(0), 100.0),
            "re-anchoring sample 0 replaces it"
        );
        assert_eq!(clock.anchors().len(), 1);

        let mut late = TimestampHistory {
            rate: 16_000,
            anchors: vec![Anchor {
                sample: 16_000,
                time: 5.0,
            }],
        };
        assert!(close(late.time_at(0), 4.0));
        late.anchor(16_000, 6.0).unwrap();
        assert!(close(late.time_at(0), 5.0));
    }

    #[test]
    fn test_anchors_stay_monotone() {
        let mut clock = TimestampHistory::uniform(16_000);
        clock.anchor(1_000, 1.0).unwrap();

        assert!(clock.anchor(999, 2.0).is_err(), "backwards in sample");
        assert!(clock.anchor(2_000, f64::NAN).is_err());
        assert!(clock.anchor(2_000, f64::INFINITY).is_err());
        assert_eq!(
            clock.anchors().len(),
            2,
            "a rejected anchor leaves no trace"
        );

        // Time may go backwards (a re-synced source); samples may not.
        clock.anchor(2_000, 0.5).unwrap();
        assert!(
            clock
                .anchors()
                .windows(2)
                .all(|w| w[0].sample <= w[1].sample)
        );
    }

    /// `slice(from, to).time_at(s) == time_at(from + s)` for every sample
    /// inside the slice, anchors or not. Past its end the slice knows
    /// nothing of the parent's later anchors, which is the point: a region
    /// decoded as its own stream does not see the future.
    #[test]
    fn test_slice_round_trips() {
        let mut clock = TimestampHistory::uniform(16_000);
        clock.anchor(10_000, 3.0).unwrap();
        clock.anchor(40_000, 9.0).unwrap();

        for (from, to) in [
            (0, 5_000),
            (5_000, 20_000),
            (10_000, 50_000),
            (30_000, 30_001),
        ] {
            let sliced = clock.slice(from, to);
            assert_eq!(sliced.rate(), clock.rate());
            assert_eq!(sliced.anchors()[0].sample, 0);

            for s in [0, 1, 320, 4_999, 12_000, 25_000]
                .into_iter()
                .filter(|&s| from + s < to)
            {
                assert!(
                    close(sliced.time_at(s), clock.time_at(from + s)),
                    "from {from} to {to}, sample {s}",
                );
            }
        }

        // Anchors strictly inside the slice come along, shifted; the edges
        // do not, because the opening anchor already says what they say.
        let sliced = clock.slice(5_000, 40_000);
        assert_eq!(sliced.anchors().len(), 2);
        assert_eq!(sliced.anchors()[1].sample, 5_000);
        assert!(close(sliced.anchors()[1].time, 3.0));
    }

    /// The timestamp-token resolution the driver will do: a token at step
    /// `i` of a window that began at sample `origin` is at `origin + 320 i`.
    #[test]
    fn test_timestamp_token_resolution() {
        let mut clock = TimestampHistory::uniform(16_000);
        clock.anchor(0, 60.0).unwrap();

        let origin = 48_000; // a window cut 3 s into the stream
        assert!(close(clock.time_at(origin + 100 * 320), 60.0 + 3.0 + 2.0));
    }
}
