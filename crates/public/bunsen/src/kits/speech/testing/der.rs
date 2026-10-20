//! Speaker-diarization test helpers: diarization error rate.
//!
//! A diarizer names its speakers by its own indices, a reference by whatever
//! its annotators wrote, so the two are compared under the one-to-one mapping
//! of names that makes them agree longest, as NIST's `md-eval` and
//! `pyannote.metrics` compare them. A speaker-attributed transcript is scored
//! by words instead:
//! [`cp_word_error_rate`](crate::kits::speech::testing::wer::cp_word_error_rate).

use crate::kits::speech::diarization::RttmSegment;

/// The parts of a diarization error rate, in seconds of scored time.
#[derive(Clone, Debug, PartialEq)]
pub struct DiarizationError {
    /// The scored reference speech: the sum over the scored time of the
    /// number of reference speakers talking.
    pub total: f64,

    /// Reference speech with too few hypothesis speakers.
    pub missed: f64,

    /// Hypothesis speech with too few reference speakers.
    pub false_alarm: f64,

    /// Speech attributed to the wrong speaker under the mapping.
    pub confusion: f64,
}

impl DiarizationError {
    /// `(missed + false_alarm + confusion) / total`; 0 when there is no
    /// reference speech and no error, and infinite when there is only error.
    pub fn der(&self) -> f64 {
        let errors = self.missed + self.false_alarm + self.confusion;
        if self.total > 0.0 {
            errors / self.total
        } else if errors > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

/// The diarization error of `hypothesis` against `reference`, one recording
/// each (the segments' `file` is not read).
///
/// Time within `collar` seconds of a reference segment's start or end is not
/// scored (`md-eval`'s collar, which is half of `pyannote.metrics`'
/// `collar`); with `skip_overlap`, neither is time where two or more
/// reference speakers talk. Speakers are mapped one to one, maximizing the
/// scored time on which the pairs agree.
pub fn diarization_error(
    reference: &[RttmSegment],
    hypothesis: &[RttmSegment],
    collar: f64,
    skip_overlap: bool,
) -> DiarizationError {
    let ref_names = names(reference);
    let hyp_names = names(hypothesis);

    let mut cuts: Vec<f64> = Vec::new();
    for s in reference.iter().chain(hypothesis) {
        cuts.extend([s.start, s.end()]);
    }
    let mut no_score: Vec<(f64, f64)> = Vec::new();
    if collar > 0.0 {
        for s in reference {
            for t in [s.start, s.end()] {
                no_score.push((t - collar, t + collar));
                cuts.extend([t - collar, t + collar]);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();

    // Each elementary interval: its length, and who talks in it.
    let active = |segments: &[RttmSegment], names: &[String], t: f64| -> Vec<usize> {
        let mut who: Vec<usize> = segments
            .iter()
            .filter(|s| s.start <= t && t < s.end())
            .map(|s| names.iter().position(|n| *n == s.speaker).expect("named"))
            .collect();
        who.sort_unstable();
        who.dedup();
        who
    };
    let mut intervals: Vec<(f64, Vec<usize>, Vec<usize>)> = Vec::new();
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mid = 0.5 * (a + b);
        if b <= a || no_score.iter().any(|&(s, e)| s <= mid && mid < e) {
            continue;
        }
        let r = active(reference, &ref_names, mid);
        if skip_overlap && r.len() > 1 {
            continue;
        }
        let h = active(hypothesis, &hyp_names, mid);
        intervals.push((b - a, r, h));
    }

    // The mapping that maximizes agreement.
    let mut overlap = vec![vec![0.0; hyp_names.len()]; ref_names.len()];
    for (d, r, h) in &intervals {
        for &i in r {
            for &j in h {
                overlap[i][j] += d;
            }
        }
    }
    let mapping = max_weight_assignment(&overlap);

    let mut out = DiarizationError {
        total: 0.0,
        missed: 0.0,
        false_alarm: 0.0,
        confusion: 0.0,
    };
    for (d, r, h) in &intervals {
        let (nr, nh) = (r.len(), h.len());
        let correct = r
            .iter()
            .filter(|&&i| mapping[i].is_some_and(|j| h.contains(&j)))
            .count();
        out.total += d * nr as f64;
        out.missed += d * nr.saturating_sub(nh) as f64;
        out.false_alarm += d * nh.saturating_sub(nr) as f64;
        out.confusion += d * (nr.min(nh) - correct) as f64;
    }
    out
}

/// The distinct speaker names, in order of first appearance.
fn names(segments: &[RttmSegment]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for s in segments {
        if !names.contains(&s.speaker) {
            names.push(s.speaker.clone());
        }
    }
    names
}

/// The one-to-one assignment of rows to columns that maximizes the summed
/// weight: each row's column, `None` for a row left unassigned. The
/// Hungarian algorithm, on the matrix padded square with zeros.
fn max_weight_assignment(weights: &[Vec<f64>]) -> Vec<Option<usize>> {
    let rows = weights.len();
    let cols = weights.first().map_or(0, |r| r.len());
    let n = rows.max(cols);
    if n == 0 {
        return vec![None; rows];
    }
    let max = weights.iter().flatten().copied().fold(0.0f64, f64::max);
    let cost = |i: usize, j: usize| {
        if i < rows && j < cols {
            max - weights[i][j]
        } else {
            max
        }
    };

    // Potentials and matching, 1-indexed, column 0 the sentinel.
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let mut matched = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        matched[0] = i;
        let mut j0 = 0;
        let mut min_to = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = matched[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=n {
                if !used[j] {
                    let c = cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if c < min_to[j] {
                        min_to[j] = c;
                        way[j] = j0;
                    }
                    if min_to[j] < delta {
                        delta = min_to[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[matched[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_to[j] -= delta;
                }
            }
            j0 = j1;
            if matched[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            matched[j0] = matched[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut out = vec![None; rows];
    for (j, &i) in matched.iter().enumerate().skip(1) {
        if (1..=rows).contains(&i) && j <= cols {
            out[i - 1] = Some(j - 1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(
        speaker: &str,
        start: f64,
        end: f64,
    ) -> RttmSegment {
        RttmSegment {
            file: "f".into(),
            channel: 1,
            start,
            duration: end - start,
            speaker: speaker.into(),
        }
    }

    fn close(
        a: f64,
        b: f64,
    ) -> bool {
        (a - b).abs() < 1e-9
    }

    /// Identical turns under other names score zero.
    #[test]
    fn test_names_do_not_matter() {
        let reference = [seg("A", 0.0, 2.0), seg("B", 2.0, 5.0), seg("A", 6.0, 7.0)];
        let hypothesis = [
            seg("speaker_1", 0.0, 2.0),
            seg("speaker_0", 2.0, 5.0),
            seg("speaker_1", 6.0, 7.0),
        ];
        let e = diarization_error(&reference, &hypothesis, 0.0, false);
        assert!(close(e.total, 6.0));
        assert_eq!(e.der(), 0.0);
    }

    /// Each kind of error, by hand.
    #[test]
    fn test_missed_false_alarm_confusion() {
        let reference = [seg("A", 0.0, 4.0), seg("B", 4.0, 8.0)];

        let e = diarization_error(&reference, &[], 0.0, false);
        assert!(close(e.missed, 8.0) && close(e.der(), 1.0));

        // One speaker for the whole: B's half is confusion.
        let e = diarization_error(&reference, &[seg("x", 0.0, 8.0)], 0.0, false);
        assert!(close(e.confusion, 4.0) && close(e.missed, 0.0));
        assert!(close(e.der(), 0.5));

        // Speech past the reference is a false alarm.
        let e = diarization_error(
            &reference,
            &[seg("x", 0.0, 4.0), seg("y", 4.0, 10.0)],
            0.0,
            false,
        );
        assert!(close(e.false_alarm, 2.0) && close(e.confusion, 0.0));
        assert!(close(e.der(), 0.25));
    }

    /// Overlap counts every speaker; `skip_overlap` leaves it unscored.
    #[test]
    fn test_overlap() {
        let reference = [seg("A", 0.0, 4.0), seg("B", 2.0, 6.0)];
        let hypothesis = [seg("x", 0.0, 4.0), seg("y", 4.0, 6.0)];
        let e = diarization_error(&reference, &hypothesis, 0.0, false);
        assert!(close(e.total, 8.0));
        assert!(close(e.missed, 2.0));
        assert!(close(e.der(), 0.25));

        let e = diarization_error(&reference, &hypothesis, 0.0, true);
        assert!(close(e.total, 4.0));
        assert_eq!(e.der(), 0.0);
    }

    /// The collar leaves boundary disagreement unscored.
    #[test]
    fn test_collar() {
        let reference = [seg("A", 1.0, 3.0)];
        let hypothesis = [seg("x", 1.2, 2.9)];
        let e = diarization_error(&reference, &hypothesis, 0.0, false);
        assert!(close(e.missed, 0.3));

        let e = diarization_error(&reference, &hypothesis, 0.25, false);
        assert!(close(e.total, 1.5));
        assert_eq!(e.der(), 0.0);
    }

    /// `testdata/diarization/nemo.json`: `NeMo`'s md-eval scores of seeded
    /// turns (`tools/gen_turn_fixtures.py`).
    #[derive(serde::Deserialize)]
    struct NemoFixture {
        der: Vec<NemoDerCase>,
    }

    #[derive(serde::Deserialize)]
    struct NemoDerCase {
        reference: Vec<(f64, f64, String)>,
        hypothesis: Vec<(f64, f64, String)>,
        collar: f64,
        skip_overlap: bool,
        der: f64,
        confusion: f64,
        false_alarm: f64,
        missed: f64,
    }

    /// The error rates are the ones `NeMo`'s md-eval computes, with and
    /// without a collar, with and without overlap.
    #[test]
    fn test_matches_nemo_md_eval() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/diarization/nemo.json"
        );
        let fixture: NemoFixture =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(!fixture.der.is_empty());
        let segments = |turns: &[(f64, f64, String)]| -> Vec<RttmSegment> {
            turns.iter().map(|(s, e, n)| seg(n, *s, *e)).collect()
        };
        for (k, case) in fixture.der.iter().enumerate() {
            let e = diarization_error(
                &segments(&case.reference),
                &segments(&case.hypothesis),
                case.collar,
                case.skip_overlap,
            );
            let what = format!(
                "case {k} (collar {}, skip_overlap {})",
                case.collar, case.skip_overlap
            );
            for (name, got, want) in [
                ("der", e.der(), case.der),
                ("confusion", e.confusion / e.total, case.confusion),
                ("false_alarm", e.false_alarm / e.total, case.false_alarm),
                ("missed", e.missed / e.total, case.missed),
            ] {
                assert!((got - want).abs() < 1e-3, "{what}: {name} {got} != {want}");
            }
        }
    }

    #[test]
    fn test_assignment_is_optimal() {
        // Greedy would take (0, 0) = 5 and then (1, 1) = 0; the best is 4 + 4.
        let w = vec![vec![5.0, 4.0], vec![4.0, 0.0]];
        assert_eq!(max_weight_assignment(&w), [Some(1), Some(0)]);

        // More rows than columns: one row goes unassigned.
        let w = vec![vec![1.0], vec![3.0], vec![2.0]];
        assert_eq!(max_weight_assignment(&w), [None, Some(0), None]);
        assert!(max_weight_assignment(&[]).is_empty());
    }
}
