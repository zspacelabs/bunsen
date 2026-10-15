//! Cutting long audio at quiet points.

/// Cuts `samples` into consecutive windows of at most `max_window` samples,
/// each cut at the quietest `align`-sample block of its window's last sixth.
///
/// A model that attends over its whole input (whose cost grows with the
/// square of its length) or that was trained on bounded utterances takes a
/// long recording in windows; cutting where the signal is quietest keeps a
/// word from straddling the cut. Every cut falls on a multiple of `align`
/// (a model's frame, say), so a window's frames count from its start.
///
/// # Returns
/// `(start, end)` sample ranges that tile `0..samples.len()`, in order; one
/// window when the signal fits in one.
///
/// # Panics
/// If `align` is 0, or `max_window` is shorter than `6 * align`.
pub fn quiet_cuts(
    samples: &[f32],
    max_window: usize,
    align: usize,
) -> Vec<(usize, usize)> {
    assert!(align > 0, "quiet_cuts: align is 0");
    assert!(
        max_window >= 6 * align,
        "quiet_cuts: max_window ({max_window}) is shorter than six blocks of {align}"
    );
    let max_window = max_window / align * align;
    let search = (max_window / 6 / align).max(1) * align;
    let energy = |at: usize| -> f32 { samples[at..at + align].iter().map(|x| x * x).sum() };
    let mut cuts = Vec::new();
    let mut start = 0;
    while samples.len() - start > max_window {
        let end = start + max_window;
        let cut = (end - search..end)
            .step_by(align)
            .min_by(|&a, &b| energy(a).total_cmp(&energy(b)))
            .map_or(end, |at| at + align);
        cuts.push((start, cut));
        start = cut;
    }
    cuts.push((start, samples.len()));
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Windows tile the signal, each cut on a block boundary and at most
    /// `max_window` long, and a cut lands in the silence when there is one.
    #[test]
    fn test_quiet_cuts() {
        let frame = 10;
        let mut samples = vec![1.0f32; 1000];
        samples[540..550].fill(0.0);
        let cuts = quiet_cuts(&samples, 600, frame);
        assert_eq!(cuts.first().unwrap().0, 0);
        assert_eq!(cuts.last().unwrap().1, 1000);
        for w in cuts.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
        for &(a, b) in &cuts {
            assert!(b - a <= 600);
            assert_eq!(a % frame, 0);
        }
        assert_eq!(cuts[0], (0, 550));
        assert_eq!(quiet_cuts(&samples[..300], 600, frame), [(0, 300)]);
    }
}
