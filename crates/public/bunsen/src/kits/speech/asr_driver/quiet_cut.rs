//! Where to cut a full window that its engine cannot time.

use alloc::vec::Vec;
/// Where a window cut by length is cut before it is decoded, for an engine
/// that cannot time its words: the middle of its quietest `span`-frame
/// stretch, by mean `loudness`, of those whose middle is in the window's last
/// third; the earliest of equals.
///
/// Decoded up to a quiet point, a window neither ends inside a word nor
/// hands the next window a word's tail; a stretch of several frames keeps a
/// stop consonant's gap from passing for a pause.
///
/// # Arguments
/// * `loudness` - a level per frame of the window, louder higher (a frame's
///   mean log-mel, say).
/// * `span` - the frames a quiet stretch must hold, at most the window.
///
/// # Returns
/// A frame in the window's last third; `loudness.len()` for an empty window.
pub fn quiet_cut(
    loudness: &[f32],
    span: usize,
) -> usize {
    let frames = loudness.len();
    let span = span.clamp(1, frames.max(1));
    if frames == 0 {
        return 0;
    }
    let mut sums = Vec::with_capacity(frames + 1);
    sums.push(0.0f64);
    for &x in loudness {
        sums.push(sums.last().unwrap() + x as f64);
    }
    let late = frames * 2 / 3;
    (0..=frames - span)
        .map(|at| (at + span / 2, sums[at + span] - sums[at]))
        .filter(|&(mid, _)| mid >= late)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(frames - 1, |(mid, _)| mid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quiet_cut() {
        // Loud, with a quiet stretch at 70..80 and a quieter one, too early,
        // at 10..20: the late one, at its middle.
        let mut loudness = vec![1.0f32; 100];
        loudness[70..80].fill(0.2);
        loudness[10..20].fill(0.0);
        assert_eq!(quiet_cut(&loudness, 9), 74);
        // A single quiet frame does not outweigh a quieter stretch.
        loudness[90] = -3.0;
        assert_eq!(quiet_cut(&loudness, 9), 74);
        assert_eq!(quiet_cut(&loudness, 1), 90);
        assert_eq!(quiet_cut(&[], 9), 0);
        assert_eq!(quiet_cut(&[0.0; 3], 9), 2);
    }
}
