//! Timing a Qwen3-ASR reply from its decoder's attention onto the audio.
//!
//! Qwen3-ASR does not time its output, but each reply token is predicted
//! under self-attention that reaches the window's audio placeholders, one per
//! encoder frame, and some heads follow the speech as the reply is written:
//! the frame such a head attends to most is near where the token's word is
//! spoken. Averaged over a few such heads and aligned monotonically, the
//! attention times each token to within an encoder frame or two, a steady
//! lag after the word starts.
//!
//! The heads are a property of the checkpoint, measured against a timed
//! reference (`qwen3-asr-model-validation`'s `alignment` tests, against
//! Parakeet's word times).
use burn::Tensor;

use super::{
    Qwen3AsrBundle,
    blocks::Qwen3AsrAudioEncoderMeta,
};
use crate::errors::BunsenResult;

/// The decoder heads, `(layer, head)`, whose attention onto the audio
/// follows the speech in `Qwen3-ASR-0.6B` and `FermionResearch`'s Phonon-1
/// builds, which re-pack its weights.
///
/// Averaged and aligned ([`Qwen3AsrBundle::token_frames`]), they put 90% of
/// the words of `jfk_moon.mp3` within 0.20 s of Parakeet's word times, on
/// the base model and on each Phonon-1 build, once a constant offset is
/// taken out: the attention peaks 0.12 s after Parakeet times a word.
pub const QWEN3_ASR_0_6B_ALIGNMENT_HEADS: [(usize, usize); 8] = [
    (25, 10),
    (18, 5),
    (22, 7),
    (23, 14),
    (25, 5),
    (24, 15),
    (16, 14),
    (19, 5),
];

/// The mel frames between encoder frames within a chunk: three stride-2
/// convolutions.
const MEL_FRAMES_PER_ENCODER_FRAME: usize = 8;

/// Aligns `rows` to `cols` monotonically, through `score` (row-major, higher
/// is better): the path from the first cell to the last, moving down, right
/// or both, of the highest total score.
///
/// # Returns
/// Each row's stretch of the path, its first and last column; the stretches
/// are in order, each starting at or after the last one's end.
///
/// # Panics
/// If `score` is not `rows * cols` long.
pub fn monotonic_spans(
    score: &[f64],
    rows: usize,
    cols: usize,
) -> Vec<(usize, usize)> {
    assert_eq!(score.len(), rows * cols, "a {rows}x{cols} score");
    let at = |i: usize, j: usize| i * (cols + 1) + j;
    let mut cost = vec![f64::INFINITY; (rows + 1) * (cols + 1)];
    // 0: diagonal, 1: down, 2: right.
    let mut step = vec![0u8; (rows + 1) * (cols + 1)];
    cost[0] = 0.0;
    for i in 1..=rows {
        for j in 1..=cols {
            let (diag, down, right) = (
                cost[at(i - 1, j - 1)],
                cost[at(i - 1, j)],
                cost[at(i, j - 1)],
            );
            let (best, s) = if diag <= down && diag <= right {
                (diag, 0)
            } else if down <= right {
                (down, 1)
            } else {
                (right, 2)
            };
            cost[at(i, j)] = best - score[(i - 1) * cols + (j - 1)];
            step[at(i, j)] = s;
        }
    }
    let mut spans = vec![(0, 0); rows];
    let mut seen = vec![false; rows];
    let (mut i, mut j) = (rows, cols);
    while i > 0 && j > 0 {
        let span = &mut spans[i - 1];
        if !seen[i - 1] {
            *span = (j - 1, j - 1);
            seen[i - 1] = true;
        }
        span.0 = j - 1;
        match step[at(i, j)] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
    }
    spans
}

/// The column each row is spoken at: its highest score within its stretch
/// of the [monotonic alignment](monotonic_spans), the earliest of equals.
///
/// A row's stretch can take in a pause before it, where nothing is said, so
/// its first column can be early; its peak is where the attention settles.
pub fn monotonic_peaks(
    score: &[f64],
    rows: usize,
    cols: usize,
) -> Vec<usize> {
    monotonic_spans(score, rows, cols)
        .into_iter()
        .enumerate()
        .map(|(i, (first, last))| {
            let row = &score[i * cols..(i + 1) * cols];
            (first..=last)
                .rev()
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .expect("a non-empty stretch")
        })
        .collect()
}

impl Qwen3AsrBundle {
    /// The encoder frame each token of a reply is spoken at: `heads`' rows
    /// of [`audio_attention`](Self::audio_attention), each renormalized over
    /// the audio, averaged, aligned to the audio monotonically, and each
    /// token at its peak within its stretch ([`monotonic_peaks`]).
    ///
    /// # Arguments
    /// * `features`, `valid` - as
    ///   [`transcribe_features`](Self::transcribe_features) took them.
    /// * `reply` - the tokens the decode generated for them.
    /// * `heads` - `(layer, head)` pairs, such as
    ///   [`QWEN3_ASR_0_6B_ALIGNMENT_HEADS`].
    ///
    /// # Returns
    /// A frame per token of `reply`, non-decreasing.
    ///
    /// # Errors
    /// As [`audio_attention`](Self::audio_attention).
    ///
    /// # Panics
    /// If `reply` or `heads` is empty, or a head is past the model's.
    pub fn token_frames(
        &self,
        features: Tensor<2>,
        valid: usize,
        reply: &[usize],
        heads: &[(usize, usize)],
    ) -> BunsenResult<Vec<usize>> {
        assert!(!heads.is_empty(), "no heads to align with");
        let mut layers: Vec<usize> = heads.iter().map(|&(layer, _)| layer).collect();
        layers.sort_unstable();
        layers.dedup();
        let weights = self.audio_attention(features, valid, reply, &layers)?;
        let picked: Vec<Tensor<2>> = heads
            .iter()
            .map(|&(layer, head)| {
                let w = &weights[layers.binary_search(&layer).expect("a requested layer")];
                let [rows, cols] = [w.dims()[1], w.dims()[2]];
                let w = w.clone().slice_dim(0, head..head + 1).reshape([rows, cols]);
                w.clone() / w.sum_dim(1).clamp_min(1e-9)
            })
            .collect();
        let mean = Tensor::stack::<3>(picked, 0).mean_dim(0);
        let [_, rows, cols] = mean.dims();
        let score: Vec<f64> = mean
            .into_data()
            .convert::<f64>()
            .try_to_vec()
            .expect("f64 data");
        Ok(monotonic_peaks(&score, rows, cols))
    }

    /// The mel frame, of the window the audio was encoded from, that encoder
    /// frame `frame` starts on: each chunk of the encoder's mel frames yields
    /// its encoder frames eight mel frames apart.
    pub fn encoder_frame_mel(
        &self,
        frame: usize,
    ) -> usize {
        let tower = &self.model.audio_tower;
        let per_chunk = tower.chunk_out();
        frame / per_chunk * tower.chunk_frames() + frame % per_chunk * MEL_FRAMES_PER_ENCODER_FRAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_spans_and_peaks() {
        // Three rows, each peaked on its own column, in order: the path
        // takes the first row along to the second's peak.
        let score = [
            1.0, 0.5, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let spans = monotonic_spans(&score, 3, 4);
        assert_eq!(spans, [(0, 1), (2, 2), (3, 3)]);
        assert_eq!(monotonic_peaks(&score, 3, 4), [0, 2, 3]);

        // A row after a pause: its stretch takes in the pause, and its peak
        // is past it.
        let score = [
            1.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.1, 0.1, 0.9, 0.0, //
            0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let spans = monotonic_spans(&score, 3, 5);
        assert!(spans[1].0 < 3 && spans[1].1 >= 3, "{spans:?}");
        assert_eq!(monotonic_peaks(&score, 3, 5), [0, 3, 4]);

        // A row peaked before its predecessor's peak stays at or after it.
        let score = [
            0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let peaks = monotonic_peaks(&score, 3, 3);
        assert!(peaks.windows(2).all(|w| w[0] <= w[1]), "{peaks:?}");
    }
}
