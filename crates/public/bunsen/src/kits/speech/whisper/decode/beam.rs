//! # Beam search, as upstream does it.
//!
//! Rows are laid out `row = audio_idx * k + beam_idx`, beam varying fastest.
//! Every step, each beam proposes its `k + 1` best next tokens; the
//! `k * (k + 1)` candidates of one audio are deduplicated **by full
//! sequence** (at the first step every beam is the same prompt, and without
//! this the beam would quietly collapse to width one while still paying for
//! `k`), ranked by cumulative log probability, and the best `k` that did not
//! end survive, with the self-attention cache permuted to follow them. A
//! candidate that ended joins the audio's finished set, which `patience`
//! caps at `round(k * patience)`; the search completes when every audio's
//! set is full.
//!
//! Log probabilities accumulate in `f32`, as upstream's do, so that a
//! near-tie ranks the same way.

use std::collections::HashMap;

use burn::{
    Tensor,
    prelude::Backend,
    tensor::activation::log_softmax,
};

use crate::kits::speech::whisper::decode::TokenDecoder;

/// A finished set: insertion-ordered, unique by sequence, as a Python dict.
#[derive(Debug, Clone, Default)]
struct Finished {
    entries: Vec<(Vec<i64>, f32)>,
}

impl Finished {
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Overwrites the score of a sequence already present, in place;
    /// appends otherwise.
    fn upsert(
        &mut self,
        sequence: Vec<i64>,
        score: f32,
    ) {
        match self.entries.iter_mut().find(|(s, _)| *s == sequence) {
            Some(entry) => entry.1 = score,
            None => self.entries.push((sequence, score)),
        }
    }
}

/// Upstream's `BeamSearchDecoder`.
#[derive(Debug, Clone)]
pub struct BeamSearchDecoder {
    k: usize,
    eot: i64,
    max_candidates: usize,
    /// One per audio, once the first update has said how many there are.
    finished: Option<Vec<Finished>>,
}

impl BeamSearchDecoder {
    /// # Arguments
    /// * `beam_size` - beams per audio; at least 1.
    /// * `eot` - the stop token.
    /// * `patience` - how many finished candidates to collect before stopping,
    ///   as a multiple of the beam size; `None` is 1.
    ///
    /// # Panics
    /// If `round(beam_size * patience)` is zero.
    pub fn new(
        beam_size: usize,
        eot: i64,
        patience: Option<f64>,
    ) -> Self {
        assert!(beam_size >= 1, "a beam search needs at least one beam");
        let patience = patience.unwrap_or(1.0);
        let max_candidates = (beam_size as f64 * patience).round() as usize;
        assert!(
            max_candidates > 0,
            "invalid beam size ({beam_size}) or patience ({patience})"
        );
        Self {
            k: beam_size,
            eot,
            max_candidates,
            finished: None,
        }
    }

    /// Beams per audio.
    pub fn beam_size(&self) -> usize {
        self.k
    }

    /// Finished candidates collected per audio before the search stops.
    pub fn max_candidates(&self) -> usize {
        self.max_candidates
    }
}

impl<B: Backend> TokenDecoder<B> for BeamSearchDecoder {
    fn group_size(&self) -> usize {
        self.k
    }

    fn reset(&mut self) {
        self.finished = None;
    }

    fn update(
        &mut self,
        tokens: &mut Vec<Vec<i64>>,
        logits: Tensor<B, 2>,
        sum_logprobs: &mut [f32],
        reorder: &mut dyn FnMut(&[usize]),
    ) -> (Vec<i64>, bool) {
        let k = self.k;
        let rows = tokens.len();
        assert_eq!(
            rows % k,
            0,
            "{rows} rows is not a whole number of {k}-beam groups"
        );
        let n_audio = rows / k;
        let finished = self
            .finished
            .get_or_insert_with(|| vec![Finished::default(); n_audio]);

        // Each row's k + 1 best next tokens, with their log probabilities.
        let (values, indices) = log_softmax(logits, 1).topk_with_indices(k + 1, 1);
        let values: Vec<f32> = values.into_data().convert::<f32>().to_vec().unwrap();
        let indices: Vec<i64> = indices.into_data().convert::<i64>().to_vec().unwrap();

        let mut next: Vec<Vec<i64>> = Vec::with_capacity(rows);
        let mut next_sums: Vec<f32> = Vec::with_capacity(rows);
        let mut sources: Vec<usize> = Vec::with_capacity(rows);
        let mut newly_finished: Vec<Finished> = Vec::with_capacity(n_audio);

        for audio in 0..n_audio {
            // STEP 1: every candidate, deduplicated by full sequence. A
            // repeat keeps its first position and takes the later score and
            // source, as assigning into a dict does.
            let mut candidates: Vec<(Vec<i64>, f32, usize)> = Vec::with_capacity(k * (k + 1));
            let mut position: HashMap<Vec<i64>, usize> = HashMap::with_capacity(k * (k + 1));
            for beam in 0..k {
                let row = audio * k + beam;
                for c in 0..=k {
                    let token = indices[row * (k + 1) + c];
                    let score = sum_logprobs[row] + values[row * (k + 1) + c];
                    let mut sequence = tokens[row].clone();
                    sequence.push(token);
                    match position.get(&sequence) {
                        Some(&at) => {
                            candidates[at].1 = score;
                            candidates[at].2 = row;
                        }
                        None => {
                            position.insert(sequence.clone(), candidates.len());
                            candidates.push((sequence, score, row));
                        }
                    }
                }
            }

            // STEP 2: rank, and keep the best k that did not end. Stable, so
            // equals keep their first-seen order, as a sorted dict does.
            let mut order: Vec<usize> = (0..candidates.len()).collect();
            order.sort_by(|&a, &b| candidates[b].1.total_cmp(&candidates[a].1));

            let mut done = Finished::default();
            let mut saved = 0;
            for &c in &order {
                let (sequence, score, source) = &candidates[c];
                if sequence.last() == Some(&self.eot) {
                    done.upsert(sequence.clone(), *score);
                } else {
                    next_sums.push(*score);
                    next.push(sequence.clone());
                    sources.push(*source);
                    saved += 1;
                    if saved == k {
                        break;
                    }
                }
            }
            assert_eq!(saved, k, "fewer than {k} live candidates for audio {audio}");
            newly_finished.push(done);
        }

        reorder(&sources);
        *tokens = next;
        sum_logprobs.copy_from_slice(&next_sums);

        // Newly finished sequences join the audio's set, best first, until
        // it is full.
        for (previously, newly) in finished.iter_mut().zip(newly_finished) {
            let mut entries = newly.entries;
            entries.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (sequence, score) in entries {
                if previously.len() >= self.max_candidates {
                    break;
                }
                previously.upsert(sequence, score);
            }
        }

        let completed = finished.iter().all(|f| f.len() >= self.max_candidates);
        let feed = tokens
            .iter()
            .map(|s| *s.last().expect("non-empty"))
            .collect();
        (feed, completed)
    }

    fn finalize(
        &mut self,
        tokens: Vec<Vec<i64>>,
        sum_logprobs: Vec<f32>,
        prompt_len: usize,
    ) -> Vec<Vec<(Vec<i64>, f32)>> {
        let k = self.k;
        let n_audio = tokens.len() / k;
        let mut finished = self
            .finished
            .take()
            .unwrap_or_else(|| vec![Finished::default(); n_audio]);

        for (audio, set) in finished.iter_mut().enumerate() {
            if set.len() < k {
                // Not enough finished: take the live beams, best first,
                // ended with the stop token.
                let mut order: Vec<usize> = (0..k).collect();
                order.sort_by(|&a, &b| {
                    sum_logprobs[audio * k + b].total_cmp(&sum_logprobs[audio * k + a])
                });
                for beam in order {
                    let row = audio * k + beam;
                    let mut sequence = tokens[row].clone();
                    sequence.push(self.eot);
                    set.upsert(sequence, sum_logprobs[row]);
                    if set.len() >= k {
                        break;
                    }
                }
            }
        }

        finished
            .into_iter()
            .map(|set| {
                set.entries
                    .into_iter()
                    .map(|(sequence, score)| {
                        let end = sequence[prompt_len..]
                            .iter()
                            .position(|&t| t == self.eot)
                            .map_or(sequence.len(), |i| prompt_len + i);
                        (sequence[prompt_len..end].to_vec(), score)
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::TensorData;

    use super::*;
    use crate::support::testing::CpuBackend;

    type B = CpuBackend;

    fn logits(rows: &[&[f32]]) -> Tensor<B, 2> {
        let vocab = rows[0].len();
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::from_data(
            TensorData::new(flat, [rows.len(), vocab]),
            &Default::default(),
        )
    }

    const EOT: i64 = 0;

    /// At the first step every beam is the same prompt, so the candidates
    /// must be deduplicated by sequence: three beams become the three best
    /// distinct tokens, not three copies of the best one, and the cache is
    /// told which rows they came from.
    #[test]
    fn test_first_step_deduplicates() {
        let mut decoder = BeamSearchDecoder::new(3, EOT, None);
        let mut tokens = vec![vec![7]; 3];
        let mut sums = vec![0.0; 3];
        let mut sources = Vec::new();
        let mut reorder = |s: &[usize]| sources = s.to_vec();

        // Vocabulary of 5; token 4 best, then 3, then 2.
        let row: &[f32] = &[-9.0, -9.0, 1.0, 2.0, 3.0];
        let (feed, done) = TokenDecoder::<B>::update(
            &mut decoder,
            &mut tokens,
            logits(&[row, row, row]),
            &mut sums,
            &mut reorder,
        );
        assert_eq!(tokens, vec![vec![7, 4], vec![7, 3], vec![7, 2]]);
        assert_eq!(feed, vec![4, 3, 2]);
        assert!(!done);
        // A repeated candidate takes the last source that proposed it.
        assert_eq!(sources, vec![2, 2, 2]);
        assert!(sums[0] > sums[1] && sums[1] > sums[2]);
    }

    /// A candidate that ends goes to the finished set rather than the beam;
    /// the beam refills from the runner-up; the search completes when the
    /// set holds `round(k * patience)` sequences.
    #[test]
    fn test_finished_set_and_patience() {
        let mut decoder = BeamSearchDecoder::new(2, EOT, None);
        assert_eq!(decoder.beam_size(), 2);
        assert_eq!(decoder.max_candidates(), 2);
        let mut tokens = vec![vec![7]; 2];
        let mut sums = vec![0.0; 2];
        let mut reorder = |_: &[usize]| {};

        // Best is the stop token, then 3, then 2.
        let row: &[f32] = &[3.0, -9.0, 1.0, 2.0];
        let (feed, done) = TokenDecoder::<B>::update(
            &mut decoder,
            &mut tokens,
            logits(&[row, row]),
            &mut sums,
            &mut reorder,
        );
        assert_eq!(tokens, vec![vec![7, 3], vec![7, 2]]);
        assert_eq!(feed, vec![3, 2]);
        assert!(!done, "one finished, two wanted");

        // Both live beams end now: the set fills and the search completes.
        let (_, done) = TokenDecoder::<B>::update(
            &mut decoder,
            &mut tokens,
            logits(&[row, row]),
            &mut sums,
            &mut reorder,
        );
        assert!(done);

        // Finalize strips the prompt and the stop token; the best finished
        // candidate is the one that ended first.
        let out = TokenDecoder::<B>::finalize(&mut decoder, tokens, sums, 1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 2);
        assert_eq!(out[0][0].0, Vec::<i64>::new());
        assert_eq!(out[0][1].0, vec![3]);
        assert!(out[0][0].1 > out[0][1].1);
    }

    /// With more patience the set is larger, and finalize fills a short
    /// set from the live beams, best first.
    #[test]
    fn test_patience_and_finalize_fill() {
        let mut decoder = BeamSearchDecoder::new(2, EOT, Some(2.0));
        assert_eq!(decoder.max_candidates(), 4);
        let mut tokens = vec![vec![7]; 2];
        let mut sums = vec![0.0; 2];
        let mut reorder = |_: &[usize]| {};

        // Nothing ends.
        let row: &[f32] = &[-9.0, -9.0, 1.0, 2.0];
        let (_, done) = TokenDecoder::<B>::update(
            &mut decoder,
            &mut tokens,
            logits(&[row, row]),
            &mut sums,
            &mut reorder,
        );
        assert!(!done);
        assert_eq!(tokens, vec![vec![7, 3], vec![7, 2]]);

        let out = TokenDecoder::<B>::finalize(&mut decoder, tokens, sums, 1);
        assert_eq!(
            out[0].len(),
            2,
            "filled up to the beam size from live beams"
        );
        assert_eq!(out[0][0].0, vec![3], "the better live beam first");
        assert_eq!(out[0][1].0, vec![2]);
    }

    /// Two audios are independent groups: sources index the whole batch,
    /// and each group refills from its own candidates.
    #[test]
    fn test_groups_are_independent() {
        let mut decoder = BeamSearchDecoder::new(2, EOT, None);
        let mut tokens = vec![vec![7]; 4];
        let mut sums = vec![0.0; 4];
        let mut sources = Vec::new();
        let mut reorder = |s: &[usize]| sources = s.to_vec();

        let a: &[f32] = &[-9.0, -9.0, 1.0, 2.0];
        let b: &[f32] = &[-9.0, 2.0, 1.0, -9.0];
        let (feed, _) = TokenDecoder::<B>::update(
            &mut decoder,
            &mut tokens,
            logits(&[a, a, b, b]),
            &mut sums,
            &mut reorder,
        );
        assert_eq!(feed, vec![3, 2, 1, 2]);
        assert_eq!(sources, vec![1, 1, 3, 3]);
    }

    #[test]
    #[should_panic(expected = "invalid beam size")]
    fn test_rejects_zero_candidates() {
        let _ = BeamSearchDecoder::new(1, EOT, Some(0.1));
    }
}
