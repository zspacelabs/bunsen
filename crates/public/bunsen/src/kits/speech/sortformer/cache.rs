//! The Arrival-Order Speaker Cache: a stream's state between Sortformer steps.
use burn::{
    Tensor,
    prelude::{
        Int,
        TensorData,
    },
};

use super::SortformerStreamingConfig;
use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// One stream's Arrival-Order Speaker Cache and FIFO, between steps.
///
/// Both hold pre-encode embeddings, `[frames, d_model]` on the device; the
/// cache also keeps each of its frames' speaker probabilities, on the host,
/// for scoring when it is compressed. An empty part is held as nothing, never
/// as a zero-length tensor.
///
/// Compression scores and selects on the host: the scores are a few thousand
/// values, and selecting there breaks ties the same way on every backend
/// (the higher score first, then the earlier frame).
#[derive(Clone, Debug)]
pub struct SpeakerCache {
    config: SortformerStreamingConfig,
    n_speakers: usize,

    /// `[cache_frames, d_model]`.
    embeds: Option<Tensor<2>>,

    /// `cache_frames * n_speakers`, frame-major, once the cache is compressed.
    probs: Vec<f32>,

    compressed: bool,

    /// `[fifo_frames, d_model]`.
    fifo: Option<Tensor<2>>,

    compressions: usize,
}

impl SpeakerCache {
    /// An empty cache.
    ///
    /// # Errors
    /// As [`SortformerStreamingConfig::check`].
    pub fn new(
        config: SortformerStreamingConfig,
        n_speakers: usize,
    ) -> BunsenResult<Self> {
        config.check(n_speakers)?;
        Ok(Self {
            config,
            n_speakers,
            embeds: None,
            probs: Vec::new(),
            compressed: false,
            fifo: None,
            compressions: 0,
        })
    }

    /// The config.
    pub fn config(&self) -> &SortformerStreamingConfig {
        &self.config
    }

    /// The frames in the cache.
    pub fn cache_frames(&self) -> usize {
        self.embeds.as_ref().map_or(0, |e| e.dims()[0])
    }

    /// The frames in the FIFO.
    pub fn fifo_frames(&self) -> usize {
        self.fifo.as_ref().map_or(0, |e| e.dims()[0])
    }

    /// Whether the cache has been compressed: its frames are grouped by
    /// speaker, no longer in stream order.
    pub fn is_compressed(&self) -> bool {
        self.compressed
    }

    /// How many times the cache has been compressed.
    pub fn compressions(&self) -> usize {
        self.compressions
    }

    /// The frames a step attends to before its chunk: the cache's, then the
    /// FIFO's, `[cache_frames + fifo_frames, d_model]`; `None` when both are
    /// empty.
    pub fn cached(&self) -> Option<Tensor<2>> {
        cat_present(self.embeds.clone(), self.fifo.clone())
    }

    /// Records a step: the chunk joins the FIFO, an overflowing FIFO moves its
    /// oldest frames to the cache, and an overflowing cache is compressed.
    ///
    /// # Arguments
    /// * `step_probs` - the step's speaker probabilities at the encoder rate,
    ///   `(cached + chunk + look-ahead) * n_speakers`, frame-major, over the
    ///   frames it ran on: [`cached`](Self::cached)'s, then the chunk's.
    /// * `chunk` - `[chunk_frames, d_model]`, the chunk's embeddings, its
    ///   look-ahead excluded.
    /// * `silence` - `[d_model]`, what a compressed cache's silence slots hold.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal) if `step_probs`
    /// does not cover the cached frames and the chunk.
    pub fn update(
        &mut self,
        step_probs: &[f32],
        chunk: Tensor<2>,
        silence: &Tensor<1>,
    ) -> BunsenResult<()> {
        let s = self.n_speakers;
        let (n_cache, n_fifo) = (self.cache_frames(), self.fifo_frames());
        let n_chunk = chunk.dims()[0];
        let covered = step_probs.len() / s;
        if covered < n_cache + n_fifo + n_chunk || !step_probs.len().is_multiple_of(s) {
            return Err(BunsenError::illegal(format!(
                "SpeakerCache::update: {} probabilities do not cover {n_cache} cache, {n_fifo} FIFO \
                 and {n_chunk} chunk frames of {s} speakers",
                step_probs.len()
            )));
        }

        let fifo = cat_present(self.fifo.take(), Some(chunk)).expect("the chunk is present");
        let n_fifo = fifo.dims()[0];
        let popped = self.popped_frames(n_fifo);
        if popped == 0 {
            self.fifo = Some(fifo);
            return Ok(());
        }

        // The FIFO's probabilities are this step's; an uncompressed cache's
        // frames are plain stream frames, which this step re-estimates, and a
        // compressed one's are out of order, so its stored probabilities are
        // the only ones.
        let fifo_probs = &step_probs[n_cache * s..(n_cache + n_fifo) * s];
        let mut probs = if self.compressed {
            std::mem::take(&mut self.probs)
        } else {
            step_probs[..n_cache * s].to_vec()
        };
        probs.extend_from_slice(&fifo_probs[..popped * s]);
        let embeds = cat_present(
            self.embeds.take(),
            Some(fifo.clone().slice_dim(0, 0..popped)),
        )
        .expect("the popped frames are present");
        self.fifo = (popped < n_fifo).then(|| fifo.slice_dim(0, popped..n_fifo));

        let n = embeds.dims()[0];
        if n > self.config.spkcache_len {
            let (embeds, probs) = self.compress(embeds, &probs, silence);
            self.embeds = Some(embeds);
            self.probs = probs;
            self.compressed = true;
            self.compressions += 1;
        } else {
            self.embeds = Some(embeds);
            self.probs = probs;
        }

        #[cfg(any(debug_assertions, test))]
        if let Some(embeds) = &self.embeds {
            crate::contracts::assert_shape_contract_periodically!(
                ["frames", "d_model"],
                &embeds.dims(),
                &[
                    ("frames", self.probs.len() / s),
                    ("d_model", silence.dims()[0])
                ],
            );
        }
        Ok(())
    }

    /// The frames an overflowing FIFO of `n_fifo` frames moves to the cache:
    /// none until it overflows, then at least `update_period`, and enough to
    /// bring it back to capacity.
    fn popped_frames(
        &self,
        n_fifo: usize,
    ) -> usize {
        if n_fifo <= self.config.fifo_len {
            return 0;
        }
        self.config
            .update_period
            .max(n_fifo - self.config.fifo_len)
            .min(n_fifo)
    }

    /// Keeps the cache's `spkcache_len` most telling frames, grouped by
    /// speaker in arrival order and chronological within a speaker, each
    /// speaker's group ending in its silence slots.
    fn compress(
        &self,
        embeds: Tensor<2>,
        probs: &[f32],
        silence: &Tensor<1>,
    ) -> (Tensor<2>, Vec<f32>) {
        let s = self.n_speakers;
        let n = probs.len() / s;
        let selected = self.select(probs, n);

        let device = embeds.device();
        let indices = Tensor::<1, Int>::from_data(
            TensorData::new(
                selected.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                [selected.len()],
            ),
            &device,
        );
        let table = Tensor::cat(vec![embeds, silence.clone().unsqueeze_dim(0)], 0);
        let embeds = table.select(0, indices);

        let mut kept = Vec::with_capacity(selected.len() * s);
        for &frame in &selected {
            if frame < n {
                kept.extend_from_slice(&probs[frame * s..(frame + 1) * s]);
            } else {
                kept.extend(std::iter::repeat_n(0.0, s));
            }
        }
        (embeds, kept)
    }

    /// The frames a compression keeps, `spkcache_len` of them, each a frame
    /// index in `0..=n`, `n` naming the silence row.
    ///
    /// `NeMo`'s `_compress_spkcache`: each frame and speaker is scored from
    /// the clamped log-odds of the speaker against the others; frames where
    /// the speaker is not speaking are ineligible, and so are its
    /// non-positive speech frames once it has enough positive ones; the
    /// frames entering the cache are boosted, then each speaker's best are
    /// boosted strongly and the next weakly; each speaker gains its silence
    /// slots; the best scores across speakers win, in speaker-major order.
    fn select(
        &self,
        probs: &[f32],
        n: usize,
    ) -> Vec<usize> {
        let c = &self.config;
        let s = self.n_speakers;
        let threshold = c.pred_score_threshold as f32;
        let ln_half = 0.5f64.ln();
        let budget = c.speaker_budget(s) as f64;
        let min_positive = (budget * c.min_pos_scores_rate).floor() as usize;
        let strong = (budget * c.strong_boost_rate).floor() as usize;
        let weak = (budget * c.weak_boost_rate).floor() as usize;

        // scores[k][f]: speaker-major.
        let mut scores = vec![vec![f32::NEG_INFINITY; n]; s];
        for f in 0..n {
            let row = &probs[f * s..(f + 1) * s];
            let complements: f32 = row.iter().map(|p| (1.0 - p).max(threshold).ln()).sum();
            for (k, &p) in row.iter().enumerate() {
                if p > 0.5 {
                    scores[k][f] = p.max(threshold).ln() - (1.0 - p).max(threshold).ln()
                        + complements
                        - ln_half as f32;
                }
            }
        }
        for column in scores.iter_mut() {
            let positive = column.iter().filter(|&&v| v > 0.0).count();
            if positive >= min_positive {
                for v in column.iter_mut() {
                    if v.is_finite() && *v <= 0.0 {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
            for v in column.iter_mut().skip(c.spkcache_len) {
                *v += c.latest_boost as f32;
            }
            boost(column, strong, (-2.0 * ln_half) as f32);
            boost(column, weak, (-ln_half) as f32);
        }

        // Each speaker's frames and silence slots, flattened speaker-major.
        let slots = n + c.silence_frames_per_speaker;
        let mut flat: Vec<(f32, usize)> = Vec::with_capacity(s * slots);
        for (k, column) in scores.iter().enumerate() {
            let silence = std::iter::repeat_n(f32::INFINITY, c.silence_frames_per_speaker);
            for (f, v) in column.iter().copied().chain(silence).enumerate() {
                flat.push((v, k * slots + f));
            }
        }
        flat.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let sentinel = s * slots;
        let mut chosen: Vec<usize> = flat
            .iter()
            .take(c.spkcache_len)
            .map(|&(v, i)| if v == f32::NEG_INFINITY { sentinel } else { i })
            .collect();
        chosen.sort_unstable();
        chosen
            .into_iter()
            .map(|i| if i == sentinel { n } else { (i % slots).min(n) })
            .collect()
    }
}

/// Adds `amount` to the `count` best values of `column`, the earlier first
/// among equals.
fn boost(
    column: &mut [f32],
    count: usize,
    amount: f32,
) {
    let mut order: Vec<usize> = (0..column.len()).collect();
    order.sort_by(|&a, &b| column[b].total_cmp(&column[a]).then(a.cmp(&b)));
    for &i in order.iter().take(count) {
        column[i] += amount;
    }
}

/// `a` then `b` along the frames, either possibly absent.
fn cat_present(
    a: Option<Tensor<2>>,
    b: Option<Tensor<2>>,
) -> Option<Tensor<2>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(Tensor::cat(vec![a, b], 0)),
        (a, b) => a.or(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::cpu_device;

    fn config() -> SortformerStreamingConfig {
        SortformerStreamingConfig::new(3, 1, 6, 12, 4)
    }

    fn frames(
        n: usize,
        from: f32,
    ) -> Tensor<2> {
        let data: Vec<f32> = (0..n * 2).map(|i| from + i as f32).collect();
        Tensor::from_data(TensorData::new(data, [n, 2]), &cpu_device())
    }

    /// The FIFO fills, then overflows into the cache by at least the update
    /// period.
    #[test]
    fn test_fifo_overflows_into_the_cache() {
        let mut cache = SpeakerCache::new(config(), 3).unwrap();
        let silence = Tensor::<1>::zeros([2], &cpu_device());
        assert!(cache.cached().is_none());

        cache
            .update(&[0.1; 3 * 3], frames(3, 0.0), &silence)
            .unwrap();
        assert_eq!((cache.cache_frames(), cache.fifo_frames()), (0, 3));
        cache
            .update(&[0.1; 6 * 3], frames(3, 10.0), &silence)
            .unwrap();
        assert_eq!((cache.cache_frames(), cache.fifo_frames()), (0, 6));
        // 9 frames: 3 over, the update period of 4 moves.
        cache
            .update(&[0.1; 9 * 3], frames(3, 20.0), &silence)
            .unwrap();
        assert_eq!((cache.cache_frames(), cache.fifo_frames()), (4, 5));
        assert!(!cache.is_compressed());
        let cached = cache.cached().unwrap();
        assert_eq!(cached.dims(), [9, 2]);
        // In stream order.
        let first: Vec<f32> = cached
            .slice_dim(0, 0..1)
            .into_data()
            .try_into_vec_as()
            .unwrap();
        assert_eq!(first, [0.0, 1.0]);

        assert!(cache.update(&[0.1; 2], frames(1, 0.0), &silence).is_err());
    }

    /// A compression keeps the cache's capacity, groups frames by speaker
    /// with each group's silence slot after it, and fills the slots and any
    /// unfilled place with silence.
    #[test]
    fn test_compression_groups_by_speaker() {
        let c = config();
        let cache = SpeakerCache::new(c.clone(), 3).unwrap();
        // 16 frames: speaker 0 talks in 0..6, speaker 1 in 6..12, nobody after.
        let mut probs = Vec::new();
        for f in 0..16 {
            probs.extend(match f {
                0..6 => [0.9, 0.1, 0.1],
                6..12 => [0.1, 0.8 + f as f32 / 100.0, 0.1],
                _ => [0.1, 0.1, 0.1],
            });
        }
        let chosen = cache.select(&probs, 16);
        assert_eq!(chosen.len(), c.spkcache_len);
        // Ascending: speaker 0's frames, then speaker 1's, then silence.
        let speaker0: Vec<usize> = chosen.iter().copied().filter(|&f| f < 6).collect();
        let speaker1: Vec<usize> = chosen
            .iter()
            .copied()
            .filter(|&f| (6..12).contains(&f))
            .collect();
        assert!(!speaker0.is_empty() && !speaker1.is_empty());
        assert!(chosen.iter().all(|&f| f < 12 || f == 16), "{chosen:?}");
        assert!(chosen.windows(2).take(speaker0.len()).all(|w| w[0] < w[1]));
        // Three speakers' silence slots, at least.
        assert!(chosen.iter().filter(|&&f| f == 16).count() >= 3);
    }

    /// Ties go to the earlier frame, on every backend.
    #[test]
    fn test_ties_are_deterministic() {
        let cache = SpeakerCache::new(config(), 3).unwrap();
        let probs: Vec<f32> = (0..16).flat_map(|_| [0.9f32, 0.1, 0.1]).collect();
        let a = cache.select(&probs, 16);
        assert_eq!(a, cache.select(&probs, 16));
        // Nine places past the three silence slots, every frame alike but
        // the four entering the cache, which the latest-frames boost and then
        // the strong and weak boosts favour; the earliest fill the rest.
        let kept: Vec<usize> = a.iter().copied().filter(|&f| f < 16).collect();
        assert_eq!(kept, [0, 1, 2, 3, 4, 12, 13, 14, 15]);
    }

    #[test]
    fn test_boost_takes_the_best() {
        let mut v = vec![1.0, 3.0, f32::NEG_INFINITY, 3.0, 2.0];
        boost(&mut v, 2, 10.0);
        assert_eq!(v, [1.0, 13.0, f32::NEG_INFINITY, 13.0, 2.0]);
    }
}
