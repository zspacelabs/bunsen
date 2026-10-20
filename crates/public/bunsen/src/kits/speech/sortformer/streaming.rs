//! The Arrival-Order Speaker Cache's geometry and scoring.
use burn::config::Config;

use crate::errors::{
    BunsenResult,
    ConstraintError,
    Rule,
};

/// How a Sortformer streams: the chunks it steps in, the look-ahead each
/// sees, and the Arrival-Order Speaker Cache (AOSC) and FIFO every step
/// attends to (Medennikov et al. 2025, `NeMo`'s `sortformer_modules`).
///
/// Lengths are in encoder frames. Each step runs the model over the cache's
/// frames, the FIFO's, and one chunk with its look-ahead; the chunk joins the
/// FIFO, and when the FIFO overflows its oldest frames move to the cache.
/// When the cache overflows it is compressed back to `spkcache_len` frames,
/// keeping each speaker's most telling frames (scored from the model's
/// probabilities, with `strong_boost_rate` and `weak_boost_rate` shares of
/// each speaker's budget boosted, and the newest frames favoured by
/// `latest_boost`), grouped by speaker in arrival order, chronological within
/// a speaker, with `silence_frames_per_speaker` silence slots each.
///
/// The latency is `(chunk_len + right_context)` encoder frames. The presets
/// are the model cards' tables.
#[derive(Config, Debug, PartialEq)]
pub struct SortformerStreamingConfig {
    /// The encoder frames each step emits.
    pub chunk_len: usize,

    /// The look-ahead each step sees past its chunk.
    pub right_context: usize,

    /// The FIFO's capacity.
    pub fifo_len: usize,

    /// The cache's capacity.
    pub spkcache_len: usize,

    /// The frames moved from an overflowing FIFO to the cache, at least.
    pub update_period: usize,

    /// The silence slots each speaker keeps in a compressed cache.
    #[config(default = "1")]
    pub silence_frames_per_speaker: usize,

    /// Probabilities are clamped into `[t, 1 - t]` before scoring.
    #[config(default = "0.25")]
    pub pred_score_threshold: f64,

    /// Added to the scores of the frames entering the cache.
    #[config(default = "0.05")]
    pub latest_boost: f64,

    /// The share of a speaker's budget boosted strongly.
    #[config(default = "0.75")]
    pub strong_boost_rate: f64,

    /// The share of a speaker's budget boosted weakly.
    #[config(default = "1.5")]
    pub weak_boost_rate: f64,

    /// The share of a speaker's budget that, positive, makes its non-positive
    /// speech frames ineligible.
    #[config(default = "0.5")]
    pub min_pos_scores_rate: f64,
}

impl SortformerStreamingConfig {
    /// `Nemotron-3-Diarization`'s low-latency preset, the default: chunks of 9
    /// with 4 frames of look-ahead, 1.04 s.
    pub fn nemotron_low_latency() -> Self {
        Self::new(9, 4, 264, 264, 222)
    }

    /// `Nemotron-3-Diarization`'s very-low-latency preset: 6 and 2, 0.64 s.
    pub fn nemotron_very_low_latency() -> Self {
        Self::new(6, 2, 264, 264, 222)
    }

    /// `Nemotron-3-Diarization`'s ultra-low-latency preset: 3 and 1, 0.32 s.
    pub fn nemotron_ultra_low_latency() -> Self {
        Self::new(3, 1, 264, 264, 222)
    }

    /// `Nemotron-3-Diarization`'s offline-style preset: chunks of 340 with 40
    /// frames of look-ahead, 30.4 s.
    pub fn nemotron_offline() -> Self {
        Self::new(340, 40, 40, 264, 300)
    }

    /// The latency, in encoder frames: a chunk and its look-ahead.
    pub fn latency_frames(&self) -> usize {
        self.chunk_len + self.right_context
    }

    /// Each speaker's budget of cache frames, its silence slots excluded.
    pub fn speaker_budget(
        &self,
        n_speakers: usize,
    ) -> usize {
        (self.spkcache_len / n_speakers).saturating_sub(self.silence_frames_per_speaker)
    }

    /// Checks the geometry against a model of `n_speakers`.
    ///
    /// # Errors
    /// [`Illegal`](crate::errors::BunsenErrorKind::Illegal), with a
    /// [`ConstraintError`] cause, for an empty chunk or update period, or a
    /// cache too small to hold each speaker a frame and its silence slots:
    /// `spkcache_len >= (1 + silence_frames_per_speaker) * n_speakers`.
    pub fn check(
        &self,
        n_speakers: usize,
    ) -> BunsenResult<()> {
        for (field, value) in [
            ("chunk_len", self.chunk_len),
            ("update_period", self.update_period),
        ] {
            if value == 0 {
                return Err(
                    ConstraintError::zero_or_empty("SortformerStreamingConfig", field).into(),
                );
            }
        }
        let least = (1 + self.silence_frames_per_speaker) * n_speakers;
        if self.spkcache_len < least {
            return Err(ConstraintError::new(
                "SortformerStreamingConfig",
                "spkcache_len",
                Rule::Relation {
                    lhs: ("spkcache_len".into(), self.spkcache_len.to_string()),
                    op: ">=",
                    rhs: (
                        "(1 + silence_frames_per_speaker) * n_speakers".into(),
                        least.to_string(),
                    ),
                },
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presets() {
        let low = SortformerStreamingConfig::nemotron_low_latency();
        assert_eq!(low.latency_frames(), 13);
        assert_eq!(low.speaker_budget(8), 32);
        for preset in [
            low,
            SortformerStreamingConfig::nemotron_very_low_latency(),
            SortformerStreamingConfig::nemotron_ultra_low_latency(),
            SortformerStreamingConfig::nemotron_offline(),
        ] {
            preset.check(8).unwrap();
        }
        assert_eq!(
            SortformerStreamingConfig::nemotron_offline().latency_frames(),
            380
        );
    }

    #[test]
    fn test_check() {
        let c = SortformerStreamingConfig::new(3, 1, 6, 12, 4);
        c.check(3).unwrap();
        assert!(c.check(7).is_err());
        let empty_chunk = SortformerStreamingConfig {
            chunk_len: 0,
            ..c.clone()
        };
        assert!(empty_chunk.check(3).is_err());
        let no_period = SortformerStreamingConfig {
            update_period: 0,
            ..c
        };
        assert!(no_period.check(3).is_err());
    }
}
