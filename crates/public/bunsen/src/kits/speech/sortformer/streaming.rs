//! The Arrival-Order Speaker Cache's geometry and scoring.
use burn::config::Config;

use crate::errors::{
    BunsenError,
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
/// A step also re-encodes `left_context` frames before its chunk, as context
/// only: they neither come out nor join the FIFO again.
///
/// The latency is `(chunk_len + right_context)` encoder frames. The presets
/// are the model cards' tables.
#[derive(Config, Debug, PartialEq)]
pub struct SortformerStreamingConfig {
    /// The encoder frames each step emits.
    pub chunk_len: usize,

    /// The look-ahead each step sees past its chunk.
    pub right_context: usize,

    /// The frames before its chunk each step re-encodes, as context.
    #[config(default = "0")]
    pub left_context: usize,

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

    /// A frame whose speaker probabilities sum below this is silent: what a
    /// running-mean silence fill takes in
    /// ([`SilenceFill::RunningMean`](super::SilenceFill::RunningMean)).
    #[config(default = "0.2")]
    pub silence_threshold: f64,
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

    /// A streaming Sortformer v2 or v2.1 preset: one frame of left context,
    /// three silence slots per speaker, a cache of 188.
    fn v2(
        chunk_len: usize,
        right_context: usize,
        fifo_len: usize,
        update_period: usize,
    ) -> Self {
        Self::new(chunk_len, right_context, fifo_len, 188, update_period)
            .with_left_context(1)
            .with_silence_frames_per_speaker(3)
    }

    /// Sortformer v2's very-high-latency preset: chunks of 340 with 40 frames
    /// of look-ahead, 30.4 s; the most accurate, for files.
    pub fn sortformer_v2_very_high_latency() -> Self {
        Self::v2(340, 40, 40, 300)
    }

    /// Sortformer v2's high-latency preset: chunks of 124 with 1 frame of
    /// look-ahead, 10 s.
    pub fn sortformer_v2_high_latency() -> Self {
        Self::v2(124, 1, 124, 124)
    }

    /// Sortformer v2's low-latency preset: chunks of 6 with 7 frames of
    /// look-ahead, 1.04 s.
    pub fn sortformer_v2_low_latency() -> Self {
        Self::v2(6, 7, 188, 144)
    }

    /// Sortformer v2's ultra-low-latency preset: chunks of 3 with 1 frame of
    /// look-ahead, 0.32 s.
    pub fn sortformer_v2_ultra_low_latency() -> Self {
        Self::v2(3, 1, 188, 144)
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

/// How far behind the audio a Sortformer runs: the latencies its model
/// card's streaming presets come in. A family has a preset for some of them
/// ([`SortformerFamily::preset`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SortformerLatency {
    /// 0.32 s.
    UltraLow,

    /// 0.64 s.
    VeryLow,

    /// 1.04 s: every family has it.
    #[default]
    Low,

    /// 10 s.
    High,

    /// 30.4 s: the most accurate, for files.
    Offline,
}

impl SortformerLatency {
    /// Every latency, lowest first.
    pub const ALL: [Self; 5] = [
        Self::UltraLow,
        Self::VeryLow,
        Self::Low,
        Self::High,
        Self::Offline,
    ];
}

impl std::fmt::Display for SortformerLatency {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.write_str(match self {
            Self::UltraLow => "ultra-low",
            Self::VeryLow => "very-low",
            Self::Low => "low",
            Self::High => "high",
            Self::Offline => "offline",
        })
    }
}

/// The Sortformer families, each with its model card's streaming presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SortformerFamily {
    /// `Nemotron-3-Diarization`.
    Nemotron,

    /// The streaming Sortformer v2 and v2.1.
    SortformerV2,
}

impl SortformerFamily {
    /// The family's preset at `latency`, if its model card has one:
    ///
    /// | latency | `Nemotron` | `SortformerV2` |
    /// |---|---|---|
    /// | ultra-low | [`nemotron_ultra_low_latency`] | [`sortformer_v2_ultra_low_latency`] |
    /// | very-low | [`nemotron_very_low_latency`] | |
    /// | low | [`nemotron_low_latency`] | [`sortformer_v2_low_latency`] |
    /// | high | | [`sortformer_v2_high_latency`] |
    /// | offline | [`nemotron_offline`] | [`sortformer_v2_very_high_latency`] |
    ///
    /// [`nemotron_ultra_low_latency`]: SortformerStreamingConfig::nemotron_ultra_low_latency
    /// [`nemotron_very_low_latency`]: SortformerStreamingConfig::nemotron_very_low_latency
    /// [`nemotron_low_latency`]: SortformerStreamingConfig::nemotron_low_latency
    /// [`nemotron_offline`]: SortformerStreamingConfig::nemotron_offline
    /// [`sortformer_v2_ultra_low_latency`]: SortformerStreamingConfig::sortformer_v2_ultra_low_latency
    /// [`sortformer_v2_low_latency`]: SortformerStreamingConfig::sortformer_v2_low_latency
    /// [`sortformer_v2_high_latency`]: SortformerStreamingConfig::sortformer_v2_high_latency
    /// [`sortformer_v2_very_high_latency`]: SortformerStreamingConfig::sortformer_v2_very_high_latency
    ///
    /// # Errors
    /// [`Unsupported`](crate::errors::BunsenErrorKind::Unsupported), naming
    /// the latencies the family has, when it has none at `latency`.
    pub fn preset(
        self,
        latency: SortformerLatency,
    ) -> BunsenResult<SortformerStreamingConfig> {
        self.lookup(latency).ok_or_else(|| {
            let has = self
                .latencies()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            BunsenError::unsupported(format!(
                "{self} has no {latency}-latency preset; it has {has}"
            ))
        })
    }

    /// The latencies the family has a preset for, lowest first.
    pub fn latencies(self) -> Vec<SortformerLatency> {
        SortformerLatency::ALL
            .into_iter()
            .filter(|&l| self.lookup(l).is_some())
            .collect()
    }

    fn lookup(
        self,
        latency: SortformerLatency,
    ) -> Option<SortformerStreamingConfig> {
        use SortformerLatency as L;
        type C = SortformerStreamingConfig;
        match (self, latency) {
            (Self::Nemotron, L::UltraLow) => Some(C::nemotron_ultra_low_latency()),
            (Self::Nemotron, L::VeryLow) => Some(C::nemotron_very_low_latency()),
            (Self::Nemotron, L::Low) => Some(C::nemotron_low_latency()),
            (Self::Nemotron, L::Offline) => Some(C::nemotron_offline()),
            (Self::SortformerV2, L::UltraLow) => Some(C::sortformer_v2_ultra_low_latency()),
            (Self::SortformerV2, L::Low) => Some(C::sortformer_v2_low_latency()),
            (Self::SortformerV2, L::High) => Some(C::sortformer_v2_high_latency()),
            (Self::SortformerV2, L::Offline) => Some(C::sortformer_v2_very_high_latency()),
            (Self::Nemotron, L::High) | (Self::SortformerV2, L::VeryLow) => None,
        }
    }
}

impl std::fmt::Display for SortformerFamily {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.write_str(match self {
            Self::Nemotron => "Nemotron-3-Diarization",
            Self::SortformerV2 => "Sortformer v2",
        })
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
        assert_eq!(
            SortformerStreamingConfig::nemotron_low_latency().left_context,
            0
        );
    }

    /// The v2 presets: the model card's latencies, one frame of left
    /// context, and budgets of 44 frames per speaker of 4.
    #[test]
    fn test_v2_presets() {
        let presets = [
            (
                SortformerStreamingConfig::sortformer_v2_very_high_latency(),
                380,
            ),
            (SortformerStreamingConfig::sortformer_v2_high_latency(), 125),
            (SortformerStreamingConfig::sortformer_v2_low_latency(), 13),
            (
                SortformerStreamingConfig::sortformer_v2_ultra_low_latency(),
                4,
            ),
        ];
        for (preset, latency) in presets {
            preset.check(4).unwrap();
            assert_eq!(preset.latency_frames(), latency);
            assert_eq!(preset.left_context, 1);
            assert_eq!(preset.speaker_budget(4), 44);
        }
    }

    /// Each family's presets are its model card's, at the latency each
    /// names; one it lacks is refused, naming those it has.
    #[test]
    fn test_family_presets() {
        use SortformerLatency as L;
        type C = SortformerStreamingConfig;
        let nemotron = SortformerFamily::Nemotron;
        assert_eq!(
            nemotron.latencies(),
            [L::UltraLow, L::VeryLow, L::Low, L::Offline]
        );
        assert_eq!(nemotron.preset(L::Low).unwrap(), C::nemotron_low_latency());
        assert_eq!(nemotron.preset(L::Offline).unwrap(), C::nemotron_offline());
        let err = nemotron.preset(L::High).unwrap_err();
        assert!(
            err.to_string()
                .contains("no high-latency preset; it has ultra-low, very-low, low, offline"),
            "{err}"
        );

        let v2 = SortformerFamily::SortformerV2;
        assert_eq!(v2.latencies(), [L::UltraLow, L::Low, L::High, L::Offline]);
        assert_eq!(v2.preset(L::High).unwrap(), C::sortformer_v2_high_latency());
        assert_eq!(
            v2.preset(L::Offline).unwrap(),
            C::sortformer_v2_very_high_latency()
        );
        assert!(v2.preset(L::VeryLow).is_err());

        // A latency is the same wall time in every family that has it.
        for latency in L::ALL {
            let frames: Vec<usize> = [nemotron, v2]
                .into_iter()
                .filter_map(|f| f.preset(latency).ok())
                .map(|c| c.latency_frames())
                .collect();
            assert!(frames.windows(2).all(|w| w[0] == w[1]), "{latency}");
        }
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
