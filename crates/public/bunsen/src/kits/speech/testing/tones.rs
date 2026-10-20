/// Represents the configuration of a tone generator.
#[derive(Debug)]
pub enum ToneConfig {
    /// Tone configured by number of samples.
    Samples(usize),

    /// Tone configured by duration and hz.
    Duration {
        /// The duration of the tone in seconds.
        seconds: f32,

        /// The frequency of the tone in Hz.
        hz: f32,
    },
}

impl ToneConfig {
    /// Returns the number of samples in the tone.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            ToneConfig::Samples(samples) => *samples,
            ToneConfig::Duration { seconds, hz } => (*seconds * *hz) as usize,
        }
    }

    /// Returns the samples of the tone.
    pub fn build(&self) -> Vec<f32> {
        let samples = self.len();
        (0..samples)
            .map(|i| {
                let t = i as f32 / samples as f32;
                let s = 2.0 * std::f32::consts::PI * i as f32 / 16_000.0;
                (std::f32::consts::PI * t).sin()
                    * ((440.0 * s).sin() + 0.3 * (1_730.0 * s).sin())
                    * 0.4
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::support::testing::assert_close_to_vec;

    #[test]
    fn test_duration() {
        let a = ToneConfig::Samples(32_000).build();
        let b = ToneConfig::Duration {
            seconds: 2.0,
            hz: 16_000.0,
        }
        .build();

        assert_eq!(a.len(), 32_000);
        assert_close_to_vec(&a, &b, 0.001);
    }
}
