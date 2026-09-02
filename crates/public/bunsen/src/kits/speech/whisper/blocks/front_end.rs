//! # The audio front end a checkpoint was trained with.
//!
//! Whisper's log-mels are a grid fixed in time &mdash; a 25 ms window every
//! 10 ms &mdash; computed at 16 kHz and floored 8 dB under each window's
//! maximum. A checkpoint records none of that; it is the convention of the
//! pipeline that trained it. [`WhisperFrontEndConfig`] declares it on the
//! model, defaulting to upstream's, so every sample-domain number is
//! derived from it rather than written down, and a checkpoint trained
//! differently can say so. The mel options and the packaging it drives
//! live with the driver, as methods on it.

use burn::config::Config;

use crate::errors::{
    BunsenError,
    BunsenResult,
};

/// The audio front end a checkpoint's log-mels were computed with.
///
/// The grid is in time; [`hop`](Self::hop) and [`n_fft`](Self::n_fft) put
/// it on samples at [`sample_rate`](Self::sample_rate).
#[derive(Config, Debug, PartialEq)]
pub struct WhisperFrontEndConfig {
    /// The sample rate, in Hz.
    #[config(default = "16_000")]
    pub sample_rate: usize,

    /// The hop between mel frames, in milliseconds. One timestamp step over
    /// the encoder's stride
    /// ([`AUDIO_ENCODER_STRIDE`](super::AUDIO_ENCODER_STRIDE)), so that one
    /// encoder position is one timestamp token.
    #[config(default = "10")]
    pub hop_ms: usize,

    /// The analysis window, in milliseconds.
    #[config(default = "25")]
    pub window_ms: usize,

    /// The dynamic range kept under each window's maximum, in dB: log-mels
    /// further below the maximum are floored to it before packaging.
    #[config(default = "8.0")]
    pub range_clamp_db: f64,
}

impl WhisperFrontEndConfig {
    /// The hop, in samples.
    pub fn hop(&self) -> usize {
        self.sample_rate * self.hop_ms / 1000
    }

    /// The analysis window, in samples: the FFT length.
    pub fn n_fft(&self) -> usize {
        self.sample_rate * self.window_ms / 1000
    }

    /// Checks that the grid falls on whole samples.
    ///
    /// # Errors
    /// [`BunsenError::Invalid`] if the rate, hop or window is zero, or the
    /// rate does not put the hop and the window on whole samples. At the
    /// default 10 ms and 25 ms that is any rate not a multiple of 200 Hz.
    pub fn validate(&self) -> BunsenResult<()> {
        if self.sample_rate == 0 || self.hop_ms == 0 || self.window_ms == 0 {
            return Err(BunsenError::Invalid(format!(
                "a front end needs a rate, a hop and a window; got {} Hz, {} ms, {} ms",
                self.sample_rate, self.hop_ms, self.window_ms,
            )));
        }
        for (what, ms) in [("hop", self.hop_ms), ("window", self.window_ms)] {
            if !(self.sample_rate * ms).is_multiple_of(1000) {
                return Err(BunsenError::Invalid(format!(
                    "a {ms} ms {what} is not a whole number of samples at {} Hz",
                    self.sample_rate,
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_are_upstreams() {
        let front_end = WhisperFrontEndConfig::new();
        assert_eq!(front_end.sample_rate, 16_000);
        assert_eq!((front_end.hop(), front_end.n_fft()), (160, 400));
        assert_eq!(front_end.range_clamp_db, 8.0);
        assert!(front_end.validate().is_ok());
    }

    #[test]
    fn test_grid_scales_with_the_rate() {
        let at = |rate: usize| WhisperFrontEndConfig::new().with_sample_rate(rate);

        assert_eq!((at(8_000).hop(), at(8_000).n_fft()), (80, 200));
        assert!(at(8_000).validate().is_ok());
        assert_eq!((at(48_000).hop(), at(48_000).n_fft()), (480, 1_200));

        assert!(at(44_100).validate().is_err(), "not a whole hop");
        assert!(at(0).validate().is_err());
        assert!(at(16_000).with_hop_ms(0).validate().is_err());
        assert!(
            at(16_000).with_window_ms(3).validate().is_ok(),
            "3 ms is 48 whole samples"
        );
        assert!(
            at(16_000)
                .with_hop_ms(3)
                .with_window_ms(7)
                .validate()
                .is_ok(),
            "any whole-sample grid passes; the 10 ms / 25 ms pairing is convention, not checked"
        );
        assert!(
            at(22_050).with_hop_ms(10).validate().is_err(),
            "220.5 samples"
        );
    }
}
