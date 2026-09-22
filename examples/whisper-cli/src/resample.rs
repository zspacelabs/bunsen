//! # A windowed-sinc resampler
//!
//! The capture path's rate to the model's. bunsen decodes files at the
//! model's rate and declines to resample them, because a resample is a
//! signal-processing decision the caller makes deliberately; a microphone
//! runs at whatever rate its host offers, so `live` makes that decision
//! here, once, and only when the device would not open at the model's
//! rate.
//!
//! Each output sample is the input convolved with a Blackman-windowed sinc
//! centred on its position, cut off at the lower of the two Nyquist rates,
//! and normalized to unit DC gain for its phase. The kernel is symmetric
//! and output sample `n` sits at input position `n * in / out` exactly, so
//! there is no delay to account for, and the output is the same whatever
//! the pushes were cut into.

use std::f64::consts::PI;

/// Kernel half-width in output-rate samples; widened by the ratio when
/// downsampling, so the stop band stays where the aliases are.
const HALF_TAPS: usize = 16;

/// A streaming rate converter: feed input in any pieces, collect output.
#[derive(Debug, Clone)]
pub struct Resampler {
    in_rate: usize,
    out_rate: usize,

    /// Cut-off as a fraction of the input's Nyquist: 1 upsampling,
    /// `out / in` downsampling.
    cutoff: f64,

    /// Kernel half-width, in input samples.
    half: usize,

    /// Input the kernel may still reach: opens with `half` zeros standing
    /// in for the samples before the stream.
    buffer: Vec<f32>,

    /// Samples released from the front of `buffer` so far.
    released: usize,

    /// Output samples emitted so far.
    emitted: usize,
}

impl Resampler {
    /// A converter from `in_rate` to `out_rate`, both in samples per second.
    ///
    /// # Panics
    /// If either rate is zero.
    pub fn new(
        in_rate: usize,
        out_rate: usize,
    ) -> Self {
        assert!(in_rate > 0 && out_rate > 0, "rates must be non-zero");
        let cutoff = (out_rate as f64 / in_rate as f64).min(1.0);
        let half = (HALF_TAPS as f64 / cutoff).ceil() as usize;
        Self {
            in_rate,
            out_rate,
            cutoff,
            half,
            buffer: vec![0.0; half],
            released: 0,
            emitted: 0,
        }
    }

    /// Whether the rates are equal, so samples pass straight through.
    pub fn is_identity(&self) -> bool {
        self.in_rate == self.out_rate
    }

    /// Feeds `input`, appending every output sample it completes to `out`.
    pub fn process(
        &mut self,
        input: &[f32],
        out: &mut Vec<f32>,
    ) {
        if self.is_identity() {
            out.extend_from_slice(input);
            return;
        }

        self.buffer.extend_from_slice(input);

        // Emit every output whose rightmost tap has arrived.
        loop {
            let pos = self.relative_position(self.emitted);
            let hi = (pos + self.half as f64).floor();
            if hi >= self.buffer.len() as f64 {
                break;
            }
            let sample = self.tap(pos);
            out.push(sample);
            self.emitted += 1;
        }

        // Release what no later output reaches: everything left of the next
        // output's leftmost tap.
        let next = self.relative_position(self.emitted);
        let lo = ((next - self.half as f64).ceil().max(0.0) as usize).min(self.buffer.len());
        self.buffer.drain(..lo);
        self.released += lo;
    }

    /// The input position of output `n`, in samples from `buffer[0]`.
    ///
    /// Computed from the counts rather than accumulated, so the same
    /// output has the same position however the input was pushed.
    fn relative_position(
        &self,
        n: usize,
    ) -> f64 {
        let absolute = (n as u64 * self.in_rate as u64) as f64 / self.out_rate as f64;
        // `buffer[0]` stands `half` samples before the released count: the
        // leading zeros sit at negative positions.
        absolute - self.released as f64 + self.half as f64
    }

    /// The kernel at `x` input samples from the output's position.
    fn kernel(
        &self,
        x: f64,
    ) -> f64 {
        let t = x / self.half as f64;
        if t.abs() >= 1.0 {
            return 0.0;
        }
        let window = 0.42 + 0.5 * (PI * t).cos() + 0.08 * (2.0 * PI * t).cos();
        let arg = self.cutoff * x;
        let sinc = if arg.abs() < 1e-9 {
            1.0
        } else {
            (PI * arg).sin() / (PI * arg)
        };
        self.cutoff * sinc * window
    }

    /// One output sample at position `pos` in `buffer`.
    fn tap(
        &self,
        pos: f64,
    ) -> f32 {
        let lo = (pos - self.half as f64).ceil().max(0.0) as usize;
        let hi = ((pos + self.half as f64).floor() as usize).min(self.buffer.len() - 1);
        let mut acc = 0.0f64;
        let mut norm = 0.0f64;
        for (k, &sample) in self.buffer.iter().enumerate().take(hi + 1).skip(lo) {
            let w = self.kernel(k as f64 - pos);
            acc += sample as f64 * w;
            norm += w;
        }
        (acc / norm) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(
        rate: usize,
        hz: f64,
        secs: f64,
    ) -> Vec<f32> {
        (0..(rate as f64 * secs) as usize)
            .map(|n| (2.0 * PI * hz * n as f64 / rate as f64).sin() as f32)
            .collect()
    }

    fn rms(x: &[f32]) -> f64 {
        (x.iter().map(|v| (v * v) as f64).sum::<f64>() / x.len() as f64).sqrt()
    }

    #[test]
    fn test_identity_passes_through() {
        let input = tone(16000, 440.0, 0.1);
        let mut resampler = Resampler::new(16000, 16000);
        assert!(resampler.is_identity());

        let mut out = Vec::new();
        resampler.process(&input, &mut out);
        assert_eq!(out, input);
    }

    /// A tone below the output's Nyquist comes through at its frequency and
    /// amplitude, aligned sample for sample with the tone the output rate
    /// would have sampled.
    fn assert_tone_preserved(
        in_rate: usize,
        out_rate: usize,
        hz: f64,
    ) {
        let input = tone(in_rate, hz, 1.0);
        let mut resampler = Resampler::new(in_rate, out_rate);
        assert!(!resampler.is_identity());

        let mut out = Vec::new();
        resampler.process(&input, &mut out);

        // The kernel's reach holds back the last `half` input samples.
        let expected = input.len() * out_rate / in_rate;
        let held = resampler.half * out_rate / in_rate + 1;
        assert!(
            out.len() <= expected && out.len() + held >= expected,
            "{in_rate} -> {out_rate}: {} samples out, expected about {expected}",
            out.len(),
        );

        // Away from the edges, where the kernel has full context.
        let margin = out.len() / 10;
        let mut worst = 0.0f64;
        for (n, &sample) in out.iter().enumerate().take(out.len() - margin).skip(margin) {
            let reference = (2.0 * PI * hz * n as f64 / out_rate as f64).sin();
            worst = worst.max((sample as f64 - reference).abs());
        }
        assert!(
            worst < 0.02,
            "{in_rate} -> {out_rate}: worst error {worst:.4} against the {hz} Hz tone",
        );
    }

    #[test]
    fn test_48k_to_16k() {
        assert_tone_preserved(48000, 16000, 1000.0);
    }

    #[test]
    fn test_44100_to_16k() {
        assert_tone_preserved(44100, 16000, 1000.0);
    }

    #[test]
    fn test_8k_to_16k_upsamples() {
        assert_tone_preserved(8000, 16000, 1000.0);
    }

    /// A tone above the output's Nyquist is removed, not folded down.
    #[test]
    fn test_alias_removed() {
        let input = tone(48000, 12000.0, 1.0);
        let mut resampler = Resampler::new(48000, 16000);
        let mut out = Vec::new();
        resampler.process(&input, &mut out);

        let margin = out.len() / 10;
        let level = rms(&out[margin..out.len() - margin]);
        assert!(level < 0.01, "a 12 kHz tone survived at rms {level:.4}");
    }

    /// The output does not depend on how the input was cut into pushes.
    #[test]
    fn test_push_invariance() {
        let input = tone(44100, 700.0, 0.5);

        let mut whole = Vec::new();
        Resampler::new(44100, 16000).process(&input, &mut whole);

        let mut pieces = Vec::new();
        let mut resampler = Resampler::new(44100, 16000);
        let sizes = [1usize, 7, 100, 333, 1024];
        let mut at = 0;
        let mut i = 0;
        while at < input.len() {
            let end = (at + sizes[i % sizes.len()]).min(input.len());
            resampler.process(&input[at..end], &mut pieces);
            at = end;
            i += 1;
        }

        assert_eq!(pieces, whole);
    }
}
