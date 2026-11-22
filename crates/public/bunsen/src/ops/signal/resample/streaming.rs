//! [`StreamingResampler`]: a Blackman-windowed sinc, streaming.

use alloc::{
    vec,
    vec::Vec,
};

// `std` has these float methods inherently; `core` has them through `Float`.
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::ops::signal::kernels::{
    sinc::normalized_sinc,
    window::{
        DualCosineWindow,
        WindowTaper,
    },
};

/// The classic Blackman window, `0.42 + 0.5·cos(πx) + 0.08·cos(2πx)`, its
/// coefficients as written ([`DualCosineWindow::blackman`] computes `gamma`
/// as `1 - 0.92`, a rounding off `0.08`).
const BLACKMAN: DualCosineWindow = DualCosineWindow {
    alpha: 0.42,
    beta: 0.5,
    gamma: 0.08,
    periodic: false,
};

/// Kernel half-width in output-rate samples; widened by the ratio when
/// downsampling, so the stop band stays where the aliases are.
const HALF_TAPS: usize = 16;

/// A streaming rate converter: feed input in any pieces, collect output;
/// [`resample`](Self::resample) converts a whole signal.
///
/// bunsen decodes files at a model's rate and declines to resample them
/// ([`load_audio_mono_sr`](crate::support::audio::load_audio_mono_sr)), because a resample
/// is a signal-processing decision the caller makes deliberately: a
/// microphone opened at another rate, or one model's output read by
/// another.
///
/// Each output sample is the input convolved with a Blackman-windowed sinc
/// ([`kernels::sinc`](crate::ops::signal::kernels::sinc)) centred on its
/// position, cut off at the lower of the two Nyquist rates, and normalized to
/// unit DC gain for its phase. The kernel is symmetric
/// and output sample `n` sits at input position `n * in / out` exactly, so
/// there is no delay to account for, and the output is the same whatever
/// the pushes were cut into.
#[derive(Debug, Clone)]
pub struct StreamingResampler {
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

impl StreamingResampler {
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

    /// `input`, a whole signal at `in_rate`, at `out_rate`: `in_len * out /
    /// in` samples ([`new`](Self::new), then [`finish`](Self::finish)).
    ///
    /// # Panics
    /// If either rate is zero.
    pub fn resample(
        input: &[f32],
        in_rate: usize,
        out_rate: usize,
    ) -> Vec<f32> {
        let mut resampler = Self::new(in_rate, out_rate);
        let mut out = Vec::with_capacity(input.len() * out_rate / in_rate + 1);
        resampler.process(input, &mut out);
        resampler.finish(&mut out);
        out
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

    /// Ends the stream, appending the outputs left to `out`, the input past
    /// its end taken as silence (as before its start): every output whose
    /// position is within the input, `in_len * out / in` in all.
    pub fn finish(
        mut self,
        out: &mut Vec<f32>,
    ) {
        if self.is_identity() {
            return;
        }
        let received = self.released + self.buffer.len() - self.half;
        let total = (received as u64 * self.out_rate as u64 / self.in_rate as u64) as usize;
        let before = self.emitted;
        let mut tail = Vec::new();
        self.process(&vec![0.0; self.half + 1], &mut tail);
        out.extend(tail.into_iter().take(total.saturating_sub(before)));
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
        self.cutoff * normalized_sinc(self.cutoff * x) * BLACKMAN.taper(t)
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
    use core::f64::consts::PI;

    use super::*;
    use crate::ops::signal::kernels::sinc::WindowedSinc;

    /// The kernel is `kernels::sinc`'s windowed sinc in the sinc's units: `t =
    /// cutoff · x`, over `cutoff · half` zero crossings, scaled by the cut-off.
    #[test]
    fn test_kernel_is_the_windowed_sinc() {
        for (in_rate, out_rate) in [(48000, 16000), (44100, 16000), (8000, 16000)] {
            let resampler = StreamingResampler::new(in_rate, out_rate);
            let kernel = WindowedSinc {
                taper: BLACKMAN,
                zero_crossings: resampler.cutoff * resampler.half as f64,
            };
            let half = resampler.half as f64;
            for i in -400..=400 {
                let x = i as f64 * half / 380.0;
                let expected = if x.abs() >= half {
                    0.0
                } else {
                    resampler.cutoff * kernel.at(resampler.cutoff * x)
                };
                let got = resampler.kernel(x);
                assert!(
                    (got - expected).abs() < 1e-15,
                    "{in_rate}->{out_rate} at {x}: {got} vs {expected}"
                );
            }
        }
    }

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
        let mut resampler = StreamingResampler::new(16000, 16000);
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
        let mut resampler = StreamingResampler::new(in_rate, out_rate);
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
        let mut resampler = StreamingResampler::new(48000, 16000);
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
        StreamingResampler::new(44100, 16000).process(&input, &mut whole);

        let mut pieces = Vec::new();
        let mut resampler = StreamingResampler::new(44100, 16000);
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

    /// A whole signal comes out at its full length, its tail included.
    #[test]
    fn test_resample_whole() {
        let input = tone(44100, 1000.0, 1.0);
        let out = StreamingResampler::resample(&input, 44100, 16000);
        assert_eq!(out.len(), 16000);
        // The tail, whose kernel reaches past the end, still holds the tone.
        let tail = &out[16000 - 200..16000 - 20];
        assert!(rms(tail) > 0.6, "the tail's rms {:.3}", rms(tail));

        assert_eq!(StreamingResampler::resample(&input, 44100, 44100), input);
        assert_eq!(
            StreamingResampler::resample(&input[..441], 44100, 8000).len(),
            80
        );
    }
}
