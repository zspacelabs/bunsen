//! # Cascaded biquad IIR filter.
//!
//! The anti-alias filter the reference pitch estimator runs before decimating
//! 16 kHz down to 4 kHz (`src/biquad.cc`).
//!
//! Each section is Direct Form II:
//!
//! ```text
//! w0 = x - a1*w1 - a2*w2
//! y  = g * (b0*w0 + b1*w1 + b2*w2)
//! w2 = w1;  w1 = w0
//! ```
//!
//! and sections are applied in series, each carrying its own two-sample
//! state across calls. Sections are run one at a time over the whole block,
//! matching the reference; because each section's state depends only on its
//! own input sequence, that is equivalent to running every section per
//! sample, and both are equivalent to filtering the unsegmented stream.
//!
//! Like the rest of the pitch branch this is host-side scalar code. It is a
//! plain DSP primitive with nothing ten-vad-specific in it but the
//! coefficients in [`super::coeff`], so it would move to
//! [`crate::ops::signal`] unchanged if a tensor-side caller ever wanted it.

/// One second-order section of a [`BiquadCascade`].
///
/// `a[0]` is assumed to be `1`, as it is in every tabulated ten-vad section;
/// the difference equation ignores it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadSection {
    /// Numerator coefficients `[b0, b1, b2]`.
    pub b: [f32; 3],

    /// Denominator coefficients `[_, a1, a2]`; `a[0]` is unused.
    pub a: [f32; 3],

    /// The section's output gain.
    pub g: f32,
}

/// A series of [`BiquadSection`]s with per-section state.
///
/// Built by [`BiquadCascade::new`]; reset by [`BiquadCascade::reset`].
#[derive(Debug, Clone, PartialEq)]
pub struct BiquadCascade<const NSECT: usize> {
    sections: [BiquadSection; NSECT],

    /// Per-section `[w1, w2]` delay state.
    state: [[f32; 2]; NSECT],
}

impl<const NSECT: usize> BiquadCascade<NSECT> {
    /// Builds a cascade with zeroed state.
    pub fn new(sections: [BiquadSection; NSECT]) -> Self {
        Self {
            sections,
            state: [[0.0; 2]; NSECT],
        }
    }

    /// Builds a cascade from the reference's parallel coefficient tables.
    ///
    /// # Arguments
    /// * `b`: per-section numerator coefficients.
    /// * `a`: per-section denominator coefficients.
    /// * `g`: per-section gains.
    pub fn from_tables(
        b: &[[f32; 3]; NSECT],
        a: &[[f32; 3]; NSECT],
        g: &[f32; NSECT],
    ) -> Self {
        Self::new(core::array::from_fn(|i| BiquadSection {
            b: b[i],
            a: a[i],
            g: g[i],
        }))
    }

    /// The number of sections.
    pub fn len(&self) -> usize {
        NSECT
    }

    /// Whether the cascade has no sections, in which case it is a pass-through.
    pub fn is_empty(&self) -> bool {
        NSECT == 0
    }

    /// Zeroes every section's delay state.
    pub fn reset(&mut self) {
        self.state = [[0.0; 2]; NSECT];
    }

    /// Filters `buf` in place, carrying state across calls.
    pub fn process_in_place(
        &mut self,
        buf: &mut [f32],
    ) {
        for (section, state) in self.sections.iter().zip(self.state.iter_mut()) {
            let [b0, b1, b2] = section.b;
            let a1 = section.a[1];
            let a2 = section.a[2];
            let g = section.g;

            let [mut w1, mut w2] = *state;
            for y in buf.iter_mut() {
                let w0 = *y - a1 * w1 - a2 * w2;
                *y = g * (b0 * w0 + b1 * w1 + b2 * w2);
                w2 = w1;
                w1 = w0;
            }
            *state = [w1, w2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::coeff::{
            ANTI_ALIAS_A_4KHZ,
            ANTI_ALIAS_B_4KHZ,
            ANTI_ALIAS_G_4KHZ,
            ANTI_ALIAS_SECTIONS,
        },
        *,
    };

    fn anti_alias() -> BiquadCascade<ANTI_ALIAS_SECTIONS> {
        BiquadCascade::from_tables(&ANTI_ALIAS_B_4KHZ, &ANTI_ALIAS_A_4KHZ, &ANTI_ALIAS_G_4KHZ)
    }

    /// A single section reproducing the difference equation by hand.
    fn reference_section(
        section: &BiquadSection,
        input: &[f32],
    ) -> Vec<f32> {
        let (mut w1, mut w2) = (0.0f32, 0.0f32);
        input
            .iter()
            .map(|&x| {
                let w0 = x - section.a[1] * w1 - section.a[2] * w2;
                let y = section.g * (section.b[0] * w0 + section.b[1] * w1 + section.b[2] * w2);
                w2 = w1;
                w1 = w0;
                y
            })
            .collect()
    }

    #[test]
    fn test_single_section_matches_the_difference_equation() {
        let section = BiquadSection {
            b: ANTI_ALIAS_B_4KHZ[0],
            a: ANTI_ALIAS_A_4KHZ[0],
            g: ANTI_ALIAS_G_4KHZ[0],
        };
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.3).sin()).collect();

        let mut cascade = BiquadCascade::new([section]);
        let mut got = input.clone();
        cascade.process_in_place(&mut got);

        assert_eq!(got, reference_section(&section, &input));
    }

    #[test]
    fn test_block_split_matches_whole_stream() {
        // The state carry is the whole point: a stream chopped into hops must
        // filter identically to the same stream filtered at once.
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.11).sin() * 1000.0).collect();

        let mut whole = input.clone();
        anti_alias().process_in_place(&mut whole);

        let mut split = input.clone();
        let mut cascade = anti_alias();
        for chunk in split.chunks_mut(64) {
            cascade.process_in_place(chunk);
        }

        assert_eq!(whole, split);
    }

    #[test]
    fn test_reset_restores_start_of_stream() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.7).cos()).collect();
        let mut cascade = anti_alias();

        let mut first = input.clone();
        cascade.process_in_place(&mut first);

        // Without a reset the tail state colors the next block.
        let mut carried = input.clone();
        cascade.process_in_place(&mut carried);
        assert_ne!(first, carried);

        cascade.reset();
        let mut after_reset = input.clone();
        cascade.process_in_place(&mut after_reset);
        assert_eq!(first, after_reset);
    }

    #[test]
    fn test_sections_apply_in_series() {
        let input: Vec<f32> = (0..96).map(|i| (i as f32 * 0.23).sin()).collect();

        let mut got = input.clone();
        anti_alias().process_in_place(&mut got);

        let mut expected = input;
        for i in 0..ANTI_ALIAS_SECTIONS {
            expected = reference_section(
                &BiquadSection {
                    b: ANTI_ALIAS_B_4KHZ[i],
                    a: ANTI_ALIAS_A_4KHZ[i],
                    g: ANTI_ALIAS_G_4KHZ[i],
                },
                &expected,
            );
        }

        assert_eq!(got, expected);
    }

    #[test]
    fn test_anti_alias_attenuates_above_2khz() {
        // Decimating 16 kHz by 4 puts the new Nyquist at 2 kHz; the cascade
        // exists to keep everything above it out of the correlation branch.
        let response = |freq_hz: f32| {
            let input: Vec<f32> = (0..4096)
                .map(|i| (core::f32::consts::TAU * freq_hz * i as f32 / 16000.0).sin())
                .collect();
            let mut out = input;
            anti_alias().process_in_place(&mut out);
            // Peak amplitude over the settled tail.
            out[2048..].iter().fold(0.0f32, |m, v| m.max(v.abs()))
        };

        let passband = response(300.0);
        let stopband = response(4000.0);

        assert!(passband > 0.5, "300 Hz should pass: {passband}");
        assert!(stopband < 0.02, "4 kHz should be rejected: {stopband}");
    }

    #[test]
    fn test_empty_cascade_is_pass_through() {
        let mut cascade: BiquadCascade<0> = BiquadCascade::new([]);
        assert!(cascade.is_empty());
        assert_eq!(cascade.len(), 0);

        let mut buf = [1.0, -2.0, 3.0];
        cascade.process_in_place(&mut buf);
        assert_eq!(buf, [1.0, -2.0, 3.0]);
    }
}
