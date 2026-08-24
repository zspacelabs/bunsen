//! # Cascaded biquad IIR filtering.
//!
//! A series of second-order sections, each in Direct Form II:
//!
//! ```text
//! w0 = x - a1*w1 - a2*w2
//! y  = g * (b0*w0 + b1*w1 + b2*w2)
//! w2 = w1;  w1 = w0
//! ```
//!
//! Sections are applied in series, each carrying its own two-sample state
//! across calls, so a stream may be filtered in arbitrary blocks and get the
//! same answer as filtering it whole.
//!
//! Each section is run over the entire block before the next one starts.
//! Because a section's state depends only on its own input sequence, that is
//! equivalent to advancing every section per sample, and both are equivalent
//! to filtering the unsegmented stream. [`BiquadCascade::process_in_place`]
//! takes the first form because it keeps one section's coefficients in
//! registers for a whole block.
//!
//! ## Host-side, and deliberately so
//!
//! This is scalar `f32` code with no tensor in it. A cascade is a sample-rate
//! recurrence, so a device implementation would be sequential in the sample
//! axis and slower than the host for any realistic block. When a *decimating*
//! cascade is what you want on device, realize it as a truncated FIR instead
//! and fold the decimation into the kernel — see
//! [`DecimatingFirConfig`](super::DecimatingFirConfig), which generates its
//! impulse response by driving this type with a unit impulse. That form is
//! both parallel and better conditioned than the recurrence.
//!
//! `a[0]` is assumed to be `1`. Sections whose leading denominator coefficient
//! is not already normalized must be divided through before construction.

/// One second-order section of a [`BiquadCascade`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadSection {
    /// Numerator coefficients `[b0, b1, b2]`.
    pub b: [f32; 3],

    /// Denominator coefficients `[_, a1, a2]`; `a[0]` is assumed `1` and
    /// ignored.
    pub a: [f32; 3],

    /// The section's output gain.
    pub g: f32,
}

impl BiquadSection {
    /// A section that passes its input through unchanged.
    pub const IDENTITY: Self = Self {
        b: [1.0, 0.0, 0.0],
        a: [1.0, 0.0, 0.0],
        g: 1.0,
    };

    /// The section's DC gain, `H(1)`.
    ///
    /// Useful as a cheap sanity check on a coefficient table: a lowpass
    /// section's cascade gain should be near `1`, and a value far from the
    /// intended one usually means a section was transcribed with the wrong
    /// sign convention on `a`.
    pub fn dc_gain(&self) -> f32 {
        let num = self.b[0] + self.b[1] + self.b[2];
        let den = 1.0 + self.a[1] + self.a[2];
        self.g * num / den
    }
}

/// A series of [`BiquadSection`]s with per-section delay state.
///
/// Built by [`BiquadCascade::new`] or
/// [`BiquadCascade::from_tables`]; rewound by [`BiquadCascade::reset`].
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

    /// Builds a cascade from parallel coefficient tables.
    ///
    /// The layout filter-design tools usually emit: one row per section in
    /// each of three arrays, rather than an array of sections.
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

    /// The sections, in application order.
    pub fn sections(&self) -> &[BiquadSection; NSECT] {
        &self.sections
    }

    /// The number of sections.
    pub fn len(&self) -> usize {
        NSECT
    }

    /// Whether the cascade has no sections, in which case it is a
    /// pass-through.
    pub fn is_empty(&self) -> bool {
        NSECT == 0
    }

    /// The whole cascade's DC gain, `H(1)`.
    pub fn dc_gain(&self) -> f32 {
        self.sections.iter().map(BiquadSection::dc_gain).product()
    }

    /// Zeroes every section's delay state.
    pub fn reset(&mut self) {
        self.state = [[0.0; 2]; NSECT];
    }

    /// The cascade's impulse response, `taps` long, from a zeroed state.
    ///
    /// Does not disturb this cascade's state; it runs against a clone.
    pub fn to_vec_impulse_response(
        &self,
        taps: usize,
    ) -> Vec<f32> {
        let mut fresh = Self::new(self.sections);
        let mut buf = vec![0.0f32; taps];
        if let Some(first) = buf.first_mut() {
            *first = 1.0;
        }
        fresh.process_in_place(&mut buf);
        buf
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
    use super::*;

    /// A one-pole section: `y[n] = x[n] + r*y[n-1]`.
    ///
    /// Its impulse response is exactly `r^n`, which makes it a closed-form
    /// oracle rather than a fixture.
    fn one_pole(r: f32) -> BiquadCascade<1> {
        BiquadCascade::new([BiquadSection {
            b: [1.0, 0.0, 0.0],
            a: [1.0, -r, 0.0],
            g: 1.0,
        }])
    }

    /// A two-pole section with real poles at `0.7` and `0.5`.
    fn two_pole() -> BiquadCascade<1> {
        BiquadCascade::new([BiquadSection {
            b: [1.0, 0.0, 0.0],
            a: [1.0, -1.2, 0.35],
            g: 1.0,
        }])
    }

    #[test]
    fn test_empty_cascade_is_pass_through() {
        let mut cascade: BiquadCascade<0> = BiquadCascade::new([]);
        assert!(cascade.is_empty());
        assert_eq!(cascade.len(), 0);

        let mut buf = [1.0, -2.0, 3.5, 0.0];
        cascade.process_in_place(&mut buf);
        assert_eq!(buf, [1.0, -2.0, 3.5, 0.0]);
    }

    #[test]
    fn test_identity_section_is_pass_through() {
        let mut cascade = BiquadCascade::new([BiquadSection::IDENTITY; 3]);
        assert!(!cascade.is_empty());

        let mut buf = [1.0, -2.0, 3.5, 0.0];
        cascade.process_in_place(&mut buf);
        assert_eq!(buf, [1.0, -2.0, 3.5, 0.0]);
    }

    #[test]
    fn test_one_pole_impulse_response_is_geometric() {
        // The point of a closed-form oracle: this checks the difference
        // equation itself, not agreement with another implementation.
        let r = 0.5f32;
        let response = one_pole(r).to_vec_impulse_response(24);

        for (n, &got) in response.iter().enumerate() {
            let want = r.powi(n as i32);
            assert!(
                (got - want).abs() <= 1e-6 * want.max(1e-6),
                "tap {n}: got {got}, want {want}",
            );
        }
    }

    #[test]
    fn test_dc_gain_matches_the_summed_impulse_response() {
        // H(1) is by definition the sum of the impulse response, so these two
        // routes to the same number must agree. Catches a sign-convention slip
        // on `a` immediately.
        for cascade in [one_pole(0.5), one_pole(0.9), two_pole()] {
            let summed: f32 = cascade.to_vec_impulse_response(4096).iter().sum();
            let analytic = cascade.dc_gain();
            assert!(
                (summed - analytic).abs() < 1e-3 * analytic.abs(),
                "summed {summed} vs analytic {analytic}",
            );
        }
    }

    #[test]
    fn test_two_pole_matches_its_difference_equation() {
        let response = two_pole().to_vec_impulse_response(32);

        // y[n] = x[n] + 1.2*y[n-1] - 0.35*y[n-2]
        let mut want = vec![0.0f64; 32];
        for n in 0..32 {
            let x = if n == 0 { 1.0 } else { 0.0 };
            let y1 = if n >= 1 { want[n - 1] } else { 0.0 };
            let y2 = if n >= 2 { want[n - 2] } else { 0.0 };
            want[n] = x + 1.2 * y1 - 0.35 * y2;
        }

        for (n, (&got, &w)) in response.iter().zip(want.iter()).enumerate() {
            assert!(
                (got as f64 - w).abs() <= 1e-5 * w.abs().max(1e-6),
                "tap {n}: got {got}, want {w}",
            );
        }
    }

    #[test]
    fn test_state_carries_across_calls() {
        // The property that makes this usable on a stream: block boundaries
        // must not be observable.
        let signal: Vec<f32> = (0..64).map(|n| ((n as f32) * 0.37).sin()).collect();

        let mut whole = signal.clone();
        one_pole(0.8).process_in_place(&mut whole);

        let mut split = signal.clone();
        let mut cascade = one_pole(0.8);
        let (head, tail) = split.split_at_mut(19);
        cascade.process_in_place(head);
        cascade.process_in_place(tail);

        assert_eq!(whole, split);
    }

    #[test]
    fn test_reset_rewinds_the_state() {
        let mut cascade = one_pole(0.8);

        let mut first = [1.0, 0.0, 0.0, 0.0];
        cascade.process_in_place(&mut first);

        let mut dirty = [1.0, 0.0, 0.0, 0.0];
        cascade.process_in_place(&mut dirty);
        assert_ne!(first, dirty, "state should have carried");

        cascade.reset();
        let mut again = [1.0, 0.0, 0.0, 0.0];
        cascade.process_in_place(&mut again);
        assert_eq!(first, again);
    }

    #[test]
    fn test_sections_apply_in_series() {
        let signal: Vec<f32> = (0..48).map(|n| ((n as f32) * 0.21).cos()).collect();

        let mut chained = signal.clone();
        one_pole(0.6).process_in_place(&mut chained);
        one_pole(0.3).process_in_place(&mut chained);

        let mut cascaded = signal;
        BiquadCascade::new([one_pole(0.6).sections()[0], one_pole(0.3).sections()[0]])
            .process_in_place(&mut cascaded);

        assert_eq!(chained, cascaded);
    }

    #[test]
    fn test_impulse_response_does_not_disturb_state() {
        let mut cascade = one_pole(0.8);

        let baseline = cascade.to_vec_impulse_response(8);
        let again = cascade.to_vec_impulse_response(8);
        assert_eq!(
            baseline, again,
            "generating a response must not carry state"
        );

        let mut buf = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        cascade.process_in_place(&mut buf);
        assert_eq!(&buf[..], &baseline[..]);
    }

    #[test]
    fn test_from_tables_matches_explicit_sections() {
        let b = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]];
        let a = [[1.0, -0.5, 0.0], [1.0, -0.2, 0.1]];
        let g = [1.0, 2.0];

        let tabled = BiquadCascade::<2>::from_tables(&b, &a, &g);
        let explicit = BiquadCascade::new([
            BiquadSection {
                b: b[0],
                a: a[0],
                g: g[0],
            },
            BiquadSection {
                b: b[1],
                a: a[1],
                g: g[1],
            },
        ]);

        assert_eq!(tabled, explicit);
    }
}
