//! bunsen's float32 logits against the reference's float32, over every
//! request: the prefill and the teacher-forced decode steps.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;
use crate::fixtures::{
    REFERENCE,
    REQUESTS,
};

/// The reference margin (top-1 minus top-2) under which two float32
/// implementations may pick either id.
pub const NEAR_TIE: f32 = 0.02;

/// The most a top-8 or sampled logit may stray from the reference's.
pub const LOGIT_ABS: f32 = 5e-4;

/// The most the prefill logits' sums may stray, relative to their absolute
/// sum.
pub const SUMS_REL: f64 = 2e-6;

mod zonos3_mini {
    use super::*;

    /// Every request's prefill and steps: no argmax differs but at a near
    /// tie, every top-8 and sampled logit within [`LOGIT_ABS`], the sums
    /// within [`SUMS_REL`].
    #[test]
    #[serial_test::serial]
    fn logits() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let construct = load_construct(&device);
        let reference = FixtureFile::load(REFERENCE);
        for name in REQUESTS {
            let prompt = reference_prompt(&reference, name);
            let run = run_lm(
                &construct,
                &prompt,
                &reference_generated(&reference, name),
                &device,
            );
            let stride = reference.request(name).stats_stride;
            let report = compare_logits(&run, &reference, name, stride, NEAR_TIE);
            eprintln!(
                "{name}: argmax {}/{} differ ({} decided, worst margin {:.4}), top8 max {:.2e}, \
                 sample max {:.2e}, sums rel {:.1e} {:.1e}",
                report.argmax_differ,
                report.argmaxes,
                report.argmax_differ_decided,
                report.worst_differing_margin,
                report.top8_max(),
                report.sample_max(),
                report.sums_rel[0],
                report.sums_rel[1],
            );
            assert_eq!(report.argmax_differ_decided, 0, "{name}: {report:?}");
            assert!(
                report.top8_max() <= LOGIT_ABS,
                "{name}: top8 {}",
                report.top8_max()
            );
            assert!(
                report.sample_max() <= LOGIT_ABS,
                "{name}: sample {}",
                report.sample_max()
            );
            assert!(
                report.sums_rel.iter().all(|&r| r <= SUMS_REL),
                "{name}: sums {:?}",
                report.sums_rel
            );
        }
    }
}
