//! bunsen's prompt and float32 logits against `qwen_tts`' float32, over
//! every request's take, dictated upstream's greedy codes.

use bunsen::support::testing::{
    DeviceMemoryGuard,
    performance_device,
};

use super::*;
use crate::fixtures::{
    REFERENCE,
    REQUESTS,
    SAMPLE_STRIDE,
};

/// The reference margin (top-1 minus top-2) under which two float32
/// implementations may pick either id.
pub const NEAR_TIE: f32 = 0.02;

/// The most a top-8 logit may stray from the reference's. Measured: at most
/// 1.1e-4 on wgpu, the predictor's.
pub const LOGIT_ABS: f32 = 5e-4;

/// The most a logit row's sums may stray, relative to its absolute sum.
/// Measured: at most 4.5e-6.
pub const SUMS_REL: f64 = 5e-5;

/// The most a prompt row's norm, or a sampled prompt value, may stray.
/// Measured: at most 9.5e-7.
pub const PROMPT_ABS: f32 = 1e-5;

mod qwen3_tts_12hz_0_6b_customvoice {
    use super::*;

    /// Every request: the prompt's rows and trailing text as upstream's;
    /// the talker's and the code predictor's logits with no argmax
    /// differing but at a near tie, every top-8 logit within
    /// [`LOGIT_ABS`], the rows' sums within [`SUMS_REL`].
    #[test]
    #[serial_test::serial]
    fn logits() {
        let device = performance_device();
        let _memory = DeviceMemoryGuard::new(&device);
        let construct = load_construct(&device);
        let reference = FixtureFile::load(REFERENCE);
        for name in REQUESTS {
            let request = reference_request(&construct.prompt, &reference, name);
            let codes = reference_codes(&reference, name);
            let run = run_forced(&construct, &request, &codes, &device);

            let near = |got: &[f32], want: &[f32], what: &str| {
                assert_eq!(got.len(), want.len(), "{name}: {what}");
                let gap = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(gap <= PROMPT_ABS, "{name}: {what} off by {gap}");
                gap
            };
            let norms = near(
                &run.prompt_norms,
                &reference.floats(&format!("{name}.prompt.norms")),
                "prompt norms",
            );
            let sample: Vec<f32> = run.prompt.iter().step_by(SAMPLE_STRIDE).copied().collect();
            near(
                &sample,
                &reference.floats(&format!("{name}.prompt.sample")),
                "prompt sample",
            );
            near(
                &run.trailing_norms,
                &reference.floats(&format!("{name}.trailing.norms")),
                "trailing norms",
            );

            let [talker, predictor] = compare_logits(&run, &reference, name, NEAR_TIE);
            for (part, report) in [("talker", &talker), ("predictor", &predictor)] {
                eprintln!(
                    "{name} {part}: argmax {}/{} differ ({} decided, worst margin {:.4}), \
                     top8 max {:.2e}, sums rel {:.1e}; prompt norms {norms:.1e}",
                    report.argmax_differ,
                    report.argmaxes,
                    report.argmax_differ_decided,
                    report.worst_differing_margin,
                    report.top8_max(),
                    report.sums_rel,
                );
                assert_eq!(report.argmax_differ_decided, 0, "{name} {part}: {report:?}");
                assert!(
                    report.top8_max() <= LOGIT_ABS,
                    "{name} {part}: top8 {}",
                    report.top8_max()
                );
                assert!(
                    report.sums_rel <= SUMS_REL,
                    "{name} {part}: sums {}",
                    report.sums_rel
                );
            }
        }
    }
}
