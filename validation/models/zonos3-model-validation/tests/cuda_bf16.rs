//! bunsen's ZONOS3-Mini held in bf16 on CUDA, over every request,
//! teacher-forced over the float32 reference's rows, against the float32
//! reference: no further from it than the reference's own bf16 is.
//!
//! The device is CUDA's, its default float f32; the weights load in bf16
//! under [`LoadPolicy::performance`]. A test binary of its own, so the f32
//! suites stay off CUDA. Needs `checkpoints`, `gpu-tests` and `cuda`.
#![cfg(all(feature = "checkpoints", feature = "gpu-tests", feature = "cuda"))]

use bunsen::data::formats::LoadPolicy;
use burn::{
    prelude::*,
    tensor::DType,
};
use zonos3_model_validation::{
    compare_logits,
    fixtures::{
        FixtureFile,
        REFERENCE,
        REFERENCE_BF16,
        REQUESTS,
    },
    load_construct_with,
    reference_generated,
    reference_prompt,
    run_lm,
};

/// The reference margin under which two bf16 implementations may pick
/// either id.
const BF16_NEAR_TIE: f32 = 0.5;

/// How much less often bunsen's bf16 argmaxes may agree with the float32
/// reference's than the reference's own bf16 ones do.
const AGREEMENT_SLACK: f64 = 0.03;

/// The most the top-8 logits' 90th-percentile distance from the float32
/// reference's may be. The reference's own bf16 is 0.13 to 0.63 from it, by
/// request; bunsen's, 0.15 to 0.58.
const TOP8_Q90: f32 = 0.75;

/// The most the prefill logits' sums may stray from the float32 reference's,
/// relative to their absolute sum.
const SUMS_REL: f64 = 5e-3;

/// How often the reference's own bf16 argmaxes agree with its float32 ones
/// over the request `name`.
fn own_agreement(
    reference: &FixtureFile,
    bf16: &FixtureFile,
    name: &str,
) -> f64 {
    let (mut same, mut all) = (0, 0);
    for part in ["prefill", "steps"] {
        let key = format!("{name}.{part}.argmax");
        for (a, b) in reference.ints(&key).iter().zip(bf16.ints(&key)) {
            same += usize::from(*a == b);
            all += 1;
        }
    }
    same as f64 / all as f64
}

mod zonos3_mini {
    use super::*;

    /// Every request's prefill and steps, within the bf16 gates.
    #[test]
    fn bf16() {
        let device = Device::cuda(0);
        let construct = load_construct_with(&device, LoadPolicy::performance());
        assert_eq!(
            construct.lm.head.weight.val().dtype(),
            DType::BF16,
            "performance() keeps bf16 where CUDA accelerates it"
        );
        let reference = FixtureFile::load(REFERENCE);
        let bf16 = FixtureFile::load(REFERENCE_BF16);
        for name in REQUESTS {
            let run = run_lm(
                &construct,
                &reference_prompt(&reference, name),
                &reference_generated(&reference, name),
                &device,
            );
            let stride = reference.request(name).stats_stride;
            let own = own_agreement(&reference, &bf16, name);
            let r = compare_logits(&run, &reference, name, stride, BF16_NEAR_TIE);
            eprintln!(
                "{name}: agree {:.4} (the reference's own bf16 {own:.4}; {}/{} differ, worst \
                 margin {:.3}), top8 q90 {:.3} max {:.3}, sums rel {:.1e} {:.1e}",
                r.agreement(),
                r.argmax_differ,
                r.argmaxes,
                r.worst_differing_margin,
                r.top8_quantile(0.9),
                r.top8_max(),
                r.sums_rel[0],
                r.sums_rel[1],
            );
            assert!(
                r.agreement() >= own - AGREEMENT_SLACK,
                "{name}: agreement {} under the reference's own bf16 {own}",
                r.agreement()
            );
            assert!(
                r.top8_quantile(0.9) <= TOP8_Q90,
                "{name}: top8 q90 {}",
                r.top8_quantile(0.9)
            );
            assert!(
                r.sums_rel.iter().all(|&x| x <= SUMS_REL),
                "{name}: sums {:?}",
                r.sums_rel
            );
        }
    }
}
