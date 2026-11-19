//! bunsen's Qwen3-TTS held in bf16 on CUDA, over every request's take
//! dictated upstream's greedy codes, against `qwen_tts`' float32: its
//! argmaxes agreeing as often as `qwen_tts`' own bf16's do.
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
use qwen3_tts_model_validation::{
    compare_logits,
    fixtures::{
        FixtureFile,
        REFERENCE,
        REFERENCE_BF16,
        REQUESTS,
    },
    load_construct_with,
    reference_codes,
    reference_request,
    run_forced,
};

/// The reference margin under which a bf16 run may pick either id.
const BF16_NEAR_TIE: f32 = 0.5;

/// How much less often bunsen's bf16 argmaxes, the talker's and the
/// predictor's together, may agree with the float32 reference's than
/// `qwen_tts`' own bf16 ones do. Measured: bunsen's agree 0.865 to 0.896 by
/// request, `qwen_tts`' own bf16's 0.870 to 0.891.
const AGREEMENT_SLACK: f64 = 0.03;

/// The most the top-8 logits' 90th-percentile distance from the float32
/// reference's may be, the talker's or the predictor's. Measured: 0.37 to
/// 0.42 by request, the predictor's the larger.
const TOP8_Q90: f32 = 0.6;

/// How often `qwen_tts`' own bf16 argmaxes agree with its float32 ones over
/// the request `name`, the talker's and the predictor's together.
fn own_agreement(
    reference: &FixtureFile,
    bf16: &FixtureFile,
    name: &str,
) -> f64 {
    let (mut same, mut all) = (0, 0);
    for part in ["talker", "predictor"] {
        let key = format!("{name}.{part}.argmax");
        for (a, b) in reference.ints(&key).iter().zip(bf16.ints(&key)) {
            same += usize::from(*a == b);
            all += 1;
        }
    }
    same as f64 / all as f64
}

mod qwen3_tts_12hz_0_6b_customvoice {
    use super::*;

    /// Every request's take, within the bf16 gates.
    #[test]
    fn bf16() {
        let device = Device::cuda(0);
        let construct = load_construct_with(&device, LoadPolicy::performance());
        assert_eq!(
            construct.model.talker.lm.float_dtype(),
            DType::BF16,
            "the talker loads in bf16"
        );
        let reference = FixtureFile::load(REFERENCE);
        let bf16 = FixtureFile::load(REFERENCE_BF16);
        let mut failures = Vec::new();
        for name in REQUESTS {
            let request = reference_request(&construct.prompt, &reference, name);
            let codes = reference_codes(&reference, name);
            let run = run_forced(&construct, &request, &codes, &device);
            let [talker, predictor] = compare_logits(&run, &reference, name, BF16_NEAR_TIE);
            let agreement = (talker.argmaxes - talker.argmax_differ + predictor.argmaxes
                - predictor.argmax_differ) as f64
                / (talker.argmaxes + predictor.argmaxes) as f64;
            let q90 = talker.top8_quantile(0.9).max(predictor.top8_quantile(0.9));
            let own = own_agreement(&reference, &bf16, name);
            eprintln!(
                "{name}: argmax agreement {agreement:.3} (qwen_tts' own bf16 {own:.3}; talker \
                 {:.3}, predictor {:.3}), top8 q90 {q90:.3} (talker {:.3}, predictor {:.3})",
                talker.agreement(),
                predictor.agreement(),
                talker.top8_quantile(0.9),
                predictor.top8_quantile(0.9),
            );
            if agreement < own - AGREEMENT_SLACK {
                failures.push(format!(
                    "{name}: agreement {agreement}, qwen_tts' own {own}"
                ));
            }
            if q90 > TOP8_Q90 {
                failures.push(format!("{name}: top8 q90 {q90}"));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
