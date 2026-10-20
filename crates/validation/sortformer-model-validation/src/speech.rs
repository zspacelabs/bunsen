//! Diarizations of `bunsen-test-audio`'s AMI excerpt, against `NeMo`'s own.
//!
//! The probabilities are compared elementwise, and the turns by their
//! diarization error against the RTTM: a frame near 0.5 can land either side
//! of the threshold on another backend's last digits, and the error rate is
//! what a diarization is judged on.

use bunsen::{
    kits::speech::{
        diarization::{
            RttmSegment,
            TurnFilterConfig,
        },
        sortformer::SortformerStreamingConfig,
    },
    support::{
        audio::load_audio_mono_sr,
        testing::{
            DeviceMemoryGuard,
            diarization::diarization_error,
            performance_device,
        },
    },
};

use super::*;

/// The clip, the preset, and `NeMo`'s run of it, through bunsen's whole
/// pipeline: the factory, the front end, the speaker cache; then the turns'
/// error against the RTTM, beside `NeMo`'s.
fn check_preset(
    name: &str,
    streaming: SortformerStreamingConfig,
) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let bundle = load_bundle(&device);
    let fixture = ami_reference();
    let reference = &fixture.presets[name];
    let ami = bunsen_test_audio::AMI_EN2002D_2132;

    let samples = load_audio_mono_sr(ami.audio_path(), 16_000).expect("read the AMI clip");
    let diarizer = bundle.diarizer(streaming).unwrap();
    let t0 = std::time::Instant::now();
    let activity = diarizer.activity(&samples).unwrap();
    eprintln!(
        "{name}: {} frames in {:.2?}",
        activity.frames(),
        t0.elapsed()
    );
    let probs = Tensor::<1>::from_floats(activity.probs.as_slice(), &device).reshape([
        1,
        activity.frames(),
        N_SPEAKERS,
    ]);
    reference
        .probs
        .assert_matches(&format!("{name}: probabilities"), probs, 0.0, PROBS_ATOL);

    let turns = TurnFilterConfig::none().turns(&activity.probs, N_SPEAKERS, 0.01);
    let hypothesis: Vec<RttmSegment> = turns
        .iter()
        .map(|t| RttmSegment::from_turn("ami", t))
        .collect();
    let truth = RttmSegment::parse(&std::fs::read_to_string(ami.rttm_path()).unwrap()).unwrap();
    let e = diarization_error(&truth, &hypothesis, 0.25, false);
    eprintln!(
        "{name}: DER {:.4} (confusion {:.4}, false alarm {:.4}, missed {:.4}); NeMo {:.4}",
        e.der(),
        e.confusion / e.total,
        e.false_alarm / e.total,
        e.missed / e.total,
        reference.der.der
    );
    assert!(
        (e.der() - reference.der.der).abs() < DER_MARGIN,
        "{name}: DER {} where NeMo's is {}",
        e.der(),
        reference.der.der
    );
}

/// The largest difference allowed in a probability. Measured on wgpu against
/// `NeMo` on the CPU, the worst is 1.3e-4 at low latency (84 steps through
/// the speaker cache) and 1.5e-5 offline-style; the margin is for backends
/// whose matmuls drift further.
const PROBS_ATOL: f64 = 2e-3;

/// How far bunsen's error rate may sit from `NeMo`'s.
const DER_MARGIN: f64 = 0.01;

/// The low-latency preset (1.04 s): `NeMo`'s probabilities and error rate.
#[test]
#[serial_test::serial]
fn test_low_latency_diarizes_like_nemo() {
    check_preset(
        "low_latency",
        SortformerStreamingConfig::nemotron_low_latency(),
    );
}

/// The offline-style preset (30.4 s): `NeMo`'s probabilities and error rate.
#[test]
#[serial_test::serial]
fn test_offline_diarizes_like_nemo() {
    check_preset("offline", SortformerStreamingConfig::nemotron_offline());
}
