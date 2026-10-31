//! Diarizations of `bunsen-test-audio`'s AMI excerpt, against `NeMo`'s own:
//! `Nemotron-3-Diarization`'s, and the streaming Sortformer v2's and v2.1's.
//!
//! The probabilities are compared elementwise, and the turns by their
//! diarization error against the RTTM: a frame near 0.5 can land either side
//! of the threshold on another backend's last digits, and the error rate is
//! what a diarization is judged on.

use bunsen::{
    kits::speech::{
        diarization::{
            RttmSegment,
            SpeakerActivityModel,
            TurnFilterConfig,
        },
        sortformer::{
            SortformerLatency,
            SortformerStreamingConfig,
        },
        testing::der::diarization_error,
    },
    support::{
        audio::load_audio_mono_sr,
        testing::{
            DeviceMemoryGuard,
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
    check_diarization(
        name,
        &bundle,
        streaming,
        &ami_reference().presets[name],
        PROBS_ATOL,
        &device,
    );
}

/// `bundle` diarizes the clip under `streaming` as `reference` records
/// `NeMo` did: its probabilities to `atol`, its turns' error against the
/// RTTM to [`DER_MARGIN`].
fn check_diarization(
    what: &str,
    bundle: &bunsen::kits::speech::sortformer::SortformerBundle,
    streaming: SortformerStreamingConfig,
    reference: &AmiPreset,
    atol: f64,
    device: &Device,
) {
    let ami = bunsen_test_audio::AMI_EN2002D_2132;
    let samples = load_audio_mono_sr(ami.audio_path(), 16_000).expect("read the AMI clip");
    let diarizer = bundle.diarizer(streaming).unwrap();
    let n_speakers = diarizer.max_speakers();
    let frame_seconds = diarizer.frame_samples() as f64 / diarizer.sample_rate() as f64;
    let t0 = std::time::Instant::now();
    let activity = diarizer.activity(&samples).unwrap();
    eprintln!(
        "{what}: {} frames in {:.2?}",
        activity.frames(),
        t0.elapsed()
    );
    let probs = Tensor::<1>::from_floats(activity.probs.as_slice(), device).reshape([
        1,
        activity.frames(),
        n_speakers,
    ]);
    let error = reference
        .probs
        .assert_matches(&format!("{what}: probabilities"), probs, 0.0, atol);

    let turns = TurnFilterConfig::none().turns(&activity.probs, n_speakers, frame_seconds);
    let hypothesis: Vec<RttmSegment> = turns
        .iter()
        .map(|t| RttmSegment::from_turn("ami", t))
        .collect();
    let truth = RttmSegment::parse(&std::fs::read_to_string(ami.rttm_path()).unwrap()).unwrap();
    let e = diarization_error(&truth, &hypothesis, 0.25, false);
    eprintln!(
        "{what}: probabilities {error}; DER {:.4} (confusion {:.4}, false alarm {:.4}, missed {:.4}); NeMo {:.4}",
        e.der(),
        e.confusion / e.total,
        e.false_alarm / e.total,
        e.missed / e.total,
        reference.der.der
    );
    assert!(
        (e.der() - reference.der.der).abs() < DER_MARGIN,
        "{what}: DER {} where NeMo's is {}",
        e.der(),
        reference.der.der
    );
}

/// The streaming Sortformer `model` (`v2` or `v2_1`), loaded by its
/// well-known name, diarizes the clip at its family's `latency` preset as
/// `NeMo` did at `preset`.
fn check_v2(
    model: &str,
    name: &str,
    preset: &str,
    latency: SortformerLatency,
) {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let bundle = load_named_bundle(name, &device);
    let streaming = bundle.preset(latency).unwrap();
    check_diarization(
        &format!("{name} {preset}"),
        &bundle,
        streaming,
        &ami_reference_v2().models[model].presets[preset],
        V2_PROBS_ATOL,
        &device,
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

/// The largest difference allowed in a v2 probability. Measured on wgpu
/// against `NeMo` on the CPU, the worst is 2.5e-6 over both models and both
/// presets; the margin is [`PROBS_ATOL`]'s.
const V2_PROBS_ATOL: f64 = 2e-3;

/// Sortformer v2 at low latency (1.04 s): `NeMo`'s probabilities and error
/// rate.
#[test]
#[serial_test::serial]
fn test_v2_low_latency_diarizes_like_nemo() {
    check_v2("v2", "sortformer-v2", "low_latency", SortformerLatency::Low);
}

/// Sortformer v2 at very high latency (30.4 s).
#[test]
#[serial_test::serial]
fn test_v2_very_high_latency_diarizes_like_nemo() {
    check_v2(
        "v2",
        "sortformer-v2",
        "very_high_latency",
        SortformerLatency::Offline,
    );
}

/// Sortformer v2.1 at low latency (1.04 s).
#[test]
#[serial_test::serial]
fn test_v2_1_low_latency_diarizes_like_nemo() {
    check_v2(
        "v2_1",
        "sortformer-v2.1",
        "low_latency",
        SortformerLatency::Low,
    );
}

/// Sortformer v2.1 at very high latency (30.4 s).
#[test]
#[serial_test::serial]
fn test_v2_1_very_high_latency_diarizes_like_nemo() {
    check_v2(
        "v2_1",
        "sortformer-v2.1",
        "very_high_latency",
        SortformerLatency::Offline,
    );
}
