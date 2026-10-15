//! Transcriptions of `jfk_moon.mp3`.
//!
//! Token ids are compared exactly and their differences printed, but the gates
//! are on text: a greedy argmax over 8193 logits can flip on a backend's last
//! digits and the text barely move.

use bunsen::{
    kits::speech::parakeet::{
        ParakeetBundle,
        ParakeetFrontEndMeta,
        pretrained::DEFAULT_PARAKEET_MODEL,
    },
    support::testing::{
        DeviceMemoryGuard,
        asr::{
            normalize_transcript,
            word_error_rate,
        },
        performance_device,
    },
};

use super::*;

/// The clip every test here transcribes.
const CLIP: &str = "jfk_moon.mp3";

/// Prints where two id sequences part.
fn report_ids(
    what: &str,
    ours: &[usize],
    theirs: &[usize],
) {
    match ours.iter().zip(theirs).position(|(a, b)| a != b) {
        None if ours.len() == theirs.len() => eprintln!("{what}: ids identical ({})", ours.len()),
        first => eprintln!(
            "{what}: ids differ at {:?} (ours {}, theirs {})",
            first,
            ours.len(),
            theirs.len()
        ),
    }
}

fn wer(
    hypothesis: &str,
    reference: &str,
) -> f64 {
    word_error_rate(
        &normalize_transcript(hypothesis),
        &normalize_transcript(reference),
    )
}

fn transcribe(
    bundle: &ParakeetBundle,
    device: &Device,
) -> bunsen::kits::speech::parakeet::ParakeetTranscript {
    let samples = speech_clip(CLIP);
    let n = samples.len();
    let waveform = Tensor::<2>::from_data(TensorData::new(samples, [1, n]), device);
    bundle
        .transcribe(waveform, None)
        .expect("transcribe")
        .remove(0)
}

/// The reference graphs, behind bunsen's front end, decode what onnxruntime
/// decodes behind the export's own front end.
#[test]
#[serial_test::serial]
fn test_reference_graphs_behind_bunsen_front_end() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference().speech;

    let front_end: bunsen::kits::speech::parakeet::ParakeetFrontEnd =
        bunsen::burner::module::ModuleInit::init(
            &bunsen::kits::speech::parakeet::ParakeetFrontEndConfig::new(),
            &device,
        );
    let samples = speech_clip(CLIP);
    assert_eq!(samples.len(), fixture.n_samples);
    let n = samples.len();
    let (features, frames) = front_end.forward(
        Tensor::<2>::from_data(TensorData::new(samples, [1, n]), &device),
        None,
    );
    assert_eq!(frames, [front_end.frame_len(n)]);
    let features = features.slice_dim(2, 0..frames[0]);

    let encoder = reference::EncoderModel::load_pretrained(&device);
    let length = Tensor::<1, Int>::from_data(TensorData::from([frames[0] as i64]), &device);
    let (encoded, lengths) = encoder.forward(features, length);
    let length = lengths.into_data().try_into_vec_as::<i64>().unwrap()[0] as usize;
    assert_eq!(length, fixture.encoded_length);
    drop(encoder);

    let decoder = reference::DecoderJointModel::load_pretrained(&device);
    let (mut s1, mut s2) = (
        Tensor::<3>::zeros([2, 1, D_PRED], &device),
        Tensor::<3>::zeros([2, 1, D_PRED], &device),
    );
    let (mut t, mut last, mut at_frame, mut ids) = (0usize, BLANK, 0usize, Vec::new());
    while t < length {
        let (logits, _, n1, n2) = decoder.forward(
            encoded.clone().slice_dim(2, t..t + 1),
            Tensor::<2, Int>::from_data(TensorData::from([[last as i32]]), &device),
            Tensor::<1, Int>::from_data(TensorData::from([1i32]), &device),
            s1.clone(),
            s2.clone(),
        );
        let row = logits.reshape([BLANK + 1 + N_DURATIONS]);
        let token = row
            .clone()
            .slice(0..BLANK + 1)
            .argmax(0)
            .into_scalar::<i64>() as usize;
        let mut duration = row.slice(BLANK + 1..).argmax(0).into_scalar::<i64>() as usize;
        if token != BLANK {
            ids.push(token);
            (last, s1, s2) = (token, n1, n2);
            at_frame += 1;
        }
        if duration == 0 && (token == BLANK || at_frame >= 10) {
            duration = 1;
        }
        if duration > 0 {
            at_frame = 0;
        }
        t += duration;
    }
    report_ids(
        "reference graphs behind bunsen's front end",
        &ids,
        &fixture.ids,
    );
    let differing = ids.iter().zip(&fixture.ids).filter(|(a, b)| a != b).count()
        + ids.len().abs_diff(fixture.ids.len());
    assert!(differing <= 2, "{differing} ids differ");
}

/// bunsen's whole pipeline, on the base weights, decodes what the reference
/// decodes, and transcribes the clip about as well.
#[test]
#[serial_test::serial]
fn test_bunsen_base_transcribes_like_the_reference() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = onnx_reference().speech;
    let bundle = load_bundle("nvidia/parakeet-tdt-0.6b-v3", &device);

    let transcript = transcribe(&bundle, &device);
    report_ids(
        "bunsen base vs onnxruntime",
        &transcript.hypothesis.tokens,
        &fixture.ids,
    );
    let reference_text = bundle.vocab.transcript(&fixture.ids).unwrap();
    let against_reference = wer(&transcript.text, &reference_text);
    let against_truth = wer(&transcript.text, &speech_transcript(CLIP));
    eprintln!("bunsen base: {}", transcript.text);
    eprintln!("WER vs onnxruntime {against_reference:.4}, vs transcript {against_truth:.4}");
    // Measured on wgpu: 0.0 (ids identical) and 0.0342.
    assert!(
        against_reference <= 0.0,
        "WER vs the reference {against_reference}"
    );
    assert!(
        against_truth <= 0.04,
        "WER vs the transcript {against_truth}"
    );
}

/// bunsen's Phonon-2 decodes what Phonon-2's own reference path
/// (`transformers`) decodes, and transcribes the clip about as well.
#[test]
#[serial_test::serial]
fn test_bunsen_phonon2_transcribes_like_its_reference() {
    let device = performance_device();
    let _memory = DeviceMemoryGuard::new(&device);
    let fixture = phonon2_reference();
    let bundle = load_bundle(DEFAULT_PARAKEET_MODEL, &device);

    let transcript = transcribe(&bundle, &device);
    report_ids(
        "bunsen phonon-2 vs transformers",
        &transcript.hypothesis.tokens,
        &fixture.ids,
    );
    let against_reference = wer(&transcript.text, &fixture.text);
    let against_truth = wer(&transcript.text, &speech_transcript(CLIP));
    eprintln!("bunsen phonon-2: {}", transcript.text);
    eprintln!("WER vs transformers {against_reference:.4}, vs transcript {against_truth:.4}");
    // Measured on wgpu: 0.0 (ids identical) and 0.0427.
    assert!(
        against_reference <= 0.0,
        "WER vs the reference {against_reference}"
    );
    assert!(
        against_truth <= 0.05,
        "WER vs the transcript {against_truth}"
    );
}
