//! Checks on upstream's own `silero_vad_16k.safetensors`, which need no
//! reference graph: the probabilities the kits replay, the stream, and the
//! factory's pathway from a bundled cache.
//!
//! These ran inside `bunsen` under its `silero-weights` feature, which made
//! every workspace build fetch the checkpoint. The kits now replay a
//! recording, [`RecordedVoiceActivity::silero_jfk_moon_4s`]; this module holds
//! that recording to the model it was taken from.
//!
//! Run on the CPU, as the recording was:
//!
//! ```sh
//! cargo test -p silero-model-validation --features weights
//! ```

use bunsen::{
    data::{
        cache::BunsenDiskCacheOptions,
        pretrained::{
            PretrainedCache,
            PretrainedCacheOptions,
            Provenance,
        },
    },
    kits::speech::{
        silero_vad::{
            SileroVad,
            SileroVadContextConfig,
            SileroVadMeta,
            pretrained::{
                CHECKPOINT,
                default_silero_factory,
            },
        },
        testing::RecordedVoiceActivity,
    },
    support::{
        audio::load_audio_mono_sr,
        testing::cpu_device,
    },
};
use burn::{
    Tensor,
    prelude::TensorData,
    tensor::Distribution,
};

/// Where the kits' recording is committed.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../public/bunsen/testdata/silero/jfk_moon_4s.json"
);

/// Upstream's 16 kHz model, from the file `bunsen-bundled-silero` fetched.
fn upstream(device: &burn::tensor::Device) -> SileroVad {
    SileroVad::load_16khz_from_safetensors_file(
        bunsen_bundled_silero::silero_vad_16k_safetensors(),
        device,
    )
    .unwrap()
}

/// A cache rooted at the bundle's directory, offline.
fn bundled_cache() -> PretrainedCache {
    PretrainedCache::new(
        PretrainedCacheOptions::default()
            .with_disk(
                BunsenDiskCacheOptions::default()
                    .with_cache_dir(Some(bunsen_bundled_silero::cache_dir().to_path_buf()))
                    .without_transfer_observers(),
            )
            .with_offline(true),
    )
    .unwrap()
}

/// What the kits replay, measured: upstream's model over
/// `bunsen_test_audio::JFK_MOON_4S`, zero-padded to whole chunks, in one
/// call through the streaming context, on the CPU.
fn record() -> RecordedVoiceActivity {
    let device = cpu_device();
    let vad = upstream(&device);
    let sample_rate: usize = 16_000;
    let mut wav = load_audio_mono_sr(bunsen_test_audio::JFK_MOON_4S.path(), sample_rate).unwrap();
    let samples = wav.len();
    let chunk = vad.chunk_size();
    wav.resize(wav.len().div_ceil(chunk) * chunk, 0.0);
    let steps = wav.len() / chunk;
    let chunks: Tensor<3> = Tensor::from_data(TensorData::new(wav, [steps, 1, chunk]), &device);
    let context = SileroVadContextConfig::new(sample_rate).init(&vad, &device);
    let (probs, _) = vad.context_forward_sequence(chunks, context);
    RecordedVoiceActivity {
        sample_rate,
        chunk_size: chunk,
        samples,
        probabilities: probs.to_data().try_into_vec_as::<f32>().unwrap(),
    }
}

/// The kits' recording is still what upstream's model says: chunk for
/// chunk, to well inside the margin a speech decision would notice.
#[test]
#[serial_test::serial]
fn test_the_kits_recording_is_upstreams() {
    let measured = record();
    let committed = RecordedVoiceActivity::silero_jfk_moon_4s();
    assert_eq!(measured.sample_rate, committed.sample_rate);
    assert_eq!(measured.chunk_size, committed.chunk_size);
    assert_eq!(measured.samples, committed.samples);
    assert_eq!(measured.probabilities.len(), committed.probabilities.len());
    for (i, (a, b)) in measured
        .probabilities
        .iter()
        .zip(&committed.probabilities)
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-4,
            "chunk {i}: measured {a}, committed {b}; rerun `write_the_kits_recording`"
        );
    }
}

/// Rewrites the kits' recording from upstream's model. Not a check: run it
/// by name when the model or the clip changes, and review the diff.
///
/// ```sh
/// cargo test -p silero-model-validation --features weights \
///   -- --ignored write_the_kits_recording
/// ```
#[test]
#[ignore = "writes the fixture; run by name"]
fn write_the_kits_recording() {
    let json = serde_json::to_string_pretty(&record()).unwrap();
    std::fs::write(FIXTURE, json + "\n").unwrap();
}

/// Streamed in two calls, Silero's probabilities are those of one call
/// over the same chunks.
#[test]
#[serial_test::serial]
fn test_silero_streams_as_one_call() {
    use bunsen::kits::speech::asr_driver::VoiceActivityModel;

    let model = upstream(&cpu_device());
    let chunk = VoiceActivityModel::chunk_size(&model);
    let mut samples = load_audio_mono_sr(bunsen_test_audio::JFK_MOON_4S.path(), 16_000).unwrap();
    samples.truncate(samples.len() / chunk * chunk);

    let whole = model.new_stream().probabilities(samples.clone());
    assert_eq!(whole.len(), samples.len() / chunk);

    let mut stream = model.new_stream();
    let cut = 37 * chunk;
    let mut parts = stream.probabilities(samples[..cut].to_vec());
    parts.extend(stream.probabilities(samples[cut..].to_vec()));

    assert_eq!(parts.len(), whole.len());
    for (a, b) in parts.iter().zip(&whole) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
    assert!(whole.iter().any(|&p| p > 0.5));
}

/// `silero/vad-16k` loads through the factory from the cache
/// `bunsen-bundled-silero` filled, offline: the file is found where the
/// build laid it out, and the model scores as the file read directly does.
#[test]
#[serial_test::serial]
fn test_the_bundled_cache_resolves_upstream_offline() {
    let device = cpu_device();
    let loaded = default_silero_factory()
        .unwrap()
        .load("silero/vad-16k", &bundled_cache(), &device)
        .unwrap();
    let part = loaded.resources.get(CHECKPOINT).unwrap();
    assert_eq!(part.provenance, Provenance::Cached);
    assert_eq!(
        part.path,
        bunsen_bundled_silero::silero_vad_16k_safetensors()
    );

    let a = loaded.handle.expect_branch(16000);
    let b = upstream(&device);
    let batch = 2;
    let input = Tensor::<2>::random([batch, 64 + a.chunk_size()], Distribution::Default, &device);
    let (pa, _) = a.forward(input.clone(), a.init_state(batch, &device));
    let (pb, _) = b.forward(input, b.init_state(batch, &device));
    let pa: Vec<f32> = pa.into_data().try_into_vec_as().unwrap();
    let pb: Vec<f32> = pb.into_data().try_into_vec_as().unwrap();
    assert_eq!(pa, pb);
}
