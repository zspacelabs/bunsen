//! The toy speaker encoder against the `-hf` port's.
//!
//! `testdata/zonos3/speaker_toy.safetensors` holds a three-stage
//! `Zonos3MiniSpeakerEncoder` (the `-hf` port's, op for op with upstream's
//! MECT-B2 and ASTP) under the `-hf` file's names, and its answers over a
//! fixed `[1, 16, 23]` feature map, computed in float64
//! (`tools/gen_zonos3_speaker_toy_fixtures.py`): the head, each block, the
//! stage mix, the pooling and the embedding. The geometry comes from
//! `speaker_toy.config.json`'s `speaker_encoder` section, and the weights
//! load through [`load_zonos3_speaker_tensors`].

use std::path::PathBuf;

use crate::{
    burner::module::ModuleInit,
    kits::speech::tts::zonos3::{
        blocks::{
            Zonos3SpeakerEncoder,
            Zonos3SpeakerEncoderConfig,
            Zonos3SpeakerEncoderMeta,
        },
        interop::zonos::{
            Zonos3SpeakerEncoderMirror,
            load_zonos3_speaker_tensors,
        },
    },
    support::testing::{
        ToyFixture,
        assert_io_close,
        assert_scalar_close,
        performance_device,
    },
};

fn mirror() -> Zonos3SpeakerEncoderMirror {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/zonos3");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("speaker_toy.config.json")).expect("read the toy config"),
    )
    .expect("parse the toy config");
    serde_json::from_value(json["speaker_encoder"].clone()).expect("the toy's speaker section")
}

fn load() -> (Zonos3SpeakerEncoderConfig, Zonos3SpeakerEncoder, ToyFixture) {
    let device = performance_device();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/zonos3");
    let config = mirror().speaker_config().expect("the toy's config");
    let mut fixture = ToyFixture::load(dir.join("speaker_toy.safetensors"), &device);
    let mut model: Zonos3SpeakerEncoder = config.init(&device);
    load_zonos3_speaker_tensors(&mut model, &config, fixture.take_weights(), None)
        .expect("load the toy weights");
    (config, model, fixture)
}

/// The toy's config and encoder answer the meta alike.
#[test]
fn test_toy_meta() {
    let (config, model, _) = load();
    for meta in [&config as &dyn Zonos3SpeakerEncoderMeta, &model] {
        assert_eq!(
            (
                meta.mel_bins(),
                meta.width(),
                meta.embed_dim(),
                meta.n_stages()
            ),
            (16, 64, 5, 3)
        );
    }
    let blocks: Vec<usize> = model.stages.iter().map(Vec::len).collect();
    assert_eq!(blocks, [1, 2, 2]);
    assert!(model.stages[0][0].reduce.is_some(), "expansion 2");
    assert!(model.stages[2][0].reduce.is_none(), "expansion 1");
    assert!(
        model.stages[1][0].cnn[0].shortcut.is_some(),
        "the first residual block strides"
    );
    assert!(model.stages[1][0].cnn[1].shortcut.is_none());
}

/// Every stage, from the features to the embedding.
#[test]
fn test_toy_against_reference() {
    let (_, model, fixture) = load();
    let trace = model.forward_trace(fixture.io::<3>("speaker.features"));
    assert_io_close("head", trace.head, fixture.io::<4>("speaker.head"));
    for (s, stage) in trace.blocks.into_iter().enumerate() {
        for (j, block) in stage.into_iter().enumerate() {
            let name = format!("speaker.block_{s}_{j}");
            assert_io_close(&name, block, fixture.io::<4>(&name));
        }
    }
    assert_io_close("mix", trace.mix, fixture.io::<3>("speaker.mix"));
    assert_io_close("pooled", trace.pooled, fixture.io::<2>("speaker.pooled"));
    assert_io_close(
        "embedding",
        trace.embedding,
        fixture.io::<2>("speaker.embedding"),
    );
}

/// The release's section reads: the geometry upstream's `ownmect_s2`
/// documents (4 stages of 2, 4, 5 and 3 blocks; a width of 2560; 192-d).
#[test]
fn test_release_section() {
    let mirror: Zonos3SpeakerEncoderMirror = serde_json::from_str(
        r#"{"bottleneck_dim": 128, "embed_dim": 192, "feat_dim": 80,
            "frame_length_ms": 25.0, "frame_shift_ms": 10.0, "hidden_dim": 64,
            "in_channels": 32, "max_seconds": 60.0, "num_experts": 4, "num_mel_bins": 80,
            "sample_rate": 16000,
            "stages": [[[3,32,1,1,4],[3,32,1,1,4]],
                       [[3,64,2,1,2],[3,64,1,2,2],[3,64,1,2,2],[3,64,1,2,2]],
                       [[3,128,2,1,2],[3,128,1,2,2],[3,128,1,2,2],[3,128,1,2,2],[3,128,1,2,2]],
                       [[3,256,2,1,1],[2,256,1,2,1],[2,256,1,2,1]]],
            "waveform_scale": 32768.0}"#,
    )
    .unwrap();
    let config = mirror.speaker_config().unwrap();
    assert_eq!(config.width(), 2560);
    assert_eq!(config.front_end.max_samples(), 960_000);
    let blocks: Vec<usize> = config.stages.iter().map(Vec::len).collect();
    assert_eq!(blocks, [2, 4, 5, 3]);

    let mut bad = mirror.clone();
    bad.num_mel_bins = 64;
    assert!(
        bad.speaker_config().is_err(),
        "front end and network disagree"
    );
    let mut bad = mirror.clone();
    bad.stages[1][0][1] = 48;
    assert!(
        bad.speaker_config().is_err(),
        "a block that is not c_in * stride_f"
    );
    let mut bad = mirror;
    bad.hidden_dim = 66;
    assert!(bad.speaker_config().is_err(), "heads that do not divide");
}

/// A clip too short for one filterbank frame is refused, naming its length;
/// a clip is resampled, then scaled. (`torchaudio`'s 48 kHz to 16 kHz kernel
/// passes a constant at a gain of 1.00047: 0.5 becomes `16391.64` at
/// int16's scale, measured with `torchaudio.transforms.Resample` 2.9.1.)
#[test]
fn test_short_clip() {
    let (_, model, _) = load();
    let err = model.embed(&[0.0; 100], 16000).unwrap_err();
    assert!(format!("{err}").contains("100 samples"), "{err}");
    let prepared = model.prepare(&[0.5; 48_000], 48_000);
    assert_eq!(prepared.len(), 16_000, "resampled to 16 kHz");
    assert_scalar_close(prepared[8000], 16391.64, 0.05);
}
