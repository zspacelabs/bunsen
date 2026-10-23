//! The `.nemo` reader on a toy Sortformer v2 that `NeMo` saved itself.
use std::{
    collections::BTreeMap,
    path::PathBuf,
};

use serde::Deserialize;

use super::*;
use crate::{
    blocks::transformers::encoders::{
        SubsampledEncoder,
        conformer::ConformerEncoderConfig,
        residual::{
            ResidualEncoderTransformer,
            ResidualEncoderTransformerConfig,
        },
    },
    burner::module::{
        ModuleInit,
        Rekey,
    },
    data::pretrained::apply_named_tensors,
    errors::BunsenErrorKind,
    support::testing::cpu_device,
};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/nemo")
        .join(name)
}

fn toy() -> NemoArchive {
    NemoArchive::open(&fixture("toy_sortformer_v2.nemo")).unwrap()
}

/// The sections of the toy's config these tests read.
#[derive(Debug, Deserialize)]
struct ToyConfig {
    preprocessor: NemoPreprocessorConfig,
    encoder: NemoConformerEncoderConfig,
    transformer_encoder: NemoTransformerEncoderConfig,
    max_num_of_spks: usize,
}

/// One manifest entry.
#[derive(Debug, Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    sum: f64,
}

fn manifest() -> BTreeMap<String, Entry> {
    #[derive(Deserialize)]
    struct Manifest {
        tensors: BTreeMap<String, Entry>,
    }
    let text = std::fs::read_to_string(fixture("toy_sortformer_v2.json")).unwrap();
    serde_json::from_str::<Manifest>(&text).unwrap().tensors
}

/// The toy's weights, extracted and read.
fn weights() -> Vec<burn::store::burn_pack::Tensor> {
    let dir = tempfile::tempdir().unwrap();
    let ckpt = dir.path().join(NEMO_WEIGHTS_MEMBER);
    toy().extract_weights(&ckpt).unwrap();
    assert!(!ckpt.with_extension("partial").exists());
    read_nemo_weights(&ckpt).unwrap()
}

/// The config parses into typed sections, with `NeMo`'s values.
#[test]
fn test_config_sections() {
    let config: ToyConfig = toy().config().unwrap();
    assert_eq!(config.max_num_of_spks, 3);
    assert_eq!(config.preprocessor.features, 24);
    assert_eq!(config.preprocessor.normalize, "NA");
    assert_eq!(config.encoder.n_layers, 2);
    assert_eq!(config.encoder.subsampling_conv_channels, 4);
    assert_eq!(config.encoder.att_context_size, [-1, -1]);
    assert_eq!(config.transformer_encoder.hidden_act, "relu");
    assert!(!config.transformer_encoder.pre_ln);
    assert!(config.transformer_encoder.pre_ln_final_layer_norm);
    assert!(!config.transformer_encoder.final_norm());
}

/// Each section gives the bunsen config it describes.
#[test]
fn test_bunsen_configs() {
    use crate::{
        blocks::transformers::encoders::{
            conformer::ConformerEncoderMeta,
            residual::{
                NormPlacement,
                ResidualEncoderAttentionBlockMeta,
            },
        },
        kits::speech::nemo::FeatureNormalization,
    };
    let config: ToyConfig = toy().config().unwrap();

    let front_end = config.preprocessor.front_end().unwrap();
    assert_eq!(
        (front_end.win_length, front_end.hop, front_end.n_mels),
        (400, 160, 24)
    );
    assert_eq!(front_end.normalize, FeatureNormalization::None);
    assert_eq!(front_end.preemphasis, 0.97);
    assert!(!front_end.bf16_tables);

    let conformer = config.encoder.conformer().unwrap();
    assert_eq!(conformer.d_model(), 16);
    assert_eq!(conformer.n_layers(), 2);
    assert_eq!(conformer.subsampling_factor(), 8);
    assert!(conformer.scale_input);
    assert_eq!(conformer.block.d_ff, 32);

    let block = config.transformer_encoder.block().unwrap();
    assert_eq!((block.d_model, block.n_heads()), (8, 2));
    assert_eq!(block.expansion_factor, 2);
    assert_eq!(block.norm_placement(), NormPlacement::Post);
    assert!(block.key_bias);
}

/// A setting bunsen does not compute is refused, naming the field.
#[test]
fn test_unsupported_settings() {
    let yaml = toy().config_yaml().to_string();
    for (from, to, field) in [
        ("window: hann", "window: hamming", "preprocessor.window"),
        (
            "normalize: NA",
            "normalize: all_features",
            "preprocessor.normalize",
        ),
        (
            "subsampling: dw_striding",
            "subsampling: striding",
            "encoder.subsampling",
        ),
        (
            "self_attention_model: rel_pos",
            "self_attention_model: abs_pos",
            "encoder.self_attention_model",
        ),
        (
            "hidden_act: relu",
            "hidden_act: swish",
            "transformer_encoder.hidden_act",
        ),
    ] {
        assert!(yaml.contains(from), "{from}");
        let config: ToyConfig = parse_nemo_yaml(&yaml.replacen(from, to, 1)).unwrap();
        let err = match field.split('.').next().unwrap() {
            "preprocessor" => config.preprocessor.front_end().err(),
            "encoder" => config.encoder.conformer().err(),
            _ => config.transformer_encoder.block().err(),
        }
        .unwrap_or_else(|| panic!("{field} accepted"));
        assert_eq!(err.kind(), BunsenErrorKind::Unsupported);
        assert!(err.to_string().contains(field), "{err}");
    }
    let err = parse_nemo_yaml::<ToyConfig>("encoder: [").unwrap_err();
    assert_eq!(err.kind(), BunsenErrorKind::InvalidResource);
}

/// The state dict reads through the burn-store's reader: every entry, with
/// the dtype, shape and values `NeMo` saved.
#[test]
fn test_weights_match_the_manifest() {
    let want = manifest();
    let got = weights();
    let names: Vec<&str> = got.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, want.keys().map(String::as_str).collect::<Vec<_>>());
    for tensor in got {
        let entry = &want[&tensor.name];
        assert_eq!(tensor.shape.to_vec(), entry.shape, "{}", tensor.name);
        let data = burn::store::bridge::to_data(&tensor).unwrap();
        let sum: f64 = match entry.dtype.as_str() {
            "float32" => data
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|&v| v as f64)
                .sum(),
            "int64" => data
                .try_to_vec::<i64>()
                .unwrap()
                .iter()
                .map(|&v| v as f64)
                .sum(),
            other => panic!("{}: {other}", tensor.name),
        };
        assert!(
            (sum - entry.sum).abs() <= 1e-4 * (1.0 + entry.sum.abs()),
            "{}: {sum} != {}",
            tensor.name,
            entry.sum
        );
    }
}

/// `with_weights` hands `f` the state dict `extract_weights` and
/// `read_nemo_weights` give, every value readable inside `f`, and returns
/// what `f` does, its error too.
#[test]
fn test_with_weights() {
    use burn::store::bridge::to_data;
    let want: Vec<(String, burn::tensor::TensorData)> = weights()
        .iter()
        .map(|t| (t.name.clone(), to_data(t).unwrap()))
        .collect();
    let got = toy()
        .with_weights(|tensors| {
            Ok(tensors
                .iter()
                .map(|t| (t.name.clone(), to_data(t).unwrap()))
                .collect::<Vec<_>>())
        })
        .unwrap();
    assert_eq!(got, want);

    let err = toy()
        .with_weights(|_| Err::<(), _>(crate::errors::BunsenError::illegal("no")))
        .unwrap_err();
    assert_eq!(err.kind(), BunsenErrorKind::Illegal);
}

/// The encoder's entries load into the bunsen `SubsampledEncoder` built from
/// the config, filling every parameter; only the batch-norm counters are
/// left over.
#[test]
fn test_conformer_loads() {
    let config: ToyConfig = toy().config().unwrap();
    let conformer = config.encoder.conformer().unwrap();
    let n_stages = conformer.subsampling.n_stages;
    let mut encoder: SubsampledEncoder = conformer.init(&cpu_device());
    let tensors = weights()
        .into_iter()
        .filter(|t| t.name.starts_with("encoder."));
    let rekey =
        Rekey::drop_rest().mount("encoder.", "", ConformerEncoderConfig::nemo_rekey(n_stages));
    let applied = apply_named_tensors(&mut encoder, tensors, rekey.as_fn()).unwrap();
    assert!(!applied.applied.is_empty());
    assert!(
        applied
            .unused
            .iter()
            .all(|n| n.ends_with(".num_batches_tracked")),
        "{:?}",
        applied.unused
    );
}

/// The transformer's entries load into a stack of residual blocks built
/// from the config, filling every parameter.
#[test]
fn test_transformer_loads() {
    let config: ToyConfig = toy().config().unwrap();
    let block = config.transformer_encoder.block().unwrap();
    let mut stack: ResidualEncoderTransformer =
        ResidualEncoderTransformerConfig::new(block, config.transformer_encoder.num_layers)
            .init(&cpu_device());
    let tensors = weights()
        .into_iter()
        .filter(|t| t.name.starts_with("transformer_encoder."));
    let rekey = Rekey::drop_rest().mount(
        "transformer_encoder.",
        "",
        ResidualEncoderTransformer::nemo_rekey(),
    );
    let applied = apply_named_tensors(&mut stack, tensors, rekey.as_fn()).unwrap();
    assert!(applied.unused.is_empty(), "{:?}", applied.unused);
}

/// A gzipped archive, as older `NeMo`s wrote, reads the same.
#[test]
fn test_gzipped_archive() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("toy.nemo");
    let mut gz = flate2::write::GzEncoder::new(
        std::fs::File::create(&path).unwrap(),
        flate2::Compression::fast(),
    );
    gz.write_all(&std::fs::read(fixture("toy_sortformer_v2.nemo")).unwrap())
        .unwrap();
    gz.finish().unwrap();

    let archive = NemoArchive::open(&path).unwrap();
    assert_eq!(archive.config_yaml(), toy().config_yaml());
    let ckpt = dir.path().join("weights.ckpt");
    archive.extract_weights(&ckpt).unwrap();
    assert_eq!(read_nemo_weights(&ckpt).unwrap().len(), manifest().len());
    let head = std::fs::read(&path).unwrap();
    assert_eq!(NemoArchive::config_from_prefix(&head), None);
}

/// The config reads from the archive's first bytes once they hold it, and
/// not before. `NeMo`'s own `save_to` writes a PAX header before each member,
/// so the toy's config runs from byte 3072 to 4926.
#[test]
fn test_config_from_prefix() {
    let bytes = std::fs::read(fixture("toy_sortformer_v2.nemo")).unwrap();
    let yaml = toy().config_yaml().to_string();
    let end = 3072 + yaml.len();
    assert_eq!(NemoArchive::config_from_prefix(&bytes[..end]), Some(yaml));
    assert_eq!(NemoArchive::config_from_prefix(&bytes[..end - 1]), None);
    assert_eq!(NemoArchive::config_from_prefix(&bytes[..1200]), None);
    assert_eq!(NemoArchive::config_from_prefix(&[]), None);
}

/// An archive of `members`, in order, as `NeMo` names them.
fn tar_of(members: &[(&str, &[u8])]) -> Vec<u8> {
    let mut tar = tar::Builder::new(Vec::new());
    for (name, bytes) in members {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, format!("./{name}"), *bytes)
            .unwrap();
    }
    tar.into_inner().unwrap()
}

/// Members before the config (a tokenizer's files) are passed over while
/// the prefix holds them; weights before it end the search.
#[test]
fn test_config_from_prefix_past_other_members() {
    let yaml = "target: x\n";
    let vocab = vec![b'v'; 3000];
    let bytes = tar_of(&[
        ("vocab.txt", &vocab),
        (NEMO_CONFIG_MEMBER, yaml.as_bytes()),
        (NEMO_WEIGHTS_MEMBER, &[0u8; 1024]),
    ]);
    // Each member is a 512-byte header and its data, padded to 512.
    let end = 512 + 3072 + 512 + yaml.len();
    assert_eq!(
        NemoArchive::config_from_prefix(&bytes[..end]).as_deref(),
        Some(yaml)
    );
    assert_eq!(NemoArchive::config_from_prefix(&bytes[..2048]), None);

    let weights_first = tar_of(&[
        (NEMO_WEIGHTS_MEMBER, &[0u8; 1024]),
        (NEMO_CONFIG_MEMBER, yaml.as_bytes()),
    ]);
    assert_eq!(NemoArchive::config_from_prefix(&weights_first), None);
}

/// A file that is not a `.nemo`, or lacks a member, is refused.
#[test]
fn test_not_an_archive() {
    let dir = tempfile::tempdir().unwrap();
    let junk = dir.path().join("junk.nemo");
    std::fs::write(&junk, vec![7u8; 2048]).unwrap();
    assert_eq!(
        NemoArchive::open(&junk).unwrap_err().kind(),
        BunsenErrorKind::InvalidResource
    );

    let partial = dir.path().join("config_only.nemo");
    let mut builder = tar::Builder::new(std::fs::File::create(&partial).unwrap());
    let yaml = toy().config_yaml().as_bytes().to_vec();
    let mut header = tar::Header::new_gnu();
    header.set_size(yaml.len() as u64);
    header.set_mode(0o644);
    header.set_cksum();
    builder
        .append_data(&mut header, NEMO_CONFIG_MEMBER, yaml.as_slice())
        .unwrap();
    builder.finish().unwrap();
    drop(builder);
    let err = NemoArchive::open(&partial).unwrap_err();
    assert!(err.to_string().contains(NEMO_WEIGHTS_MEMBER), "{err}");

    let missing = dir.path().join("missing.nemo");
    assert_eq!(
        NemoArchive::open(&missing).unwrap_err().kind(),
        BunsenErrorKind::Lookup
    );
}
