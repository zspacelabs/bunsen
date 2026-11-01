//! NVIDIA's streaming Sortformer v2 and v2.1 checkpoints, as bunsen reads
//! them: the `.nemo`'s config is `SortformerV2Config`'s defaults, and every
//! parameter is filled from its weights.

use std::path::Path;

use bunsen::{
    burner::module::{
        ModuleInit,
        ToStructureConfig,
    },
    kits::speech::{
        nemo::{
            FeatureNormalization,
            NEMO_WEIGHTS_MEMBER,
            NemoArchive,
            read_nemo_weights,
        },
        sortformer::{
            NemoSortformerConfig,
            Sortformer,
            SortformerV2Config,
            pretrained::CHECKPOINT,
            weights::load_sortformer_v2_nemo_tensors,
        },
    },
    support::testing::cpu_device,
};

use super::*;

/// The `.nemo` at `path` describes the default v2 geometry, behind `NeMo`'s
/// front end without normalization, and loads into it, every parameter
/// filled, leaving over only what the loader drops.
fn assert_loads(path: &Path) {
    let archive = NemoArchive::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let config: NemoSortformerConfig = archive.config().unwrap();
    let v2 = SortformerV2Config::diar_streaming_sortformer_4spk_v2().to_structure();
    assert_eq!(
        serde_json::to_value(config.structure().unwrap()).unwrap(),
        serde_json::to_value(v2).unwrap(),
        "{}",
        path.display()
    );
    let front_end = config.front_end().unwrap();
    assert_eq!(
        (front_end.n_mels, front_end.normalize, front_end.bf16_tables),
        (128, FeatureNormalization::None, false)
    );
    let dir = tempfile::tempdir().unwrap();
    let ckpt = dir.path().join(NEMO_WEIGHTS_MEMBER);
    archive.extract_weights(&ckpt).unwrap();
    let tensors = read_nemo_weights(&ckpt).unwrap();

    let mut model: Sortformer =
        SortformerV2Config::diar_streaming_sortformer_4spk_v2().init(&cpu_device());
    let applied = load_sortformer_v2_nemo_tensors(&mut model, tensors)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    for name in &applied.unused {
        assert!(
            name.ends_with(".num_batches_tracked")
                || name.starts_with("preprocessor.featurizer.")
                || name.starts_with("sortformer_modules.hidden_to_spks."),
            "{}: unused {name}",
            path.display()
        );
    }
    // 17 Conformer layers of 39 tensors, the subsampling's 12, 18
    // transformer layers of 16, the projection's 2 and the head's 4.
    assert_eq!(applied.applied.len(), 17 * 39 + 12 + 18 * 16 + 2 + 4);
}

mod sortformer_v2 {
    use super::*;

    #[test]
    fn loads() {
        assert_loads(&row_file("sortformer-v2", CHECKPOINT));
    }
}

mod sortformer_v2_1 {
    use super::*;

    #[test]
    fn loads() {
        assert_loads(&row_file("sortformer-v2.1", CHECKPOINT));
    }
}
