//! NVIDIA's streaming Sortformer v2 and v2.1 checkpoints, as bunsen's
//! `SortformerV2Config` loads them: every parameter filled from the `.nemo`.

use std::path::Path;

use bunsen::{
    burner::module::ModuleInit,
    kits::speech::{
        nemo::{
            NEMO_WEIGHTS_MEMBER,
            NemoArchive,
            read_nemo_weights,
        },
        sortformer::{
            Sortformer,
            SortformerV2Config,
            weights::load_sortformer_v2_nemo_tensors,
        },
    },
    support::testing::cpu_device,
};

/// The `.nemo` at `path` loads into the default v2 geometry, every parameter
/// filled, leaving over only what the loader drops.
fn assert_loads(path: &Path) {
    let archive = NemoArchive::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
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

#[test]
fn test_v2_checkpoint_loads() {
    assert_loads(bunsen_bundled_sortformer::v2_nemo());
}

#[test]
fn test_v2_1_checkpoint_loads() {
    assert_loads(bunsen_bundled_sortformer::v2_1_nemo());
}
