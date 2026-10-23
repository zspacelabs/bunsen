//! The Whisper kit's pretrained pathway on real files: `OpenAI`'s `base.pt`
//! and both `.tiktoken` vocabularies, as `bunsen-bundled-whisper` lays them
//! out.
//!
//! These ran inside `bunsen` under its `whisper-weights` feature, which made
//! the build fetch ~150 MB. They need no reference graph and load at most one
//! model, so they run on the CPU, under `weights` alone:
//!
//! ```sh
//! cargo test -p whisper-model-validation --features weights
//! ```
//!
//! * `vocabulary`: the rank files, detokenized and filtered, against ids and
//!   lists read off upstream's `whisper.tokenizer`.
//! * `cache`: the bundle's directory is a pretrained cache of the kit's own
//!   maps, and a table of local files serves `openai/base` in place.
//! * `construct`: the kit's hook plans, scans and builds a real checkpoint.
//! * `hf`, under `hf-weights`: `transformers`' `model.safetensors` of the same
//!   model, through the kit's `transformers` loader, is `base.pt`, weight for
//!   weight.

use std::{
    path::PathBuf,
    sync::Arc,
};

use bunsen::{
    data::{
        cache::BunsenDiskCacheOptions,
        pretrained::{
            BUNDLED,
            PretrainedCache,
            PretrainedCacheOptions,
            PretrainedFactory,
            PretrainedGroup,
            PretrainedTable,
        },
    },
    kits::{
        speech::whisper::{
            driver::WhisperSpecialIds,
            pretrained::{
                CHECKPOINT,
                OPENAI,
                VOCABULARY,
                WHISPER_KIT,
                WhisperConstruct,
                WhisperVocabulary,
                default_whisper_factory,
            },
        },
        tokens::TiktokenRanks,
    },
};

mod cache;
mod construct;
#[cfg(feature = "hf-weights")]
mod hf;
mod vocabulary;

/// A cache rooted at the bundle's directory, offline.
pub fn offline_cache() -> PretrainedCache {
    cache_in(bunsen_bundled_whisper::cache_dir().to_path_buf())
}

/// An offline cache rooted at `dir`.
pub fn cache_in(dir: PathBuf) -> PretrainedCache {
    PretrainedCache::new(
        PretrainedCacheOptions::default()
            .with_disk(
                BunsenDiskCacheOptions::default()
                    .with_cache_dir(Some(dir))
                    .without_transfer_observers(),
            )
            .with_offline(true),
    )
    .unwrap()
}

/// The bundled rank file a layout selects.
pub fn bundled_vocabulary_path(ids: &WhisperSpecialIds) -> PathBuf {
    let map = WhisperVocabulary::for_layout(ids).map().to_map();
    offline_cache()
        .resolve(WHISPER_KIT, map.try_get(VOCABULARY).unwrap())
        .unwrap()
        .path
}

/// The bundled rank file a layout selects, parsed.
pub fn bundled_vocabulary(ids: &WhisperSpecialIds) -> TiktokenRanks {
    WhisperVocabulary::for_layout(ids)
        .load(&offline_cache())
        .unwrap()
}

/// A [`BUNDLED`] table of one row, `bundled:openai/base`: the well-known
/// `openai/base` row with its two resources served in place from the
/// bundle's files rather than fetched.
///
/// `bunsen` used to register this under `whisper-weights`; it is how a
/// deployment that ships its files would serve them, so it is built here
/// the way such a caller would.
pub fn bundled_table() -> PretrainedTable {
    let mut base = OPENAI.lookup("base").unwrap().to_pretrained();
    base.resources = base
        .resources
        .with_local_files(
            BUNDLED,
            &[
                (CHECKPOINT, bunsen_bundled_whisper::base_pt()),
                (VOCABULARY, bunsen_bundled_whisper::multilingual_tiktoken()),
            ],
        )
        .unwrap();
    PretrainedTable {
        name: BUNDLED.to_string(),
        description: "the Whisper assets bunsen-bundled-whisper ships, used in place".to_string(),
        groups: vec![PretrainedGroup {
            name: OPENAI.name.to_string(),
            description: OPENAI.description.to_string(),
            license: OPENAI.license.map(str::to_string),
            origin: OPENAI.origin.map(str::to_string),
            items: vec![base],
        }],
    }
}

/// The default Whisper factory with [`bundled_table`] registered.
pub fn factory_with_bundle() -> PretrainedFactory<WhisperConstruct> {
    default_whisper_factory()
        .unwrap()
        .with_provider(Arc::new(bundled_table()))
        .unwrap()
}
