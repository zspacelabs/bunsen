//! The Whisper kit's pretrained pathway on the bundle's files.

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
mod hf;
mod vocabulary;

/// A cache rooted at the bundle's directory, offline.
pub fn offline_cache() -> PretrainedCache {
    cache_in(bind_whisper_onnx::cache_dir().to_path_buf())
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
/// It is how a deployment that ships its files would serve them, so it is
/// built here the way such a caller would.
pub fn bundled_table() -> PretrainedTable {
    let mut base = OPENAI.lookup("base").unwrap().to_pretrained();
    base.resources = base
        .resources
        .with_local_files(
            BUNDLED,
            &[
                (CHECKPOINT, bind_whisper_onnx::base_pt()),
                (VOCABULARY, bind_whisper_onnx::multilingual_tiktoken()),
            ],
        )
        .unwrap();
    PretrainedTable {
        name: BUNDLED.to_string(),
        description: "the Whisper assets bind-whisper-onnx ships, used in place".to_string(),
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
