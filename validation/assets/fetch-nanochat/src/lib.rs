//! nanochat validation assets.
//!
//! **This crate hosts nothing.** Each community nanochat checkpoint is about
//! 2 GB, too large to commit, so this crate *fetches* them, pins each file to
//! a SHA-256, and keeps them in `OUT_DIR`. Only `nanochat-model-validation`
//! depends on it.
//!
//! The files are laid out in `OUT_DIR` as a **pretrained cache**,
//! `pretrained/nanochat/<namespace>/<sha256>/<file>`, `<file>` the file's
//! path in its repository: how bunsen's `data::pretrained::PretrainedCache`
//! roots a resource. A cache pointed at `cache_dir()`, offline, hits the rows
//! without knowing they were bundled.
//!
//! | Feature | Repository | Files |
//! |---|---|---|
//! | `base_d20` | `nanochat-students/base-d20` | `model_021400.pt`, `meta_021400.json`, `tokenizer.pkl`, `token_bytes.pt` |
//! | `mid_d20` | `nanochat-students/mid-d20` | `model_000765.pt`, `meta_000765.json`, `tokenizer.pkl`, `token_bytes.pt` |
//! | `rl_d20` | `nanochat-students/rl-d20` | `model_000650.pt`, `meta_000650.json`, `tokenizer.pkl`, `token_bytes.pt` |
//! | `d18_base` | `nitishpandey04/nanochat-d18-base` | `base_checkpoints/d18/{model,meta}_003712.*`, `tokenizer/{tokenizer.pkl,token_bytes.pt}` |
//! | `d18_sft` | `nitishpandey04/nanochat-d18-sft` | `chatsft_checkpoints/d18/{model,meta}_000478.*`, `tokenizer/{tokenizer.pkl,token_bytes.pt}` |
//!
//! These are upstream nanochat's own checkpoint files, as its training
//! scripts write them: no conversion.
//!
//! ## Using local copies
//!
//! Each file has a variable naming a local file to use instead, read only
//! when the feature that fetches it is on: `NANOCHAT_`, the row, and the
//! file's name, in capitals with `_` for `.` and `-`
//! (`NANOCHAT_RL_D20_MODEL_000650_PT`, `NANOCHAT_D18_SFT_TOKENIZER_PKL`). A
//! file that is the pinned asset is put where the download would have gone;
//! any other file is used where it is, with a warning.
//!
//! ## Crate Features
#![doc = document_features::document_features!()]

/// `OUT_DIR` laid out as a pretrained cache: the root a
/// `bunsen::data::pretrained::PretrainedCache` is pointed at to read the
/// bundled rows. A cache that fetches other models into it will change it,
/// so a test wants it offline.
pub fn cache_dir() -> &'static std::path::Path {
    std::path::Path::new(env!("NANOCHAT_CACHE_DIR"))
}
