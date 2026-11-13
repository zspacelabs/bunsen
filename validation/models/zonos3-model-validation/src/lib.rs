//! # ZONOS3-Mini cross-check against Zyphra's reference.
//!
//! Validates [`bunsen`]'s ZONOS3-Mini (`kits::speech::tts::zonos3`) against
//! upstream's reference implementation over the pretrained checkpoint
//! (`Zyphra/ZONOS3-Mini-hf`): its language model's logits, its prompts, its
//! tokenizer and its codec's audio.
//!
//! ## This crate hosts nothing
//!
//! The checkpoint comes through bunsen's own pinned row, fetched at run time
//! into the weights cache by the tests that read it (feature
//! **`checkpoints`**), which is off by default. No ZONOS3 file is committed:
//! the fixtures are rows, ids, logit summaries and audio summaries
//! (`validation/tools/gen_zonos3_fixtures.py` writes them; [`fixtures`]
//! reads them):
//!
//! - `testdata/reference.safetensors`: upstream's float32 reference language
//!   model over upstream's own prompt requests, its prefill logits and
//!   teacher-forced decode steps;
//! - `testdata/reference_bf16.safetensors`: the same in bfloat16;
//! - `testdata/codec.safetensors`: the `-hf` codec's audio from fixed codes;
//! - `testdata/tokenizer.safetensors`: the `-hf` tokenizer's ids for the UDHR
//!   texts.
//!
//! `cargo validate` (`validation/tools/validate`) runs this crate under a
//! profile (`validation/models.yml`).

pub mod fixtures;
