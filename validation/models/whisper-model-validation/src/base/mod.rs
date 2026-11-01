//! `openai/base`, the one model this crate checks: every check of it.

use super::*;

/// Staged cross-checks on synthetic input, against the generated reference.
#[cfg(all(feature = "checkpoints", feature = "reference", feature = "gpu-tests"))]
pub mod staged;

/// End-to-end validation over the committed speech fixtures.
pub mod audio;

/// The kit's pretrained pathway on the bundle's files: `OpenAI`'s `base.pt`,
/// both `.tiktoken` vocabularies and `transformers`' `model.safetensors`, as
/// `bind-whisper-onnx` lays them out under `reference`. Bundling is for
/// verification, so these run where the ONNX conversion is cross-checked.
/// They load at most two models, on the CPU.
///
/// * `vocabulary`: the rank files, detokenized and filtered, against ids and
///   lists read off upstream's `whisper.tokenizer`.
/// * `cache`: the bundle's directory is a pretrained cache of the kit's own
///   maps, and a table of local files serves `openai/base` in place.
/// * `construct`: the kit's hook plans, scans and builds a real checkpoint.
/// * `hf`: `transformers`' `model.safetensors` of the same model, through the
///   kit's `transformers` loader, is `base.pt`, weight for weight.
#[cfg(feature = "reference")]
mod bundled;
