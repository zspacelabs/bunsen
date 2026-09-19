//! Named Whisper models: from a name to a loaded model.
//!
//! The index is bunsen's. A **prefab** is a public *configuration*, the
//! geometry of a model with no weights, in the kit's
//! [`WHISPER_PREFABS`](bunsen::kits::speech::whisper::pretrained::WHISPER_PREFABS).
//! A **pretrained** is a trained set of weights *for* a prefab, with a
//! format, a SHA-256, and the **sources** its bytes can be had from, in a
//! provider's table
//! ([`OPENAI`](bunsen::kits::speech::whisper::pretrained::OPENAI)); a
//! [`WeightsCache`](bunsen::data::pretrained::WeightsCache) brings them
//! local.
//!
//! What is still this crate's is [`loader`]: a model is named
//! `provider/name` &mdash; `openai/tiny.en` &mdash; or by a bare name when
//! only one provider has it, or by a path to a checkpoint, as upstream's
//! `whisper.load_model` accepts. [`loader`] resolves the name, brings the
//! weights into the cache, and loads them through bunsen's
//! [`PytorchWhisperScanner`](bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner),
//! checking what the checkpoint reports against what the prefab promised.
//! That is the name-to-model pathway, and it is the shape a
//! `kits::speech::whisper::pretrained::load_named` could take.

pub mod loader;
