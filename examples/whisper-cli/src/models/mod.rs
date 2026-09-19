//! Named Whisper models: prefabs, pretrained weights, and the local cache.
//!
//! Three things are kept apart here, because they vary independently:
//!
//! * A **prefab** is a public *configuration*: the geometry of a model with no
//!   weights &mdash; `n_mels`, the vocabulary size, `d_model`, the layer
//!   counts. bunsen's
//!   [`StaticPreFabMap`](bunsen::data::pretrained::StaticPreFabMap) carries
//!   them, in the kit's
//!   [`WHISPER_PREFABS`](bunsen::kits::speech::whisper::pretrained::WHISPER_PREFABS).
//! * A **pretrained** is a trained set of weights *for* a prefab. One prefab
//!   commonly has several (`large-v1` and `large-v2` share `large`); one
//!   pretrained may have several names (`large` is `large-v3`) and several
//!   places to get it from, all the same bytes; and a quantization family is
//!   several pretrained over one prefab, told apart by
//!   [`WeightsFormat`](pretrained::WeightsFormat). See [`pretrained`].
//! * A **source** is one of those places: a URL, upstream's own
//!   `~/.cache/whisper`, or the file `bunsen-bundled-whisper` fetched at build
//!   time. Every source is pinned by the pretrained's SHA-256, which is what
//!   makes them interchangeable. See [`weights_cache`].
//!
//! A model is named `provider/name` &mdash; `openai/tiny.en` &mdash; or by a
//! bare name when only one provider has it, or by a path to a checkpoint,
//! as upstream's `whisper.load_model` accepts. [`loader`] resolves the name,
//! brings the weights into the cache, and loads them through bunsen's
//! [`PytorchWhisperScanner`](bunsen::kits::speech::whisper::pretrained::PytorchWhisperScanner),
//! checking what the checkpoint reports against what the prefab promised.
//!
//! Nothing here is bunsen API: it is the shape a
//! `kits::speech::whisper::pretrained` index could take, worked out where it
//! can be run against real files.

pub mod loader;
pub mod pretrained;
pub mod weights_cache;
