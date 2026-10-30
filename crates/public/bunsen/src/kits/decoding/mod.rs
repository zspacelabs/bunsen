//! # Decoding: choosing tokens from logits
//!
//! Not a model: the search machinery the decoding kits share, from the
//! logits of a step to the ids a decode returns. A decode loop (the model's,
//! which owns its tensors and its cache) calls these every step:
//!
//! - [`LogitFilter`]: rewrites the last position's logits before the search
//!   sees them; [`SuppressTokens`] is the generic one, a kit adds its own
//!   (Whisper's timestamp grammar,
//!   [`ApplyTimestampRules`](crate::kits::speech::whisper::logit_filters::ApplyTimestampRules)).
//! - [`TokenSearch`]: the search seam. It owns the sequences' bookkeeping, and
//!   tells the loop how to permute its cache's rows when beams branch.
//!   [`GreedySearch`] is the argmax, or sampling above temperature zero;
//!   [`BeamSearch`] is a beam search with patience.
//! - [`SequenceRanker`]: which finished candidate of a group the decode
//!   returns; [`MaximumLikelihoodRanker`] normalizes by length.
//! - [`SearchConfig`]: the value object a model's decode config embeds, which
//!   picks the search and the ranker.
//! - [`TemperatureLadder`]: re-decoding at higher temperatures until a model's
//!   judge passes a result.
//!
//! [`speech::whisper::decode`](crate::kits::speech::whisper::decode) runs
//! its own loop over these parts (its cache is grouped across beams);
//! [`gpts::qwen3`](crate::kits::gpts::qwen3) is a [`CachedStep`], decoded by
//! [`decode_cached`].

mod beam;
mod cached_decode;
mod filters;
mod greedy;
mod ranker;
mod search;
mod temperature;

pub use beam::*;
pub use cached_decode::*;
pub use filters::*;
pub use greedy::*;
pub use ranker::*;
pub use search::*;
pub use temperature::*;
