//! Whisper Model.
//!
//! [Whisper][w] is a large-scale, general-purpose speech recognition model.
//!
//! The kit has the model ([`blocks`]), its audio front end ([`mel`]) with
//! the clamp policy that decides a window's floor ([`clamp`]), a chunked
//! greedy decode over it ([`decode`]), and the token layer under that: the
//! special-id layout and [`TokenPolicy`] ([`tokens`]), the `.tiktoken` rank
//! file ([`vocab`]), and the vocabulary table for [`kits::tokens`]
//! ([`text`]). Ids need no dependency; text is behind the `tokenizer`
//! feature.
//!
//! [w]: https://github.com/openai/whisper
//! [`blocks`]: crate::kits::speech::whisper::blocks
//! [`clamp`]: crate::kits::speech::whisper::clamp
//! [`mel`]: crate::kits::speech::whisper::mel
//! [`decode`]: crate::kits::speech::whisper::decode
//! [`TokenPolicy`]: crate::kits::speech::whisper::tokens::TokenPolicy
//! [`tokens`]: crate::kits::speech::whisper::tokens
//! [`vocab`]: crate::kits::speech::whisper::vocab
//! [`text`]: crate::kits::speech::whisper::text
//! [`kits::tokens`]: crate::kits::tokens

pub mod blocks;
pub mod clamp;
pub mod decode;
pub mod mel;
pub mod pretrained;
pub mod text;
pub mod tokens;
pub mod vocab;

#[doc(inline)]
pub use blocks::{
    Whisper,
    WhisperApiConfig,
    WhisperMeta,
    WhisperStructuralConfig,
};
#[doc(inline)]
pub use clamp::*;
#[doc(inline)]
pub use decode::*;
#[doc(inline)]
pub use mel::*;
#[doc(inline)]
pub use tokens::*;
#[doc(inline)]
pub use vocab::*;
