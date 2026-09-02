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
//! On top of those sits the stream driver ([`driver`]): one configured
//! baseline that pushes audio in and hands transcript out, with its clock
//! ([`clock`]), its emission policy ([`emission`]), and voice activity as a
//! gate over Silero's probabilities ([`gate`]) producing regions
//! ([`regions`]).
//!
//! [w]: https://github.com/openai/whisper
//! [`blocks`]: crate::kits::speech::whisper::blocks
//! [`clamp`]: crate::kits::speech::whisper::clamp
//! [`clock`]: crate::kits::speech::whisper::clock
//! [`driver`]: crate::kits::speech::whisper::driver
//! [`emission`]: crate::kits::speech::whisper::emission
//! [`gate`]: crate::kits::speech::whisper::va_filter
//! [`regions`]: crate::kits::speech::whisper::regions
//! [`mel`]: crate::kits::speech::whisper::mel
//! [`decode`]: crate::kits::speech::whisper::decode
//! [`TokenPolicy`]: crate::kits::speech::whisper::tokens::TokenPolicy
//! [`tokens`]: crate::kits::speech::whisper::tokens
//! [`vocab`]: crate::kits::speech::whisper::vocab
//! [`text`]: crate::kits::speech::whisper::text
//! [`kits::tokens`]: crate::kits::tokens

pub mod blocks;
pub mod decode;
pub mod driver;
pub mod pretrained;

mod clamp;
mod clock;
mod emission;
mod mel;
mod regions;
mod segments;
mod text;
mod tokens;
mod va_filter;
mod vocab;

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
pub use clock::*;
#[doc(inline)]
pub use decode::*;
#[doc(inline)]
pub use driver::*;
#[doc(inline)]
pub use emission::*;
#[doc(inline)]
pub use mel::*;
#[doc(inline)]
pub use regions::*;
#[doc(inline)]
pub use segments::*;
#[doc(inline)]
pub use text::*;
#[doc(inline)]
pub use tokens::*;
#[doc(inline)]
pub use va_filter::*;
#[doc(inline)]
pub use vocab::*;
