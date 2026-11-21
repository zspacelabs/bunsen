//! # Text encoders
//!
//! Bidirectional transformer encoders of text: token ids in, a normalized
//! hidden state for each token out, an item's padding masked by its length.
//! A model built on one (a classifier, a decision head) holds the encoder
//! and loads it from its own checkpoint's prefix.
//!
//! - [`modernbert`]: Answer.AI's `ModernBERT`, as `transformers`'
//!   `ModernBertModel` defines it.

pub mod modernbert;
