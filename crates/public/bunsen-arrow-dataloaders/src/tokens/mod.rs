//! Tokenization and dense-block packing utilities.
//!
//! * [`tokenize_text_batches`](crate::tokens::tokenize_text_batches) adapts an
//!   iterator of text batches into an iterator of token-id batches using a
//!   [`wordchipper::Tokenizer`].
//! * [`DenseTokenBlockBatcher`](crate::tokens::DenseTokenBlockBatcher) (and its
//!   convenience wrappers
//!   [`DenseTokenBlocksOptions`](crate::tokens::DenseTokenBlocksOptions) and
//!   [`compact_dense_token_blocks`](crate::tokens::compact_dense_token_blocks))
//!   pack variable-length token sequences into fixed-shape `batch_size x
//!   batch_seq_len` blocks, optionally bracketing each sequence with
//!   beginning-of-sequence and end-of-sequence token markers.

mod dense_blocks;
mod tokenize_adapter;

pub use dense_blocks::*;
pub use tokenize_adapter::*;
