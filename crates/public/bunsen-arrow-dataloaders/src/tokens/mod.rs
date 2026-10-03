//! Tokenization and dense-block packing utilities.
//!
//! * [`tokenize_text_batches`] adapts an iterator of text batches into an
//!   iterator of token-id batches using a [`wordchipper::Tokenizer`].
//! * [`DenseTokenBlockBatcher`] (and its convenience wrappers
//!   [`DenseTokenBlocksOptions`] and [`compact_dense_token_blocks`]) pack
//!   variable-length token sequences into full `[batch_size, batch_seq_len]`
//!   blocks, optionally bracketing each sequence with beginning-of-sequence and
//!   end-of-sequence token markers.

mod dense_blocks;
mod tokenize_adapter;

pub use dense_blocks::*;
pub use tokenize_adapter::*;
