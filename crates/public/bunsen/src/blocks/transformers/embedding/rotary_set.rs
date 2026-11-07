//! A stack's rotary tables.

use alloc::vec::Vec;
use core::ops::Range;

use burn::module::Module;

use crate::blocks::transformers::embedding::{
    RotaryEmbedding,
    RotaryEmbeddingMeta,
};

/// The rotary tables of a stack's layers: one or more tables, and which one
/// each layer applies.
///
/// Most stacks have one table, which every layer shares (`From` a
/// [`RotaryEmbedding`]); Gemma 3's has two, a local one for its
/// sliding-window layers and a global one for its full-attention layers.
/// Every table covers the same positions.
///
/// Made by [`DecoderStack::rotary`](crate::blocks::transformers::decoders::stack::DecoderStack::rotary).
#[derive(Module, Debug)]
pub struct RotarySet {
    /// The tables; the first is the default.
    pub tables: Vec<RotaryEmbedding>,

    /// Each layer's table, by index into [`tables`](Self::tables); a layer
    /// past the end applies the first.
    #[module(skip)]
    pub layer_tables: Vec<usize>,
}

impl From<RotaryEmbedding> for RotarySet {
    fn from(table: RotaryEmbedding) -> Self {
        Self {
            tables: vec![table],
            layer_tables: Vec::new(),
        }
    }
}

impl RotarySet {
    /// The positions every table covers.
    pub fn seq_len(&self) -> usize {
        self.tables[0].seq_len()
    }

    /// Layer `layer`'s table.
    pub fn for_layer(
        &self,
        layer: usize,
    ) -> &RotaryEmbedding {
        &self.tables[self.layer_tables.get(layer).copied().unwrap_or(0)]
    }

    /// Every table, clipped to `range`
    /// ([`RotaryEmbedding::clip_range`]).
    pub fn clip_range(
        &self,
        range: Range<usize>,
    ) -> Self {
        Self {
            tables: self
                .tables
                .iter()
                .map(|t| t.clip_range(range.clone()))
                .collect(),
            layer_tables: self.layer_tables.clone(),
        }
    }
}
