//! # LSTM blocks
//!
//! - [`FusedLstm`], built by [`FusedLstmConfig`], is a single-step LSTM cell
//!   with fused gates: one input projection and one recurrent projection, each
//!   to all four gates at once. [`step`](FusedLstm::step) takes the next
//!   `[batch, d_model]` frame and the previous `(hidden, cell)` and returns the
//!   next pair. The Silero VAD kit is built on it.

mod fused_lstm;

pub use fused_lstm::*;
