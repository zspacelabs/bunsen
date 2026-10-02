//! # LSTM blocks
//!
//! - [`FusedLstm`], built by [`FusedLstmConfig`], is a single-step LSTM cell
//!   with fused gates: one input projection and one recurrent projection, each
//!   to all four gates at once. [`step`](FusedLstm::step) takes the next
//!   `[batch, d_model]` frame and the previous `(hidden, cell)` and returns the
//!   next pair. The Silero VAD kit is built on it.
//! - [`ExtLstmState`] bundles the `cell` and `hidden` tensors, with helpers
//!   to slice, stack and reshape them together. It stands in for burn's
//!   [`LstmState`](burn::nn::LstmState), pending
//!   <https://github.com/tracel-ai/burn/pull/5167>, and converts to and from
//!   it. [`OptionalInitialLstmState`] turns a missing state into a zero one.

mod ext_lstm_state;
mod fused_lstm;

pub use ext_lstm_state::*;
pub use fused_lstm::*;
